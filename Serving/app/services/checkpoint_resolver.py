"""
checkpoint_resolver.py

Resolves model checkpoint paths at startup.  If the path looks like an S3
{bucket}/{key} reference the file is downloaded from Chameleon/S3-compatible
object storage and cached under /checkpoints/ so that container restarts do
not re-download the same file.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_CHECKPOINTS_DIR = "/checkpoints"


def resolve_checkpoint(path: str | None) -> str | None:
    """Return a local filesystem path to the checkpoint.

    Rules:
    - None / empty string  → return None
    - Starts with '/' → returns path if it exists, raises FileNotFoundError otherwise
    - Otherwise treat as '{bucket}/{key}' and download from S3
    """
    if not path:
        return None

    # Already a local absolute path that exists — use it directly.
    if path.startswith("/"):
        if os.path.isfile(path):
            logger.info("Checkpoint already present locally: %s", path)
            return path
        raise FileNotFoundError(
            f"Local checkpoint path '{path}' does not exist. "
            "Either provide a valid local path or an S3 '{bucket}/{key}' path."
        )

    # Treat as {bucket}/{key}
    parts = path.split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(
            f"Invalid checkpoint path '{path}'. "
            "Expected either an absolute local path or '{{bucket}}/{{key}}' format."
        )
    bucket, key = parts

    local_dest = os.path.join(_CHECKPOINTS_DIR, key.lstrip("/"))

    if os.path.isfile(local_dest):
        logger.info("Checkpoint cache hit — skipping download: %s", local_dest)
        return local_dest

    # Download from S3-compatible endpoint.
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    missing = [k for k, v in {
        "S3_ENDPOINT_URL": endpoint_url,
        "AWS_ACCESS_KEY_ID": aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
    }.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Cannot download checkpoint — missing env vars: {', '.join(missing)}"
        )

    try:
        import boto3  # imported lazily so missing dep surfaces clearly
    except ImportError as exc:
        raise ImportError("boto3 is required for S3 checkpoint downloads. Add it to requirements.txt.") from exc

    try:
        logger.info(
            "Downloading checkpoint s3://%s/%s → %s",
            bucket, key, local_dest,
        )
        Path(local_dest).parent.mkdir(parents=True, exist_ok=True)

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        s3.download_file(bucket, key, local_dest)
        logger.info("Checkpoint download complete: %s", local_dest)
        return local_dest
    except Exception as exc:
        logger.warning("Failed to download checkpoint s3://%s/%s: %s", bucket, key, exc)
        raise
