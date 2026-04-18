import io
import logging
import os
import random

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )


class S3ImageSource:
    """Lazily lists S3 keys once, then yields random images in a loop."""

    def __init__(self, bucket: str, prefix: str) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self._keys: list[str] = []
        self._listed: bool = False  # True after a successful listing attempt
        self._s3 = _get_s3_client()

    def _ensure_keys(self) -> None:
        """List keys on first call (lazy, cached). Raises RuntimeError if empty."""
        if self._listed:
            # Already listed; if empty, raise immediately without re-listing
            if not self._keys:
                raise RuntimeError(f"No images found in s3://{self.bucket}/{self.prefix}")
            return
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith((".jpg", ".jpeg", ".png")):
                    self._keys.append(key)
        self._listed = True
        logger.info(
            "S3ImageSource: found %d images under s3://%s/%s",
            len(self._keys),
            self.bucket,
            self.prefix,
        )
        if not self._keys:
            raise RuntimeError(f"No images found in s3://{self.bucket}/{self.prefix}")

    def random_image(self) -> tuple[bytes, str]:
        """Download a random image, return (bytes, original_s3_key_basename)."""
        self._ensure_keys()
        key = random.choice(self._keys)
        buf = io.BytesIO()
        self._s3.download_fileobj(self.bucket, key, buf)
        buf.seek(0)
        return buf.read(), os.path.basename(key)

    def __len__(self) -> int:
        self._ensure_keys()
        return len(self._keys)
