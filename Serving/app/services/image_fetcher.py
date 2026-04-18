"""Fetch image bytes from S3 object storage for reranking."""
import io
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _s3_client():
    import boto3
    from botocore.client import Config
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
    )


@lru_cache(maxsize=1)
def _pg_engine(db_url: str):
    """Return a cached SQLAlchemy engine for the given database URL.

    Cached so repeated Postgres fallback lookups across requests reuse
    the same connection pool instead of creating a new one per call.
    """
    from sqlalchemy import create_engine
    return create_engine(db_url, pool_pre_ping=True)


def fetch_image_bytes(storage_path: str) -> bytes:
    """Download image bytes from S3. Raises on failure."""
    bucket = os.environ.get("S3_BUCKET", "training-module-proj03")
    s3 = _s3_client()
    buf = io.BytesIO()
    s3.download_fileobj(bucket, storage_path, buf)
    return buf.getvalue()


def resolve_storage_path(image_id: str, payload: dict) -> str | None:
    """Return the S3 storage_path for an image.

    Tries payload first (fast path for new embeddings).
    Falls back to Postgres lookup for older Qdrant points.
    Returns None if the path cannot be determined.
    """
    # Fast path — payload has storage_path (new embeddings)
    sp = payload.get("storage_path")
    if sp:
        return sp

    # Fallback — look up in Postgres
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.warning("DATABASE_URL not set; cannot resolve storage_path for %s", image_id)
        return None
    try:
        from sqlalchemy import text
        engine = _pg_engine(db_url)
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT storage_path FROM images WHERE image_id = :id"),
                {"id": image_id},
            ).fetchone()
        return row[0] if row else None
    except Exception as exc:
        logger.warning("Postgres lookup failed for %s: %s", image_id, exc)
        return None
