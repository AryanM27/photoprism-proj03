"""
uploader.py — Upload a local image file to MinIO object storage and register it in Postgres.

Called by the ingestion worker after receiving an event from the `ingestion` queue.

Storage layout:
    s3://photoprism-proj03/raw/<image_id>.<ext>

Postgres writes:
    images       — INSERT (status=pending)
    processing_jobs — INSERT (job_type=ingestion, status=queued)
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.client import Config
from sqlalchemy.orm import Session

from src.data_pipeline.db.models import Image, ProcessingJob

logger = logging.getLogger(__name__)

BUCKET = os.environ.get("MINIO_BUCKET", "photoprism-proj03")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY = os.environ.get("MINIO_USER", "minioadmin")
S3_SECRET_KEY = os.environ.get("MINIO_PASSWORD", "minioadmin")


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def upload_image(event: dict, db: Session) -> dict:
    """Upload image to MinIO and write records to Postgres.

    Args:
        event: dict with keys image_id, file_path, source_dataset, split
        db: SQLAlchemy session

    Returns:
        dict with image_id, storage_path, image_uri
    """
    image_id = event["image_id"]
    file_path = Path(event["file_path"])
    source_dataset = event["source_dataset"]
    split = event["split"]

    ext = file_path.suffix.lower()
    storage_key = f"raw/{image_id}{ext}"
    image_uri = f"s3://{BUCKET}/{storage_key}"

    # Upload to MinIO
    s3 = _s3_client()
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, BUCKET, storage_key)
    logger.info(f"Uploaded {file_path.name} → {image_uri}")

    # Insert into images table (idempotent)
    # NOTE: S3 upload is non-atomic with Postgres insert. On retry, S3 key is
    # overwritten (safe) and the images insert is skipped — so retries are safe.
    now = datetime.now(timezone.utc)
    existing = db.get(Image, image_id)
    if existing is None:
        db.add(
            Image(
                image_id=image_id,
                image_uri=image_uri,
                storage_path=storage_key,
                source_dataset=source_dataset,
                split=split,
                status="pending",
                created_at=now,
                updated_at=now,
            )
        )
        # Only create a processing job for new images, not re-ingestions
        db.add(
            ProcessingJob(
                job_id=str(uuid.uuid4()),
                image_id=image_id,
                job_type="ingestion",
                status="queued",
                created_at=now,
                updated_at=now,
            )
        )

    db.commit()
    logger.info(f"Registered image {image_id} in Postgres (status=pending).")

    return {"image_id": image_id, "storage_path": storage_key, "image_uri": image_uri}
