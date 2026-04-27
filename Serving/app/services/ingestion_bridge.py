"""
ingestion_bridge.py — Bridge user uploads from PhotoPrism into the MLOps pipeline.

Called by the /upload/notify endpoint after PhotoPrism saves files to its staging
directory. Uploads each image to S3, registers it in Postgres, and dispatches a
Celery validation task — all before PhotoPrism's imp.Start() moves files out of staging.
"""

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tiff"}

S3_BUCKET = os.environ.get("S3_BUCKET", "training-module-proj03")
S3_PREFIX = os.environ.get("S3_PREFIX", "data_arm9337")
RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "")


def _compute_image_id(user_id: str, filename: str) -> str:
    """Stable, user-scoped image ID: MD5 of '{user_id}/{filename}'."""
    return hashlib.md5(f"{user_id}/{filename}".encode()).hexdigest()


def _compute_split(image_id: str) -> str:
    """Deterministic train/val/test split from last hex digit of image_id.

    Mirrors Data/src/data_pipeline/ingestion/scanner.py::compute_split.
    0-2 → val, 3-5 → test, 6-f → train.
    """
    d = int(image_id[-1], 16)
    if d <= 2:
        return "val"
    if d <= 5:
        return "test"
    return "train"


def upload_file_to_s3(file_path: str, image_id: str, ext: str) -> str:
    """Upload file bytes to S3 and return the storage key.

    Reuses the cached _s3_client() from image_fetcher to avoid creating
    a new boto3 client per file.
    """
    from app.services.image_fetcher import _s3_client

    storage_key = f"{S3_PREFIX}/raw/{image_id}{ext}"
    s3 = _s3_client()
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, storage_key)
    logger.info("Uploaded %s → s3://%s/%s", Path(file_path).name, S3_BUCKET, storage_key)
    return storage_key


def register_in_postgres(image_id: str, storage_key: str, user_id: str, split: str) -> None:
    """Insert image and processing_job rows into Postgres.

    Idempotent: the images INSERT uses ON CONFLICT DO NOTHING so retries are safe.
    The S3 upload is non-atomic with the Postgres write, but on retry the S3 key is
    overwritten (safe) and the images insert is skipped — matching the pattern in
    Data/src/data_pipeline/ingestion/uploader.py.
    """
    from sqlalchemy import text
    from app.services.image_fetcher import _pg_engine

    image_uri = f"swift://{S3_BUCKET}/{storage_key}"
    now = datetime.now(timezone.utc)
    db_url = os.environ["DATABASE_URL"]
    engine = _pg_engine(db_url)

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO images
                    (image_id, image_uri, storage_path, source_dataset,
                     split, status, user_id, created_at, updated_at)
                VALUES
                    (:image_id, :image_uri, :storage_path, 'user',
                     :split, 'pending', :user_id, :now, :now)
                ON CONFLICT (image_id) DO NOTHING
            """),
            {
                "image_id": image_id,
                "image_uri": image_uri,
                "storage_path": storage_key,
                "split": split,
                "user_id": user_id,
                "now": now,
            },
        )
        conn.execute(
            text("""
                INSERT INTO processing_jobs
                    (job_id, image_id, job_type, status, created_at, updated_at)
                VALUES
                    (:job_id, :image_id, 'ingestion', 'queued', :now, :now)
            """),
            {"job_id": str(uuid.uuid4()), "image_id": image_id, "now": now},
        )
    logger.info("Registered image %s in Postgres (user=%s, split=%s)", image_id, user_id, split)


def dispatch_validation_task(image_id: str, storage_key: str) -> None:
    """Publish a process_validation_event Celery task to the validation queue.

    Uses Celery purely as a broker client (send_task by name string) so the
    serving-api does not need to import the data pipeline worker code.
    """
    from celery import Celery

    broker_app = Celery(broker=RABBITMQ_URL)
    broker_app.send_task(
        "src.data_pipeline.workers.validation_worker.process_validation_event",
        args=[{"image_id": image_id, "storage_path": storage_key}],
        queue="validation",
    )
    logger.info("Dispatched validation task for image %s", image_id)


def ingest_staged_files(user_id: str, staging_path: str) -> tuple[int, int]:
    """Upload, register, and dispatch validation for every image in staging_path.

    Returns (processed, failed) counts. Per-file errors are logged and counted
    as failures so a single bad file does not abort the whole batch.
    """
    processed = 0
    failed = 0

    for file_path in Path(staging_path).iterdir():
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        image_id = _compute_image_id(user_id, file_path.name)
        split = _compute_split(image_id)

        try:
            storage_key = upload_file_to_s3(str(file_path), image_id, ext)
            register_in_postgres(image_id, storage_key, user_id, split)
            dispatch_validation_task(image_id, storage_key)
            processed += 1
        except Exception as exc:
            logger.error("Failed to ingest %s for user %s: %s", file_path.name, user_id, exc)
            failed += 1

    return processed, failed
