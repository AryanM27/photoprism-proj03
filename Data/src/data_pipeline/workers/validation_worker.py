"""
validation_worker.py — Celery task: consume validation queue events.

For each event:
  1. Download image from MinIO
  2. Run quality checks (file integrity, min resolution, format, corruption)
  3. Extract + normalise EXIF metadata
  4. Write to Postgres `image_metadata`
  5. Update `images.status` → validated / failed
  6. Update `processing_jobs.status` → done / failed
  7. For user uploads: call serving /score/aesthetic and store the score

Validation logic lives in:
    src.data_pipeline.validation.checks     (Task 9)
    src.data_pipeline.validation.normalizer (Task 9)
"""

import logging
import os

import requests

from src.data_pipeline.workers.celery_app import app
from src.data_pipeline.db.session import SessionLocal
from src.data_pipeline.db.models import Image, ProcessingJob

SERVING_API_URL = os.environ.get("SERVING_API_URL", "http://serving-api:8000")
AESTHETIC_MODEL_VERSION = os.environ.get("AESTHETIC_MODEL_VERSION", "mobilenet_v3_large_fusion")

logger = logging.getLogger(__name__)



@app.task(
    name="src.data_pipeline.workers.validation_worker.process_validation_event",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def process_validation_event(self, event: dict) -> dict:
    """Process one validation event from the validation queue.

    Args:
        event: {image_id, storage_path, message_id, timestamp}

    Returns:
        {image_id, status}
    """
    image_id = event.get("image_id", "<unknown>")
    storage_path = event.get("storage_path", "")
    logger.info(f"[validation] Processing {image_id}")

    try:
        # Import lazily so the worker starts even if validation modules are missing
        from src.data_pipeline.validation.checks import run_checks
        from src.data_pipeline.validation.normalizer import extract_metadata
    except ImportError:
        logger.warning("[validation] checks/normalizer not yet implemented — skipping")
        return {"image_id": image_id, "status": "skipped"}

    db = SessionLocal()
    try:
        passed, reason = run_checks(storage_path)

        if not passed:
            _mark_failed(image_id, reason)
            logger.warning(f"[validation] FAILED {image_id}: {reason}")
            return {"image_id": image_id, "status": "failed"}

        image = db.get(Image, image_id)
        source_dataset = image.source_dataset if image else None

        metadata = extract_metadata(storage_path, image_id, source_dataset)

        if source_dataset == "user" and image and image.storage_path:
            s3_path = f"s3://{os.environ.get('S3_BUCKET', 'training-module-proj03')}/{image.storage_path}"
            _score_user_upload(s3_path, metadata)
            _caption_user_upload(image.storage_path, metadata)

        db.add(metadata)
        if image:
            image.status = "validated"

        job = (
            db.query(ProcessingJob)
            .filter_by(image_id=image_id, job_type="ingestion")
            .order_by(ProcessingJob.created_at.desc())
            .first()
        )
        if job:
            job.status = "done"

        db.commit()
        logger.info(f"[validation] OK {image_id}")
        from src.data_pipeline.workers.embedding_worker import embed_image
        embed_image.delay(image_id)
        return {"image_id": image_id, "status": "validated"}

    except Exception as exc:
        db.rollback()
        # Open a fresh session for failure recording so a broken connection
        # from the exception above doesn't prevent status being written.
        _mark_failed(image_id, str(exc))
        logger.error(f"[validation] Error {image_id}: {exc}")
        raise self.retry(exc=exc)
    finally:
        db.close()


def _score_user_upload(image_uri: str, metadata) -> None:
    """Call serving /score/aesthetic and write score onto metadata (in-place).

    aesthetic_score in DB is stored 0.0–1.0; serving returns 1–10.
    Failures are logged and silently ignored so validation still succeeds.
    """
    from datetime import datetime
    try:
        resp = requests.post(
            f"{SERVING_API_URL}/score/aesthetic",
            json={"s3_path": image_uri},
            timeout=30,
        )
        resp.raise_for_status()
        score_1_to_10 = float(resp.json()["aesthetic_score"])
        metadata.aesthetic_score = round(score_1_to_10 / 10.0, 4)
        metadata.aesthetic_score_date = datetime.utcnow()
        metadata.aesthetic_model_version = AESTHETIC_MODEL_VERSION
        logger.info(f"[validation] aesthetic score for {image_uri}: {score_1_to_10:.3f}")
    except Exception as exc:
        logger.warning(f"[validation] aesthetic scoring failed for {image_uri}: {exc}")


def _caption_user_upload(storage_path: str, metadata) -> None:
    """Call serving /caption/image and write the generated caption onto metadata.text.

    Failures are logged and silently ignored so validation still succeeds.
    """
    try:
        resp = requests.post(
            f"{SERVING_API_URL}/caption/image",
            json={"storage_path": storage_path},
            timeout=60,
        )
        resp.raise_for_status()
        metadata.text = resp.json()["caption"]
        logger.info(f"[validation] caption for {storage_path}: {metadata.text}")
    except Exception as exc:
        logger.warning(f"[validation] captioning failed for {storage_path}: {exc}")


def _mark_failed(image_id: str, reason: str) -> None:
    """Open a fresh session to record failure — isolated from any broken session."""
    db = SessionLocal()
    try:
        image = db.get(Image, image_id)
        if image:
            image.status = "failed"

        job = (
            db.query(ProcessingJob)
            .filter_by(image_id=image_id, job_type="ingestion")
            .order_by(ProcessingJob.created_at.desc())
            .first()
        )
        if job:
            job.status = "failed"
            job.error_message = reason

        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
