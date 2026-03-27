"""
validation_worker.py — Celery task: consume validation queue events.

For each event:
  1. Download image from MinIO
  2. Run quality checks (file integrity, min resolution, format, corruption)
  3. Extract + normalise EXIF metadata
  4. Write to Postgres `image_metadata`
  5. Update `images.status` → validated / failed
  6. Update `processing_jobs.status` → done / failed

Validation logic lives in:
    src.data_pipeline.validation.checks     (Task 9)
    src.data_pipeline.validation.normalizer (Task 9)
"""

import logging

from src.data_pipeline.workers.celery_app import app
from src.data_pipeline.db.session import SessionLocal
from src.data_pipeline.db.models import Image, ProcessingJob

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
            _mark_failed(db, image_id, reason)
            logger.warning(f"[validation] FAILED {image_id}: {reason}")
            return {"image_id": image_id, "status": "failed"}

        metadata = extract_metadata(storage_path, image_id)
        db.add(metadata)

        image = db.get(Image, image_id)
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
        return {"image_id": image_id, "status": "validated"}

    except Exception as exc:
        db.rollback()
        db.close()
        # Open a fresh session for failure recording so a broken connection
        # from the exception above doesn't prevent status being written.
        _mark_failed(image_id, str(exc))
        logger.error(f"[validation] Error {image_id}: {exc}")
        raise self.retry(exc=exc)
    finally:
        # Guard: close only if not already closed above
        if db.is_active:
            db.close()


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
