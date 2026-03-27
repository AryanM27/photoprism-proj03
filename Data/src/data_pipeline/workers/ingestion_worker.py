"""
ingestion_worker.py — Celery task: consume ingestion queue events.

For each event:
  1. Upload image file to MinIO (raw/<id>.<ext>)
  2. Insert into Postgres `images` (status=pending) + `processing_jobs`
  3. Publish to validation queue
"""

import logging

from src.data_pipeline.workers.celery_app import app
from src.data_pipeline.ingestion.uploader import upload_image
from src.data_pipeline.db.session import SessionLocal

logger = logging.getLogger(__name__)


@app.task(
    name="src.data_pipeline.workers.ingestion_worker.process_ingestion_event",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def process_ingestion_event(self, event: dict) -> dict:
    """Process one ingestion event from the ingestion queue.

    Args:
        event: {image_id, file_path, source_dataset, split, message_id, timestamp}

    Returns:
        {image_id, storage_path, image_uri}
    """
    image_id = event.get("image_id", "<unknown>")
    logger.info(f"[ingestion] Processing {image_id}")

    try:
        db = SessionLocal()
        try:
            result = upload_image(event, db)
        finally:
            db.close()

        # Forward to validation queue via Celery
        from src.data_pipeline.workers.validation_worker import process_validation_event
        process_validation_event.delay({
            "image_id": result["image_id"],
            "storage_path": result["storage_path"],
        })

        logger.info(f"[ingestion] Done {image_id} → forwarded to validation queue")
        return result

    except Exception as exc:
        logger.error(f"[ingestion] Failed {image_id}: {exc}")
        raise self.retry(exc=exc)
