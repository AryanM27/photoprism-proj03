"""
backfill_worker.py — Celery task: re-embed an image with a new model version.

Consumes the backfill queue. For each image:
  1. Dispatch embed_image task (reuses the existing embedding pipeline)
  2. Update the ProcessingJob status to done
"""
import logging
from datetime import datetime, timezone

from src.data_pipeline.workers.celery_app import app
from src.data_pipeline.db.session import SessionLocal
from src.data_pipeline.db.models import Image, ProcessingJob
from src.data_pipeline.workers.embedding_worker import embed_image

logger = logging.getLogger(__name__)


@app.task(
    name="src.data_pipeline.workers.backfill_worker.reprocess_image",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def reprocess_image(self, image_id: str, model_version: str) -> dict:
    """Re-embed image_id using the embedding worker, update processing job status."""
    db = SessionLocal()
    try:
        image = db.query(Image).filter_by(image_id=image_id).first()
        if image is None:
            raise ValueError(f"Image {image_id} not found")

        # Reset embedding status so embedding_worker re-processes it
        image.embedding_status = "pending"
        image.model_version = None
        image.embedded_at = None

        job = (
            db.query(ProcessingJob)
            .filter_by(image_id=image_id, job_type="backfill", status="queued")
            .order_by(ProcessingJob.created_at.desc())
            .first()
        )
        if job:
            job.status = "running"

        db.commit()

        # Dispatch to embedding worker (reuses full encode→upsert pipeline).
        # Job stays at "running" — the embedding worker is responsible for marking it "done".
        try:
            embed_image.delay(image_id)
        except Exception as broker_exc:
            logger.warning(
                "embed_image.delay() failed for image_id=%s; resetting job to queued for retry. Error: %s",
                image_id, broker_exc,
            )
            if job:
                job.status = "queued"
                db.commit()
            raise

        logger.info("Backfill dispatched for %s (model=%s)", image_id, model_version)
        return {"status": "dispatched", "image_id": image_id, "model_version": model_version}

    except Exception as exc:
        db.rollback()
        logger.error("Backfill failed for %s: %s", image_id, exc)
        raise self.retry(exc=exc)
    finally:
        db.close()
