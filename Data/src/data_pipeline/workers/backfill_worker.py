"""
backfill_worker.py — Celery task: re-embed an image with a new model version.

Consumes the backfill queue. For each image:
  1. Dispatch embed_image task (reuses the existing embedding pipeline)
  2. Update the ProcessingJob status to done
"""
import logging
import os
from datetime import datetime, timezone

import requests

from src.data_pipeline.workers.celery_app import app
from src.data_pipeline.db.session import SessionLocal
from src.data_pipeline.db.models import Image, ImageMetadata, ProcessingJob

SERVING_API_URL = os.environ.get("SERVING_API_URL", "http://serving-api:8000")

logger = logging.getLogger(__name__)

from src.data_pipeline.observability.celery_signals import register_signals

register_signals(worker_name="backfill", metrics_port=8004)


def _fetch_aesthetic_score(image_uri: str) -> float | None:
    """Call serving /score/aesthetic; return score in 0.0–1.0 range, or None on failure."""
    try:
        resp = requests.post(
            f"{SERVING_API_URL}/score/aesthetic",
            json={"s3_path": image_uri},
            timeout=30,
        )
        resp.raise_for_status()
        score_1_to_10 = float(resp.json()["aesthetic_score"])
        logger.info("Aesthetic score for %s: %.3f", image_uri, score_1_to_10)
        return round(score_1_to_10 / 10.0, 4)
    except Exception as exc:
        logger.warning("Aesthetic scoring failed for %s: %s", image_uri, exc)
        return None


@app.task(
    name="src.data_pipeline.workers.backfill_worker.reprocess_image",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def reprocess_image(self, image_id: str, model_version: str, aesthetic_score: float | None = None, reembed: bool = True) -> dict:
    db = SessionLocal()
    try:
        image = db.query(Image).filter_by(image_id=image_id).first()
        if image is None:
            raise ValueError(f"Image {image_id} not found")

        if reembed:
            image.embedding_status = "pending"
            image.model_version = None
            image.embedded_at = None

        # Auto-fetch aesthetic score for user uploads that don't have one yet
        if aesthetic_score is None and image.source_dataset == "user" and image.storage_path:
            s3_path = f"s3://{os.environ.get('S3_BUCKET', 'training-module-proj03')}/{image.storage_path}"
            aesthetic_score = _fetch_aesthetic_score(s3_path)

        if aesthetic_score is not None:
            image_metadata = db.query(ImageMetadata).filter_by(image_id=image_id).first()
            if image_metadata is not None:
                image_metadata.aesthetic_score = aesthetic_score
                image_metadata.aesthetic_model_version = model_version
                image_metadata.aesthetic_score_date = datetime.now(timezone.utc)
            else:
                logger.warning("No ImageMetadata row for image_id=%s; aesthetic_score not persisted.", image_id)

        job = (
            db.query(ProcessingJob)
            .filter_by(image_id=image_id, job_type="backfill", status="queued")
            .order_by(ProcessingJob.created_at.desc())
            .first()
        )
        if job:
            job.status = "running" if reembed else "done"

        db.commit()

        if reembed:
            try:
                from src.data_pipeline.workers.embedding_worker import embed_image
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
        return {"status": "dispatched", "image_id": image_id, "model_version": model_version, "reembed": reembed}

    except Exception as exc:
        db.rollback()
        logger.error("Backfill failed for %s: %s", image_id, exc)
        raise self.retry(exc=exc)
    finally:
        db.close()
