"""
pipeline.py — Trigger backfill re-embedding for validated images.

trigger_backfill() queries all validated images and either:
  - dry_run=True:  returns the list of jobs without writing to DB or queuing
  - dry_run=False: inserts ProcessingJob rows and publishes to backfill queue via Publisher
"""
import logging
import uuid
from datetime import datetime, timezone

from src.data_pipeline.db.models import Image, ProcessingJob

logger = logging.getLogger(__name__)


def trigger_backfill(db_session, model_version: str, dry_run: bool = False) -> list[dict]:
    """Query all validated images and queue them for re-embedding.

    Args:
        db_session:    Active SQLAlchemy session.
        model_version: Embedding model version to re-embed with (e.g. "clip-ViT-B-32").
        dry_run:       If True, return jobs without writing to DB or publishing to queue.

    Returns:
        List of job dicts with keys: job_id, image_id, job_type, model_version.
    """
    images = db_session.query(Image).filter_by(status="validated").all()
    jobs = []

    for img in images:
        job = {
            "job_id":        str(uuid.uuid4()),
            "image_id":      img.image_id,
            "job_type":      "backfill",
            "model_version": model_version,
        }
        jobs.append(job)

        if not dry_run:
            db_session.add(ProcessingJob(
                job_id=job["job_id"],
                image_id=img.image_id,
                job_type="backfill",
                status="queued",
                created_at=datetime.now(timezone.utc),
            ))

    if not dry_run:
        db_session.commit()
        # Deferred import to avoid a circular import: pipeline.py → publisher.py would
        # otherwise create a cycle via the ingestion package's __init__.
        from src.data_pipeline.ingestion.publisher import Publisher
        # If Publisher.__enter__ raises (e.g. broker unreachable), jobs are already
        # committed with status="queued" but nothing will be published. Log a warning so
        # operators can detect and requeue them; then re-raise so the caller knows.
        try:
            with Publisher() as pub:
                for job in jobs:
                    pub.publish_backfill(job["image_id"], model_version)
        except Exception as pub_exc:
            logger.warning(
                "Jobs committed with status='queued' but publishing failed (%d jobs, model=%s): %s",
                len(jobs),
                model_version,
                pub_exc,
            )
            raise
        logger.info("Queued %d images for backfill (model=%s)", len(jobs), model_version)

    return jobs
