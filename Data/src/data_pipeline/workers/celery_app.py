"""
celery_app.py — Celery application instance shared by all workers.

Broker:  RabbitMQ  (RABBITMQ_URL env var)
Backend: RabbitMQ  (results not needed for fire-and-forget pipeline tasks)

Queues:
    ingestion  — raw image upload + Postgres insert
    validation — EXIF extraction, quality checks, normalisation
    backfill   — re-embed images with new model version
"""

import os
from celery import Celery
from celery.schedules import crontab

RABBITMQ_URL = os.environ["RABBITMQ_URL"]

_WORKER_NAME = os.environ.get("WORKER_NAME", "")
_METRICS_PORT = int(os.environ.get("METRICS_PORT", "0"))

if _WORKER_NAME and _METRICS_PORT:
    from src.data_pipeline.observability.celery_signals import register_signals
    register_signals(worker_name=_WORKER_NAME, metrics_port=_METRICS_PORT)

app = Celery(
    "data_pipeline",
    broker=RABBITMQ_URL,
    include=[
        "src.data_pipeline.workers.ingestion_worker",
        "src.data_pipeline.workers.validation_worker",
        "src.data_pipeline.workers.embedding_worker",
        "src.data_pipeline.workers.backfill_worker",
    ],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,           # ack only after task completes
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # one message at a time per worker
    task_routes={
        "src.data_pipeline.workers.ingestion_worker.process_ingestion_event": {
            "queue": "ingestion"
        },
        "src.data_pipeline.workers.validation_worker.process_validation_event": {
            "queue": "validation"
        },
        "src.data_pipeline.workers.embedding_worker.embed_image": {"queue": "embedding"},
        "src.data_pipeline.workers.backfill_worker.reprocess_image": {"queue": "backfill"},
        "src.data_pipeline.workers.backfill_worker.reconcile_backfill_queue": {"queue": "backfill"},
    },
    beat_schedule={
        "reconcile-backfill-queue": {
            "task": "src.data_pipeline.workers.backfill_worker.reconcile_backfill_queue",
            "schedule": crontab(minute="*/5"),
        },
    },
)
