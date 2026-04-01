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

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")

app = Celery(
    "data_pipeline",
    broker=RABBITMQ_URL,
    include=[
        "src.data_pipeline.workers.ingestion_worker",
        "src.data_pipeline.workers.validation_worker",
        "src.data_pipeline.workers.embedding_worker",
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
    },
)
