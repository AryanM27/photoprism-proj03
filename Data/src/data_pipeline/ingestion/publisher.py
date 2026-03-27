"""
publisher.py — Publish messages to RabbitMQ queues.

Queues:
    ingestion  — scanner → ingestion_worker
    validation — ingestion_worker → validation_worker
    backfill   — backfill/pipeline.py → backfill_worker

Message format (all queues):
    {
        "message_id": str (UUID),
        "timestamp":  ISO8601 str,
        "image_id":   str,
        ... queue-specific fields ...
    }
"""

import os
import uuid
import json
import logging
from datetime import datetime, timezone

import pika

logger = logging.getLogger(__name__)

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")

QUEUES = ["ingestion", "validation", "backfill"]


class Publisher:
    def __init__(self):
        self._conn = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        self._channel = self._conn.channel()
        for queue in QUEUES:
            self._channel.queue_declare(queue=queue, durable=True)

    def _publish(self, queue: str, payload: dict) -> None:
        payload.setdefault("message_id", str(uuid.uuid4()))
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._channel.basic_publish(
            exchange="",
            routing_key=queue,
            body=json.dumps(payload),
            properties=pika.BasicProperties(delivery_mode=2),  # persistent
        )
        logger.debug(f"Published to {queue}: {payload['message_id']}")

    def publish_ingestion(self, event: dict) -> None:
        """Publish a raw file event to the ingestion queue.

        Required keys: image_id, file_path, source_dataset, split
        """
        self._publish("ingestion", {
            "image_id": event["image_id"],
            "file_path": event["file_path"],
            "source_dataset": event["source_dataset"],
            "split": event["split"],
        })

    def publish_validation(self, image_id: str, storage_path: str) -> None:
        """Publish an uploaded image to the validation queue."""
        self._publish("validation", {
            "image_id": image_id,
            "storage_path": storage_path,
        })

    def publish_backfill(self, image_id: str, model_version: str) -> None:
        """Publish a reprocessing request to the backfill queue."""
        self._publish("backfill", {
            "image_id": image_id,
            "model_version": model_version,
        })

    def close(self) -> None:
        if self._conn and not self._conn.is_closed:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
