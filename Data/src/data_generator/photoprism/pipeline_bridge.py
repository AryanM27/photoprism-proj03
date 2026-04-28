"""
pipeline_bridge.py — After uploading to Photoprism, register the image in the ML pipeline.

Flow:
  1. Upload augmented bytes to S3 under {S3_PREFIX}/raw/{image_id}.jpg
  2. Dispatch a Celery task to the ingestion queue with storage_key pre-set,
     so the ingestion worker skips the S3 upload and goes straight to
     Postgres insert → validation → embedding → Qdrant.
"""
import hashlib
import io
import logging
import os
from functools import lru_cache

import boto3
from botocore.client import Config
from celery import Celery

logger = logging.getLogger(__name__)

BUCKET = os.environ.get("S3_BUCKET", "training-module-proj03")
S3_PREFIX = os.environ.get("S3_PREFIX", "data_arm9337")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")

# Minimal Celery app — only used to dispatch tasks, not to run workers.
_celery = Celery("datagen_bridge", broker=RABBITMQ_URL)
_celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
)


@lru_cache(maxsize=1)
def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        config=Config(signature_version="s3v4"),
    )


def compute_image_id(image_bytes: bytes, user_id: str = "") -> str:
    """Stable image ID scoped to the uploading user.

    Including user_id ensures two users uploading identical bytes get distinct
    rows in the images table and separate Qdrant points. Public-corpus images
    (no user_id) retain the content-only hash for backward compatibility.
    """
    key = image_bytes + user_id.encode() if user_id else image_bytes
    return hashlib.md5(key).hexdigest()


def register_upload(image_bytes: bytes, filename: str, user_id: str = "") -> str | None:
    """Upload bytes to S3 and dispatch to the ingestion Celery queue.

    Returns image_id on success, None on any failure. Never raises.
    """
    try:
        image_id = compute_image_id(image_bytes, user_id)
        storage_key = f"{S3_PREFIX}/raw/{image_id}.jpg"

        _s3_client().upload_fileobj(io.BytesIO(image_bytes), BUCKET, storage_key)
        logger.debug("Uploaded to S3: %s", storage_key)

        event = {
            "image_id": image_id,
            "storage_key": storage_key,
            "source_dataset": "user",
            "split": "train",
            "user_id": user_id,
        }
        _celery.send_task(
            "src.data_pipeline.workers.ingestion_worker.process_ingestion_event",
            args=[event],
            queue="ingestion",
        )
        logger.info("Dispatched ingestion task for image %s (user=%s)", image_id, user_id)
        return image_id

    except Exception as exc:
        logger.warning("pipeline_bridge: failed to register upload: %s", exc)
        return None
