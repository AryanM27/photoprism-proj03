"""
embedding_worker.py — Celery task: generate CLIP embedding and upsert to Qdrant.

For each event:
  1. Load image record from Postgres
  2. POST storage_path to serving-api /embed/image (serving-api fetches from S3 + runs CLIP)
  3. Upsert returned vector + payload to Qdrant
  4. Update images.embedding_status, embedded_at, model_version in Postgres
"""
import logging
import os
from datetime import datetime, timezone

import requests

from src.data_pipeline.workers.celery_app import app
from src.data_pipeline.embeddings.qdrant_store import QdrantStore
from src.data_pipeline.db.models import Image, ImageMetadata

logger = logging.getLogger(__name__)

SERVING_API_URL = os.environ.get("SERVING_API_URL", "http://serving-api")
CLIP_VECTOR_DIM = 512

_store: QdrantStore | None = None


def _get_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore(
            host=os.environ["QDRANT_HOST"],
            port=int(os.environ["QDRANT_PORT"]),
            collection=os.environ["QDRANT_COLLECTION"],
        )
        _store.ensure_collection(vector_size=CLIP_VECTOR_DIM)
    return _store


def get_db_session():
    from src.data_pipeline.db.session import SessionLocal
    from contextlib import contextmanager

    @contextmanager
    def _session():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    return _session()


@app.task(
    name="src.data_pipeline.workers.embedding_worker.embed_image",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def embed_image(self, image_id: str) -> None:
    """Call serving-api to embed image, then upsert vector to Qdrant."""
    with get_db_session() as session:
        image = session.query(Image).filter_by(image_id=image_id).first()
        if image is None:
            raise ValueError(f"Image {image_id} not found in Postgres")

        meta = session.query(ImageMetadata).filter_by(image_id=image_id).first()
        tags = (meta.tags or "").split(",") if meta else []
        timestamp = meta.captured_at.isoformat() if (meta and meta.captured_at) else ""
        aesthetic_score = getattr(meta, "aesthetic_score", None) if meta else None

        try:
            resp = requests.post(
                f"{SERVING_API_URL}/embed/image",
                json={"storage_path": image.storage_path},
                timeout=120,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error("serving-api /embed/image failed for %s: %s", image_id, exc)
            image.embedding_status = "failed"
            session.commit()
            raise self.retry(exc=exc)

        data = resp.json()
        vector = data["embedding"]
        model_ver = data["model_version"]

        payload = {
            "image_id": image_id,
            "storage_path": image.storage_path,
            "tags": [t.strip() for t in tags if t.strip()],
            "timestamp": timestamp,
            "aesthetic_score": float(aesthetic_score) if aesthetic_score else None,
            "model_version": model_ver,
            "source_dataset": image.source_dataset,
            "user_id": image.user_id,
        }
        _get_store().upsert(image_id, vector, payload)

        image.embedding_status = "embedded"
        image.embedded_at = datetime.now(timezone.utc)
        image.model_version = model_ver
        session.commit()
        logger.info("Embedded %s → Qdrant (%s)", image_id, model_ver)
