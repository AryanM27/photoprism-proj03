"""
embedding_worker.py — Celery task: generate CLIP embedding and upsert to Qdrant.

For each event:
  1. Load image record from Postgres
  2. Download image from S3 object storage (Chameleon CHI@TACC) to a temp file
  3. Encode with CLIPEncoder (clip-ViT-B-32, 512-dim)
  4. Upsert vector + payload to Qdrant
  5. Update images.embedding_status, embedded_at, model_version in Postgres
"""
import logging
import os
import tempfile
from datetime import datetime, timezone

from src.data_pipeline.workers.celery_app import app
from src.data_pipeline.embeddings.encoder import CLIPEncoder
from src.data_pipeline.embeddings.qdrant_store import QdrantStore
from src.data_pipeline.db.models import Image, ImageMetadata

logger = logging.getLogger(__name__)

from src.data_pipeline.observability.celery_signals import register_signals

register_signals(worker_name="embedding", metrics_port=8003)

_encoder: CLIPEncoder | None = None
_store: QdrantStore | None = None


def _get_encoder() -> CLIPEncoder:
    global _encoder
    if _encoder is None:
        model = os.environ.get("EMBEDDING_MODEL", "clip-ViT-B-32")
        _encoder = CLIPEncoder(model_name=model)
    return _encoder


def _get_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore(
            host=os.environ["QDRANT_HOST"],
            port=int(os.environ["QDRANT_PORT"]),
            collection=os.environ["QDRANT_COLLECTION"],
        )
        _store.ensure_collection(vector_size=CLIPEncoder.VECTOR_DIM)
    return _store


def get_s3_client():
    # Replaced by Chameleon native S3 (CHI@TACC)
    import boto3
    from botocore.client import Config
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
    )


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
    """Download image from MinIO, generate CLIP embedding, upsert to Qdrant."""
    with get_db_session() as session:
        image = session.query(Image).filter_by(image_id=image_id).first()
        if image is None:
            raise ValueError(f"Image {image_id} not found in Postgres")

        meta = session.query(ImageMetadata).filter_by(image_id=image_id).first()
        tags = (meta.tags or "").split(",") if meta else []
        timestamp = meta.captured_at.isoformat() if (meta and meta.captured_at) else ""
        aesthetic_score = getattr(meta, "aesthetic_score", None) if meta else None

        s3 = get_s3_client()
        bucket = os.environ.get("S3_BUCKET", "training-module-proj03")  # Replaced by Chameleon native S3 (CHI@TACC)

        encode_error = None
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            s3.download_file(bucket, image.storage_path, tmp.name)
            try:
                vector = _get_encoder().encode_image(tmp.name)
            except Exception as exc:
                encode_error = exc

        if encode_error is not None:
            image.embedding_status = "failed"
            session.commit()
            raise self.retry(exc=encode_error)

        model_ver = os.environ.get("EMBEDDING_MODEL", "clip-ViT-B-32")
        payload = {
            "image_id": image_id,
            "tags": [t.strip() for t in tags if t.strip()],
            "timestamp": timestamp,
            "aesthetic_score": float(aesthetic_score) if aesthetic_score else None,
            "model_version": model_ver,
        }
        _get_store().upsert(image_id, vector, payload)

        image.embedding_status = "embedded"
        image.embedded_at = datetime.now(timezone.utc)
        image.model_version = model_ver
        session.commit()
        logger.info("Embedded %s → Qdrant (%s)", image_id, model_ver)
