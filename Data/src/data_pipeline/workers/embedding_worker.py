"""
embedding_worker.py — Celery task: generate CLIP embedding and upsert to Qdrant.

For each event:
  1. Load image record from Postgres
  2. Download image from S3 object storage (Chameleon CHI@TACC) to a temp file
  3. Encode with CLIPEncoder (clip-ViT-B-32, 512-dim)
  4. Upsert vector + payload to Qdrant
  5. Update images.embedding_status, embedded_at, model_version in Postgres
"""
import hashlib
import logging
import os
import tempfile
from datetime import datetime, timezone

from src.data_pipeline.workers.celery_app import app
from src.data_pipeline.embeddings.encoder import CLIPEncoder
from src.data_pipeline.embeddings.qdrant_store import QdrantStore
from src.data_pipeline.db.models import Image, ImageMetadata

logger = logging.getLogger(__name__)


_encoder: CLIPEncoder | None = None
_store: QdrantStore | None = None


def _resolve_checkpoint() -> str | None:
    """Download checkpoint from S3 if needed, return local path or None."""
    path = os.environ.get("CHECKPOINT_PATH", "").strip()
    if not path:
        return None
    if path.startswith("/"):
        if os.path.isfile(path):
            return path
        logger.warning(
            "CHECKPOINT_PATH=%s is an absolute path but the file does not exist — "
            "check your volume mount. Using base weights.", path
        )
        return None
    # S3 bucket/key format — download once to /tmp, keyed by S3 path to avoid collisions
    key_hash = hashlib.sha256(path.encode()).hexdigest()[:12]
    local_dest = f"/tmp/clip_checkpoint_{key_hash}.pt"

    if os.path.isfile(local_dest):
        return local_dest
    try:
        parts = path.split("/", 1)
        if len(parts) != 2:
            logger.warning("CHECKPOINT_PATH '%s' is not a valid bucket/key path", path)
            return None
        bucket, key = parts
        logger.info("Downloading embedding checkpoint s3://%s/%s", bucket, key)
        get_s3_client().download_file(bucket, key, local_dest)
        logger.info("Embedding checkpoint downloaded to %s", local_dest)
        return local_dest
    except Exception as exc:
        logger.warning("Failed to download embedding checkpoint: %s — task will be rejected if CHECKPOINT_PATH is set", exc)
        return None


def _get_encoder() -> CLIPEncoder:
    global _encoder
    if _encoder is None:
        # EMBEDDING_MODEL is a version label (e.g. semantic-openclip-enhanced-vit-b32-v2).
        # CLIP_BASE_MODEL is the open_clip architecture to load (e.g. clip-ViT-B-32).
        base_model = os.environ.get("CLIP_BASE_MODEL", "clip-ViT-B-32")
        ckpt_path = _resolve_checkpoint()
        if ckpt_path is None and os.environ.get("CHECKPOINT_PATH", "").strip():
            # CHECKPOINT_PATH was set but could not be resolved — refuse to cache
            # a base-weight encoder so the next task attempt retries resolution.
            raise RuntimeError(
                "CHECKPOINT_PATH is set but checkpoint could not be resolved. "
                "Check S3 credentials and path. Refusing to embed with base weights."
            )
        _encoder = CLIPEncoder(model_name=base_model, checkpoint_path=ckpt_path)
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

        base_model = os.environ.get("EMBEDDING_MODEL", "clip-ViT-B-32")
        ckpt_path_env = os.environ.get("CHECKPOINT_PATH", "").strip()
        model_ver = (
            f"{base_model}+{os.path.basename(ckpt_path_env)}" if ckpt_path_env else base_model
        )
        payload = {
            "image_id": image_id,
            "storage_path": image.storage_path,
            "tags": [t.strip() for t in tags if t.strip()],
            "timestamp": timestamp,
            "aesthetic_score": float(aesthetic_score) if aesthetic_score else None,
            "model_version": model_ver,
            "checkpoint_path": ckpt_path_env or None,
            "source_dataset": image.source_dataset,
            "user_id": image.user_id,
        }
        _get_store().upsert(image_id, vector, payload)

        image.embedding_status = "embedded"
        image.embedded_at = datetime.now(timezone.utc)
        image.model_version = model_ver
        session.commit()
        logger.info("Embedded %s → Qdrant (%s)", image_id, model_ver)
