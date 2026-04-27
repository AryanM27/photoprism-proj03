"""
Patch existing Qdrant points with user_id and source_dataset from Postgres.

Idempotent — safe to re-run. No re-embedding needed; uses set_payload() to
update existing points in place.

Usage (from photoprism-proj03/Data/):
    python scripts/backfill_qdrant_payload.py

Requires env vars: QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, DATABASE_URL
"""
import hashlib
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 1000


def _stable_uint64(image_id: str) -> int:
    return int(hashlib.sha256(image_id.encode()).hexdigest(), 16) % (2 ** 63)


def main():
    from qdrant_client import QdrantClient
    from src.data_pipeline.db.session import SessionLocal
    from src.data_pipeline.db.models import Image

    qdrant = QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=int(os.environ["QDRANT_PORT"]),
    )
    collection = os.environ["QDRANT_COLLECTION"]

    db = SessionLocal()
    try:
        total = db.query(Image).filter(Image.embedding_status == "embedded").count()
        logger.info("Patching Qdrant payloads for %d embedded images...", total)

        offset = 0
        patched = 0
        skipped = 0

        while True:
            batch = (
                db.query(Image)
                .filter(Image.embedding_status == "embedded")
                .offset(offset)
                .limit(BATCH_SIZE)
                .all()
            )
            if not batch:
                break

            for image in batch:
                point_id = _stable_uint64(image.image_id)
                try:
                    qdrant.set_payload(
                        collection_name=collection,
                        payload={
                            "source_dataset": image.source_dataset,
                            "user_id": image.user_id,
                        },
                        points=[point_id],
                    )
                    patched += 1
                except Exception as exc:
                    logger.warning("Failed to patch %s: %s", image.image_id, exc)
                    skipped += 1

            offset += BATCH_SIZE
            logger.info("Progress: %d / %d (skipped %d)", offset, total, skipped)

        logger.info("Done. Patched %d points, skipped %d.", patched, skipped)

    finally:
        db.close()


if __name__ == "__main__":
    main()
