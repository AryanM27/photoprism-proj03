import hashlib
import logging
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)


def _stable_uint64(image_id: str) -> int:
    """Deterministic, collision-resistant non-negative int63 from an image_id string.

    Uses SHA-256 so the result is stable across Python interpreter restarts
    (unlike hash() which varies with PYTHONHASHSEED).
    Bounded to 2**63 for compatibility with signed 64-bit Qdrant point IDs.
    """
    return int(hashlib.sha256(image_id.encode()).hexdigest(), 16) % (2 ** 63)


class QdrantStore:
    """Manages Qdrant collection setup, upserts, and ANN search."""

    def __init__(self, host: str, port: int, collection: str):
        self._client = QdrantClient(host=host, port=port)
        self.collection = collection

    def ensure_collection(self, vector_size: int = 512) -> None:
        """Create collection if it does not exist (404 → create, other errors re-raised)."""
        try:
            self._client.get_collection(self.collection)
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
            else:
                raise

    def upsert(self, image_id: str, vector: np.ndarray, payload: dict) -> None:
        """Insert or replace a single embedding."""
        self._client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=_stable_uint64(image_id),
                vector=vector.tolist(),
                payload={**payload, "image_id": image_id},
            )],
        )
        logger.info("Upserted %s to Qdrant collection %s", image_id, self.collection)

    def search(self, query_vector: np.ndarray, top_k: int = 10,
               filter_: dict | None = None) -> list[dict]:
        """Return top-K nearest neighbours as list of {image_id, score, ...payload}."""
        hits = self._client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=filter_,
            with_payload=True,
        )
        return [
            {"image_id": h.payload.get("image_id"), "score": h.score, **h.payload}
            for h in hits
        ]
