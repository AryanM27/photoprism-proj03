import hashlib
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "photoprism_images")
EMBEDDING_DIM = 512


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def upsert_photo(client: QdrantClient, image_id: str, embedding: list[float], payload: dict):
    # Use a stable integer ID derived from the image_id string
    point_id = int(hashlib.sha256(image_id.encode()).hexdigest(), 16) % (2**63)
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={"image_id": image_id, **payload},
            )
        ],
    )


def search_photos(
    client: QdrantClient,
    query_embedding: list[float],
    top_k: int = 10,
    user_id: str | None = None,
    score_threshold: float = 0.05,
) -> list[dict]:
    query_filter = None
    if user_id:
        query_filter = Filter(must=[
            FieldCondition(key="source_dataset", match=MatchValue(value="user")),
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ])
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=query_filter,
        score_threshold=score_threshold,
    ).points
    return [
        {"image_id": hit.payload.get("image_id"), "score": hit.score, "payload": hit.payload}
        for hit in results
    ]
