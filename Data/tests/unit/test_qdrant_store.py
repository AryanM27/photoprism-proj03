from unittest.mock import patch, MagicMock
import numpy as np
from src.data_pipeline.embeddings.qdrant_store import QdrantStore

COLLECTION = "test_images"


def _make_store():
    with patch("src.data_pipeline.embeddings.qdrant_store.QdrantClient"):
        store = QdrantStore(host="localhost", port=6333, collection=COLLECTION)
        store._client = MagicMock()
        return store


def test_ensure_collection_creates_if_missing():
    from qdrant_client.http.exceptions import UnexpectedResponse
    store = _make_store()
    not_found = UnexpectedResponse(
        status_code=404, reason_phrase="Not Found",
        content=b"Not Found", headers={}
    )
    store._client.get_collection.side_effect = not_found
    store.ensure_collection(vector_size=512)
    store._client.create_collection.assert_called_once()


def test_ensure_collection_skips_if_exists():
    store = _make_store()
    store._client.get_collection.return_value = MagicMock()
    store.ensure_collection(vector_size=512)
    store._client.create_collection.assert_not_called()


def test_upsert_calls_client_with_correct_payload():
    store = _make_store()
    vec = np.random.rand(512).astype("float32")
    payload = {
        "image_id": "abc123",
        "aesthetic_score": 7.2,
        "tags": ["dog"],
        "timestamp": "2026-01-01",
        "model_version": "clip-ViT-B-32",
    }
    store.upsert("abc123", vec, payload)
    store._client.upsert.assert_called_once()
    args = store._client.upsert.call_args
    assert args.kwargs["collection_name"] == COLLECTION


def test_search_returns_list():
    store = _make_store()
    mock_hit = MagicMock()
    mock_hit.id = "abc123"
    mock_hit.score = 0.95
    mock_hit.payload = {"image_id": "abc123"}
    store._client.search.return_value = [mock_hit]
    results = store.search(np.random.rand(512).astype("float32"), top_k=5)
    assert isinstance(results, list)
    assert results[0]["image_id"] == "abc123"
    assert results[0]["score"] == 0.95
