import base64
import io

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


@patch("src.data_pipeline.features.online._StubStore")
@patch("src.data_pipeline.features.online._StubEncoder")
def test_search_endpoint_returns_results(mock_encoder_cls, mock_store_cls):
    import src.data_pipeline.features.online as _online
    _online._encoder = None
    _online._store = None
    from src.data_pipeline.features.online import app
    client = TestClient(app)

    mock_encoder = mock_encoder_cls.return_value
    mock_encoder.encode_text.return_value = np.random.rand(512).astype("float32")

    mock_store = mock_store_cls.return_value
    mock_store.search.return_value = [
        {"image_id": "abc", "score": 0.9, "aesthetic_score": 7.0}
    ]

    response = client.post("/search", json={"query": "a cat on a roof", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["image_id"] == "abc"


@patch("src.data_pipeline.features.online._StubStore")
@patch("src.data_pipeline.features.online._StubEncoder")
def test_search_empty_query_returns_422(mock_encoder_cls, mock_store_cls):
    import src.data_pipeline.features.online as _online
    _online._encoder = None
    _online._store = None
    from src.data_pipeline.features.online import app
    client = TestClient(app)
    response = client.post("/search", json={"query": "", "top_k": 5})
    assert response.status_code == 422


from PIL import Image as PILImage


def make_b64_image(w=64, h=64) -> str:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_features_endpoint_returns_200():
    from src.data_pipeline.features.online import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.post("/features", json={
        "image_id": "test_001",
        "image_b64": make_b64_image()
    })
    assert resp.status_code == 200


def test_features_response_has_required_fields():
    from src.data_pipeline.features.online import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.post("/features", json={
        "image_id": "test_002",
        "image_b64": make_b64_image()
    })
    data = resp.json()
    assert "image_id"        in data
    assert "embedding"       in data
    assert "aesthetic_score" in data
    assert "latency_ms"      in data
    assert len(data["embedding"]) == 512


def test_missing_image_returns_400():
    from src.data_pipeline.features.online import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.post("/features", json={"image_id": "test_003"})
    assert resp.status_code == 400
