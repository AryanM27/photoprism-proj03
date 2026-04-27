"""Unit tests for the /upload/notify route."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Env must be set before importing the app
os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("S3_PREFIX", "test-prefix")
os.environ.setdefault("RABBITMQ_URL", "amqp://guest:guest@localhost/")
os.environ.setdefault("DATABASE_URL", "postgresql://test/test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("QDRANT_COLLECTION", "test_collection")

# Stub heavy modules before importing anything from the serving app
for mod in [
    "torch", "torchvision", "open_clip", "transformers",
    "onnxruntime", "onnxscript",
    "prometheus_fastapi_instrumentator",
    "app.services.checkpoint_resolver",
    "app.services.captioner",
    "app.services.embedder",
    "app.services.embedder_onnx",
    "app.services.ranker",
    "app.services.feedback",
    "app.metrics",
]:
    sys.modules.setdefault(mod, MagicMock())

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.services.ingestion_bridge as bridge_mod
from app.routes.upload_notify import router, STORAGE_PATH

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Path-traversal guard
# ---------------------------------------------------------------------------

def test_path_traversal_rejected():
    resp = client.post("/upload/notify", json={
        "user_id": "u1",
        "staging_path": "/etc/passwd",
    })
    assert resp.status_code == 400
    assert "outside storage root" in resp.json()["detail"]


def test_path_traversal_dotdot_rejected(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with patch("app.routes.upload_notify.STORAGE_PATH", str(storage)):
        resp = client.post("/upload/notify", json={
            "user_id": "u1",
            "staging_path": str(outside),
        })
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Missing / non-directory staging path
# ---------------------------------------------------------------------------

def test_missing_staging_dir_rejected(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()
    staging = storage / "nonexistent"

    with patch("app.routes.upload_notify.STORAGE_PATH", str(storage)):
        resp = client.post("/upload/notify", json={
            "user_id": "u1",
            "staging_path": str(staging),
        })
    assert resp.status_code == 400
    assert "does not exist" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_upload_notify_ok(tmp_path):
    storage = tmp_path / "storage"
    staging = storage / "users" / "u1" / "token123"
    staging.mkdir(parents=True)
    (staging / "photo.jpg").write_bytes(b"FAKE")

    with patch("app.routes.upload_notify.STORAGE_PATH", str(storage)), \
         patch("app.routes.upload_notify.ingest_staged_files", return_value=(1, 0)) as mock_ingest:
        resp = client.post("/upload/notify", json={
            "user_id": "u1",
            "staging_path": str(staging),
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] == 1
    assert data["failed"] == 0
    mock_ingest.assert_called_once_with("u1", str(staging))


def test_upload_notify_partial_failure(tmp_path):
    storage = tmp_path / "storage"
    staging = storage / "users" / "u2" / "token456"
    staging.mkdir(parents=True)

    with patch("app.routes.upload_notify.STORAGE_PATH", str(storage)), \
         patch("app.routes.upload_notify.ingest_staged_files", return_value=(2, 1)):
        resp = client.post("/upload/notify", json={
            "user_id": "u2",
            "staging_path": str(staging),
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] == 2
    assert data["failed"] == 1
