import os
import pytest

@pytest.fixture(autouse=True)
def set_required_env_vars(monkeypatch):
    """Set env vars required by module-level imports before any test module is loaded."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/fake")
    monkeypatch.setenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.setenv("QDRANT_HOST", "localhost")
    monkeypatch.setenv("QDRANT_PORT", "6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "clip-ViT-B-32")
    # Replaced by Chameleon native S3 (CHI@TACC)
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret-key")
    monkeypatch.setenv("S3_BUCKET", "training-module-proj03")
