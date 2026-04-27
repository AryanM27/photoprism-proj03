"""Unit tests for ingestion_bridge helpers."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Minimal env so module-level os.environ.get calls don't error
os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("S3_PREFIX", "test-prefix")
os.environ.setdefault("RABBITMQ_URL", "amqp://guest:guest@localhost/")
os.environ.setdefault("DATABASE_URL", "postgresql://test/test")

# Stub heavy deps before importing the module under test.
# celery and sqlalchemy are imported lazily inside functions — stubbing them here
# means those lazy `from X import Y` calls get the MagicMock, not the real lib.
sys.modules.setdefault("app.services.image_fetcher", MagicMock())
sys.modules.setdefault("sqlalchemy", MagicMock())
sys.modules.setdefault("celery", MagicMock())

from app.services.ingestion_bridge import (  # noqa: E402
    _compute_image_id,
    _compute_split,
    ingest_staged_files,
    upload_file_to_s3,
    register_in_postgres,
    dispatch_validation_task,
)


# ---------------------------------------------------------------------------
# _compute_image_id
# ---------------------------------------------------------------------------

def test_compute_image_id_is_stable():
    assert _compute_image_id("u1", "photo.jpg") == _compute_image_id("u1", "photo.jpg")


def test_compute_image_id_is_hex_32():
    result = _compute_image_id("u1", "photo.jpg")
    assert len(result) == 32
    int(result, 16)  # raises if not valid hex


def test_compute_image_id_isolated_by_user():
    assert _compute_image_id("user1", "photo.jpg") != _compute_image_id("user2", "photo.jpg")


def test_compute_image_id_isolated_by_filename():
    assert _compute_image_id("u1", "a.jpg") != _compute_image_id("u1", "b.jpg")


# ---------------------------------------------------------------------------
# _compute_split
# ---------------------------------------------------------------------------

def test_compute_split_val():
    for d in "012":
        assert _compute_split("a" * 31 + d) == "val"


def test_compute_split_test():
    for d in "345":
        assert _compute_split("a" * 31 + d) == "test"


def test_compute_split_train():
    for d in "6789abcdef":
        assert _compute_split("a" * 31 + d) == "train"


# ---------------------------------------------------------------------------
# upload_file_to_s3
# ---------------------------------------------------------------------------

def test_upload_file_to_s3_returns_key(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"FAKE")

    # image_fetcher is already a MagicMock in sys.modules so _s3_client() returns
    # a MagicMock and upload_fileobj() is a no-op. Just verify the key is correct.
    key = upload_file_to_s3(str(img), "abc123", ".jpg")

    assert key == "test-prefix/raw/abc123.jpg"


# ---------------------------------------------------------------------------
# register_in_postgres
# ---------------------------------------------------------------------------

def test_register_in_postgres_executes_inserts():
    # sqlalchemy is stubbed in sys.modules; _pg_engine is on the MagicMock image_fetcher.
    # Configure the engine context manager so the `with engine.begin() as conn:` block works.
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    import app.services.image_fetcher as fetcher
    fetcher._pg_engine = MagicMock(return_value=mock_engine)

    register_in_postgres("img1", "prefix/raw/img1.jpg", "u1", "train")

    # Two INSERTs: images + processing_jobs
    assert mock_conn.execute.call_count == 2


# ---------------------------------------------------------------------------
# dispatch_validation_task
# ---------------------------------------------------------------------------

def test_dispatch_validation_task_sends_correct_task():
    # celery is stubbed in sys.modules; capture the Celery instance created inside the function
    # by configuring sys.modules["celery"].Celery to return a controllable mock.
    mock_celery_instance = MagicMock()
    import celery as celery_mod
    celery_mod.Celery = MagicMock(return_value=mock_celery_instance)

    dispatch_validation_task("img1", "prefix/raw/img1.jpg")

    mock_celery_instance.send_task.assert_called_once_with(
        "src.data_pipeline.workers.validation_worker.process_validation_event",
        args=[{"image_id": "img1", "storage_path": "prefix/raw/img1.jpg"}],
        queue="validation",
    )


# ---------------------------------------------------------------------------
# ingest_staged_files
# ---------------------------------------------------------------------------

def test_ingest_staged_files_processes_images(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(b"FAKE")
    (tmp_path / "readme.txt").write_bytes(b"skip me")

    with patch("app.services.ingestion_bridge.upload_file_to_s3", return_value="key") as mock_s3, \
         patch("app.services.ingestion_bridge.register_in_postgres") as mock_pg, \
         patch("app.services.ingestion_bridge.dispatch_validation_task") as mock_celery:
        processed, failed = ingest_staged_files("u1", str(tmp_path))

    assert processed == 1
    assert failed == 0
    mock_s3.assert_called_once()
    mock_pg.assert_called_once()
    mock_celery.assert_called_once()


def test_ingest_staged_files_counts_failures(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(b"FAKE")

    with patch("app.services.ingestion_bridge.upload_file_to_s3", side_effect=RuntimeError("s3 down")):
        processed, failed = ingest_staged_files("u1", str(tmp_path))

    assert processed == 0
    assert failed == 1


def test_ingest_staged_files_skips_non_images(tmp_path):
    (tmp_path / "document.pdf").write_bytes(b"PDF")
    (tmp_path / "archive.zip").write_bytes(b"ZIP")

    with patch("app.services.ingestion_bridge.upload_file_to_s3") as mock_s3:
        processed, failed = ingest_staged_files("u1", str(tmp_path))

    assert processed == 0
    assert failed == 0
    mock_s3.assert_not_called()
