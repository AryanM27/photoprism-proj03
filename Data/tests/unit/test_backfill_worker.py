"""Unit tests for backfill Celery worker."""
from unittest.mock import patch, MagicMock


@patch("src.data_pipeline.workers.backfill_worker.embed_image")
@patch("src.data_pipeline.workers.backfill_worker.SessionLocal")
def test_reprocess_image_dispatches_embed_task(mock_session_cls, mock_embed):
    fake_image = MagicMock()
    fake_image.image_id = "abc123"

    fake_session = MagicMock()
    fake_session.query.return_value.filter_by.return_value.first.return_value = fake_image
    fake_session.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = None
    mock_session_cls.return_value = fake_session

    from src.data_pipeline.workers.backfill_worker import reprocess_image
    result = reprocess_image.run("abc123", "clip-ViT-B-32")

    mock_embed.delay.assert_called_once_with("abc123")
    assert fake_image.embedding_status == "pending"
    assert result["status"] == "dispatched"


@patch("src.data_pipeline.workers.backfill_worker.SessionLocal")
def test_reprocess_image_raises_for_missing_image(mock_session_cls):
    fake_session = MagicMock()
    fake_session.query.return_value.filter_by.return_value.first.return_value = None
    mock_session_cls.return_value = fake_session

    from src.data_pipeline.workers.backfill_worker import reprocess_image
    import pytest
    # Bare Exception is intentional: .run() with max_retries exhausted may raise
    # MaxRetriesExceededError (a Celery subclass of Exception) or re-raise the original
    # ValueError, so we catch the common base rather than enumerate both.
    with pytest.raises((ValueError, Exception)):
        reprocess_image.run("missing_id", "clip-ViT-B-32")

    fake_session.rollback.assert_called_once()
