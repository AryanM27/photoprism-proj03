from unittest.mock import patch, MagicMock
import numpy as np


@patch("src.data_pipeline.workers.embedding_worker._get_store")
@patch("src.data_pipeline.workers.embedding_worker._get_encoder")
@patch("src.data_pipeline.workers.embedding_worker.get_s3_client")
@patch("src.data_pipeline.workers.embedding_worker.get_db_session")
def test_embed_image_happy_path(mock_session, mock_s3, mock_get_encoder, mock_get_store):
    """embed_image downloads image, encodes it, upserts to Qdrant, updates Postgres."""
    fake_vec = np.random.rand(512).astype("float32")
    mock_encoder = MagicMock()
    mock_encoder.encode_image.return_value = fake_vec
    mock_get_encoder.return_value = mock_encoder

    mock_store = MagicMock()
    mock_get_store.return_value = mock_store

    fake_session = MagicMock()
    fake_image = MagicMock()
    fake_image.image_id = "abc123"
    fake_image.storage_path = "raw/abc123.jpg"
    fake_image.aesthetic_score = None
    fake_session.query.return_value.filter_by.return_value.first.return_value = fake_image
    mock_session.return_value.__enter__ = MagicMock(return_value=fake_session)
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    from src.data_pipeline.workers.embedding_worker import embed_image
    embed_image.run("abc123")

    mock_encoder.encode_image.assert_called_once()
    mock_store.upsert.assert_called_once()
    assert fake_image.embedding_status == "embedded"
    assert fake_image.model_version is not None


@patch("src.data_pipeline.workers.embedding_worker._get_store")
@patch("src.data_pipeline.workers.embedding_worker._get_encoder")
@patch("src.data_pipeline.workers.embedding_worker.get_db_session")
def test_embed_image_missing_record_raises(mock_session, mock_get_encoder, mock_get_store):
    """embed_image raises ValueError when image_id not found in Postgres."""
    fake_session = MagicMock()
    fake_session.query.return_value.filter_by.return_value.first.return_value = None
    mock_session.return_value.__enter__ = MagicMock(return_value=fake_session)
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    from src.data_pipeline.workers.embedding_worker import embed_image
    import pytest
    with pytest.raises(ValueError, match="not found"):
        embed_image.run("missing_id")
