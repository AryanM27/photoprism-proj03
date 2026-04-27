from unittest.mock import patch, MagicMock
import numpy as np


@patch("src.data_pipeline.workers.embedding_worker._get_store")
@patch("src.data_pipeline.workers.embedding_worker.requests.post")
@patch("src.data_pipeline.workers.embedding_worker.get_db_session")
def test_embed_image_happy_path(mock_session, mock_post, mock_get_store):
    """embed_image calls serving-api /embed/image, upserts to Qdrant, updates Postgres."""
    fake_vec = np.random.rand(512).tolist()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embedding": fake_vec, "model_version": "clip-ViT-B-32"}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    mock_store = MagicMock()
    mock_get_store.return_value = mock_store

    fake_session = MagicMock()
    fake_image = MagicMock()
    fake_image.image_id = "abc123"
    fake_image.storage_path = "raw/abc123.jpg"
    fake_image.source_dataset = "user"
    fake_image.user_id = "admin"
    fake_meta = MagicMock()
    fake_meta.tags = ""
    fake_meta.captured_at = None
    fake_meta.aesthetic_score = None
    fake_session.query.return_value.filter_by.return_value.first.side_effect = [fake_image, fake_meta]
    mock_session.return_value.__enter__ = MagicMock(return_value=fake_session)
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    from src.data_pipeline.workers.embedding_worker import embed_image
    embed_image.run("abc123")

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "/embed/image" in call_kwargs[0][0]
    assert call_kwargs[1]["json"]["storage_path"] == "raw/abc123.jpg"

    mock_store.upsert.assert_called_once()
    assert fake_image.embedding_status == "embedded"
    assert fake_image.model_version == "clip-ViT-B-32"


@patch("src.data_pipeline.workers.embedding_worker._get_store")
@patch("src.data_pipeline.workers.embedding_worker.requests.post")
@patch("src.data_pipeline.workers.embedding_worker.get_db_session")
def test_embed_image_missing_record_raises(mock_session, mock_post, mock_get_store):
    """embed_image raises ValueError when image_id not found in Postgres."""
    fake_session = MagicMock()
    fake_session.query.return_value.filter_by.return_value.first.return_value = None
    mock_session.return_value.__enter__ = MagicMock(return_value=fake_session)
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    from src.data_pipeline.workers.embedding_worker import embed_image
    import pytest
    with pytest.raises(ValueError, match="not found"):
        embed_image.run("missing_id")

    mock_post.assert_not_called()
