from unittest.mock import patch, MagicMock


@patch("src.data_pipeline.workers.embedding_worker.embed_image")
@patch("src.data_pipeline.workers.validation_worker.SessionLocal")
def test_validation_worker_dispatches_embedding_on_success(mock_session_cls, mock_embed):
    """After a valid image passes all checks, embed_image.delay is called with the image_id."""
    image_id = "abc123"

    fake_image = MagicMock()
    fake_image.image_id = image_id
    fake_image.storage_path = "raw/abc123.jpg"
    fake_image.source_dataset = "yfcc"

    fake_session = MagicMock()
    fake_session.get.return_value = fake_image
    fake_session.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = None
    mock_session_cls.return_value = fake_session

    with patch("src.data_pipeline.validation.checks.run_checks", return_value=(True, None)), \
         patch("src.data_pipeline.validation.normalizer.extract_metadata", return_value=MagicMock()):
        from src.data_pipeline.workers.validation_worker import process_validation_event
        process_validation_event.run({"image_id": image_id, "storage_path": "raw/abc123.jpg"})

    mock_embed.delay.assert_called_once_with(image_id)
    fake_session.commit.assert_called()
