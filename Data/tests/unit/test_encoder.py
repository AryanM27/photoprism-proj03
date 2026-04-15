import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage
from src.data_pipeline.embeddings.encoder import CLIPEncoder


def _make_encoder(raw_vec: np.ndarray) -> CLIPEncoder:
    encoder = CLIPEncoder.__new__(CLIPEncoder)
    encoder.model_name = "clip-ViT-B-32"
    encoder.VECTOR_DIM = 512
    fake_model = MagicMock()
    fake_result = MagicMock()
    fake_result.cpu.return_value.numpy.return_value = raw_vec
    fake_model.encode_image.return_value = fake_result
    fake_model.encode_text.return_value = fake_result
    encoder._model = fake_model
    encoder._transform = MagicMock()
    encoder._tokenizer = MagicMock()
    return encoder


def test_encode_image_returns_512_dim_vector():
    raw = np.random.rand(1, 512).astype("float32")
    encoder = _make_encoder(raw)
    fake_img = MagicMock(spec=PILImage.Image)
    fake_img.convert.return_value = fake_img
    with patch("src.data_pipeline.embeddings.encoder.Image") as mock_pil:
        mock_pil.open.return_value = fake_img
        result = encoder.encode_image("/fake/img.jpg")
    mock_pil.open.assert_called_once_with("/fake/img.jpg")
    fake_img.convert.assert_called_once_with("RGB")
    assert result.shape == (512,)


def test_encode_text_returns_512_dim_vector():
    raw = np.random.rand(1, 512).astype("float32")
    encoder = _make_encoder(raw)
    result = encoder.encode_text("a dog running on grass")
    assert result.shape == (512,)


def test_encode_normalises_vector():
    raw = np.array([[3.0, 4.0] + [0.0] * 510], dtype="float32")
    encoder = _make_encoder(raw)
    result = encoder.encode_text("text")
    assert abs(np.linalg.norm(result) - 1.0) < 1e-5
