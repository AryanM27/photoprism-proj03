import logging

import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEncoder:
    VECTOR_DIM = 512

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model_name = model_name
        internal_name = model_name.removeprefix("clip-")
        self._model, _, self._transform = open_clip.create_model_and_transforms(
            internal_name, pretrained="openai"
        )
        self._tokenizer = open_clip.get_tokenizer(internal_name)
        self._model.eval()

    def encode_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        tensor = self._transform(img).unsqueeze(0)
        with torch.no_grad():
            vec = self._model.encode_image(tensor)
        raw = vec.cpu().numpy().flatten()
        if len(raw) != self.VECTOR_DIM:
            logger.warning(
                "Model '%s' produced vector of dim %d; expected %d. Truncating/padding.",
                self.model_name, len(raw), self.VECTOR_DIM,
            )
        flat = raw[:self.VECTOR_DIM]
        norm = np.linalg.norm(flat)
        return flat / norm if norm > 0 else flat

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self._tokenizer([text])
        with torch.no_grad():
            vec = self._model.encode_text(tokens)
        raw = vec.cpu().numpy().flatten()
        if len(raw) != self.VECTOR_DIM:
            logger.warning(
                "Model '%s' produced vector of dim %d; expected %d. Truncating/padding.",
                self.model_name, len(raw), self.VECTOR_DIM,
            )
        flat = raw[:self.VECTOR_DIM]
        norm = np.linalg.norm(flat)
        return flat / norm if norm > 0 else flat
