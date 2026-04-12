import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "clip-ViT-B-32", device_str: str = "auto", checkpoint_path: str = None):
        # checkpoint_path takes priority — supports swapping in a fine-tuned model without changing code.
        # It must be a directory saved in sentence-transformers format (model.safetensors + config).
        # Falls back to model_name (HuggingFace hub ID or local path) when not set.
        load_path = checkpoint_path if checkpoint_path and os.path.isdir(checkpoint_path) else model_name
        self._model_name = load_path
        self._model = SentenceTransformer(load_path)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        vec = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return self._normalize(vec.flatten())

    def embed_image(self, image: Image.Image) -> list[float]:
        vec = self._model.encode([image.convert("RGB")], convert_to_numpy=True, normalize_embeddings=True)
        return self._normalize(vec.flatten())

    def _normalize(self, vec: np.ndarray) -> list[float]:
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()
