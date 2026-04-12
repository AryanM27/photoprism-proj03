import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "clip-ViT-B-32", device_str: str = "auto", checkpoint_path: str = None):
        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        vec = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return self._normalize(vec.flatten())

    def embed_image(self, image: Image.Image) -> list[float]:
        vec = self._model.encode([image.convert("RGB")], convert_to_numpy=True, normalize_embeddings=True)
        return self._normalize(vec.flatten())

    def _normalize(self, vec: np.ndarray) -> list[float]:
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()
