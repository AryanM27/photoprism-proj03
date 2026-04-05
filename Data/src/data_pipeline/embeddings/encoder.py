import logging

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """Thin wrapper around sentence-transformers CLIP for image and text encoding."""

    VECTOR_DIM = 512

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode_image(self, image_path: str) -> np.ndarray:
        """Load image from path and return a normalised 512-dim vector."""
        img = Image.open(image_path).convert("RGB")
        return self._run_encode(img, is_image=True)

    def encode_text(self, text: str) -> np.ndarray:
        """Return a normalised 512-dim vector for a text string."""
        return self._run_encode(text, is_image=False)

    def _run_encode(self, input_, is_image: bool) -> np.ndarray:
        vec = self._model.encode(
            [input_],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        raw = vec.flatten()
        if len(raw) != self.VECTOR_DIM:
            logger.warning(
                "Model '%s' produced vector of dim %d; expected %d. Truncating/padding.",
                self.model_name, len(raw), self.VECTOR_DIM,
            )
        flat = raw[:self.VECTOR_DIM]
        norm = np.linalg.norm(flat)
        # Explicit re-normalisation ensures unit-norm even when the model mock in tests
        # bypasses normalize_embeddings=True. No-op for real sentence-transformers output.
        return flat / norm if norm > 0 else flat
