import logging
import os

import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEncoder:
    VECTOR_DIM = 512

    def __init__(self, model_name: str = "clip-ViT-B-32", checkpoint_path: str | None = None):
        self.model_name = model_name
        internal_name = model_name.removeprefix("clip-")
        self._model, _, self._transform = open_clip.create_model_and_transforms(
            internal_name, pretrained="openai"
        )
        self._tokenizer = open_clip.get_tokenizer(internal_name)

        if checkpoint_path and os.path.isfile(checkpoint_path):
            # Workers run CPU-only; map_location="cpu" is intentional
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            incompatible = self._model.load_state_dict(state, strict=False)
            if set(incompatible.missing_keys) == set(self._model.state_dict().keys()):
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path!r} loaded but zero parameters matched — "
                    "wrong checkpoint file or architecture mismatch."
                )
            if incompatible.missing_keys:
                logger.info(
                    "Checkpoint loaded with %d missing keys (strict=False): %s",
                    len(incompatible.missing_keys), checkpoint_path,
                )
            else:
                logger.info("Loaded checkpoint: %s", checkpoint_path)
        elif checkpoint_path:
            logger.warning("CHECKPOINT_PATH=%s not found — using base pretrained weights", checkpoint_path)

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
