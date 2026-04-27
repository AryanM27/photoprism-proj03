import logging
import os

import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class Embedder:
    """CLIP text+image embedder backed by open_clip.

    Loads the base open_clip model (MODEL_NAME env var, default ViT-B-32).
    If CHECKPOINT_PATH points to a valid .pt file produced by Training
    (keys: model_state_dict, ...), that state is loaded on top of the base
    weights — giving the fine-tuned model at inference time.
    Falls back to the base pretrained model if the checkpoint is missing or
    invalid (logs a warning).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        device_str: str = "auto",
        checkpoint_path: str | None = None,
    ):
        if device_str == "auto":
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            _device = device_str
        self.device = torch.device(_device)

        # open_clip uses bare arch names e.g. "ViT-B-32", not "clip-ViT-B-32"
        # Keep original name for external reporting (health endpoint) to avoid
        # breaking monitors that expect "clip-ViT-B-32".
        internal_name = model_name.removeprefix("clip-")
        self._model_name = model_name

        # Fine-tuned model — used for image embedding only.
        self._model, _, self._transform = open_clip.create_model_and_transforms(
            internal_name, pretrained="openai"
        )
        self._tokenizer = open_clip.get_tokenizer(internal_name)
        self._model = self._model.to(self.device).eval()

        if checkpoint_path and os.path.isfile(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                state = ckpt.get("model_state_dict", ckpt)
                self._model.load_state_dict(state, strict=False)
                logger.info("Loaded semantic checkpoint: %s", checkpoint_path)
            except Exception as exc:
                logger.warning(
                    "Could not load semantic checkpoint %s: %s — using base weights",
                    checkpoint_path, exc,
                )
        else:
            if checkpoint_path:
                logger.warning(
                    "CHECKPOINT_PATH=%s not found — using base pretrained weights",
                    checkpoint_path,
                )
            else:
                logger.info("No checkpoint supplied — using base pretrained weights")

        # Base CLIP model — used for text query encoding only.
        # Fine-tuned models lose zero-shot text-image alignment; base weights
        # preserve the broad language-vision alignment from 400M training pairs.
        self._text_model, _, _ = open_clip.create_model_and_transforms(
            internal_name, pretrained="openai"
        )
        self._text_model = self._text_model.to(self.device).eval()
        logger.info("Loaded base CLIP model for text query encoding")

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            vec = self._text_model.encode_text(tokens)
        return self._postprocess(vec)

    def embed_image(self, image: Image.Image) -> list[float]:
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self._model.encode_image(tensor)
        return self._postprocess(vec)

    def _postprocess(self, vec: torch.Tensor) -> list[float]:
        arr = vec.cpu().numpy().flatten().astype("float32")
        norm = np.linalg.norm(arr)
        return (arr / norm).tolist() if norm > 0 else arr.tolist()
