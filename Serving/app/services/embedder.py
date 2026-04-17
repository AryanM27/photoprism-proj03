import os
import numpy as np
import open_clip
import torch
from PIL import Image


class Embedder:
    def __init__(self, model_name: str = "clip-ViT-B-32", device_str: str = "auto", checkpoint_path: str = None):
        # Strip "clip-" prefix — open_clip uses "ViT-B-32" not "clip-ViT-B-32"
        # Matches Data team's CLIPEncoder exactly (same library, same pretrained weights)
        internal_name = model_name.removeprefix("clip-")

        # checkpoint_path takes priority — supports swapping in Milind's fine-tuned model
        # Must be a directory saved in open_clip compatible format
        load_path = checkpoint_path if checkpoint_path and os.path.isdir(checkpoint_path) else None

        self._model_name = model_name
        self._model, _, self._transform = open_clip.create_model_and_transforms(
            internal_name,
            pretrained=load_path if load_path else "openai",
        )
        self._tokenizer = open_clip.get_tokenizer(internal_name)
        self._model.eval()

        device = device_str
        if device_str == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._model = self._model.to(self._device)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            vec = self._model.encode_text(tokens)
        return self._normalize(vec.cpu().numpy().flatten())

    def embed_image(self, image: Image.Image) -> list[float]:
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            vec = self._model.encode_image(tensor)
        return self._normalize(vec.cpu().numpy().flatten())

    def _normalize(self, vec: np.ndarray) -> list[float]:
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()
