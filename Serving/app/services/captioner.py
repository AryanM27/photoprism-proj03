import logging

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

logger = logging.getLogger(__name__)


class Captioner:
    """BLIP image captioner backed by transformers AutoModelForImageTextToText."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device_str: str = "auto",
    ):
        if device_str == "auto":
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            _device = device_str
        self.device = torch.device(_device)
        self._model_name = model_name

        logger.info("Loading BLIP captioner: %s on %s", model_name, _device)
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = (
            AutoModelForImageTextToText.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        logger.info("BLIP captioner ready")

    @property
    def model_name(self) -> str:
        return self._model_name

    def caption(self, image: Image.Image, max_new_tokens: int = 30) -> str:
        inputs = self._processor(
            images=image.convert("RGB"), return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._processor.decode(out[0], skip_special_tokens=True).strip()
