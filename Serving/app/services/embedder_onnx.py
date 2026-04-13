import numpy as np
import open_clip
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class OnnxEmbedder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        text_onnx_path: str = "checkpoints/text_encoder.onnx",
        image_onnx_path: str = "checkpoints/image_encoder.onnx",
    ):
        self._model_name = model_name
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.text_session = ort.InferenceSession(
            text_onnx_path, providers=["CPUExecutionProvider"]
        )
        self.image_session = ort.InferenceSession(
            image_onnx_path, providers=["CPUExecutionProvider"]
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        tokens = self.tokenizer([text]).numpy()
        emb = self.text_session.run(None, {"tokens": tokens})[0]
        emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
        return emb[0].tolist()

    def embed_image(self, image: Image.Image) -> list[float]:
        tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).numpy()
        emb = self.image_session.run(None, {"image": tensor})[0]
        emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
        return emb[0].tolist()
