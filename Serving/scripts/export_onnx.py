import os
import torch
import torch.nn as nn
import open_clip
import numpy as np
import onnxruntime as ort


MODEL_NAME = os.getenv("MODEL_NAME", "ViT-B-32")
ONNX_TEXT_PATH = os.getenv("ONNX_TEXT_PATH", "/serving/onnx_models/text_encoder.onnx")
ONNX_IMAGE_PATH = os.getenv("ONNX_IMAGE_PATH", "/serving/onnx_models/image_encoder.onnx")


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, tokens):
        return self.model.encode_text(tokens)


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, image):
        return self.model.encode_image(image)


def export():
    print(f"Loading {MODEL_NAME}...")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained="openai")
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    os.makedirs("checkpoints", exist_ok=True)

    text_encoder = TextEncoder(model).eval()
    image_encoder = ImageEncoder(model).eval()

    # --- Export text encoder ---
    dummy_tokens = tokenizer(["a photo of a cat"])
    with torch.no_grad():
        torch.onnx.export(
            text_encoder,
            dummy_tokens,
            ONNX_TEXT_PATH,
            input_names=["tokens"],
            output_names=["embeddings"],
            dynamic_axes={"tokens": {0: "batch"}, "embeddings": {0: "batch"}},
            opset_version=18,
        )
    print(f"Text encoder exported to {ONNX_TEXT_PATH}")

    # --- Export image encoder ---
    dummy_image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            image_encoder,
            dummy_image,
            ONNX_IMAGE_PATH,
            input_names=["image"],
            output_names=["embeddings"],
            dynamic_axes={"image": {0: "batch"}, "embeddings": {0: "batch"}},
            opset_version=18,
        )
    print(f"Image encoder exported to {ONNX_IMAGE_PATH}")

    # --- Verify both exports ---
    print("\nVerifying exports...")

    sess_text = ort.InferenceSession(ONNX_TEXT_PATH, providers=["CPUExecutionProvider"])
    out_text = sess_text.run(None, {"tokens": dummy_tokens.numpy()})
    print(f"Text encoder output shape: {out_text[0].shape}")

    sess_image = ort.InferenceSession(ONNX_IMAGE_PATH, providers=["CPUExecutionProvider"])
    out_image = sess_image.run(None, {"image": dummy_image.numpy()})
    print(f"Image encoder output shape: {out_image[0].shape}")

    print("\nONNX export complete.")


if __name__ == "__main__":
    export()
