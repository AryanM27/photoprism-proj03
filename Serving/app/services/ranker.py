import io
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


class AestheticRegressor(nn.Module):
    """ResNet18 scalar regressor — matches Milind's training architecture (resnet18_linear)."""

    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Linear(512, 1)
        self.model = base

    def forward(self, x):
        return self.model(x).squeeze(1)   # (batch,) scalar


_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class AestheticRanker:
    def __init__(self, device_str: str = "auto", checkpoint_path: str = None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            if device_str == "auto" else device_str
        )
        self.model = AestheticRegressor().to(self.device).eval()

        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])

    @torch.no_grad()
    def score_image_bytes(self, image_bytes: bytes) -> float:
        """Return aesthetic score (1.0–10.0) for raw image bytes."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0).to(self.device)
        score = self.model(tensor).item()
        return round(float(torch.clamp(torch.tensor(score), 1.0, 10.0).item()), 4)

    @torch.no_grad()
    def score_image_path(self, path: str) -> float:
        """Return aesthetic score for a local file path."""
        with open(path, "rb") as f:
            return self.score_image_bytes(f.read())

    def rerank(self, results: list[dict], image_bytes_map: dict[str, bytes]) -> list[dict]:
        """
        Re-rank search results by aesthetic score.

        Args:
            results: list of {"image_id": ..., "score": ..., "payload": ...}
            image_bytes_map: {image_id: raw_bytes} for each result

        Returns:
            Same results list, sorted by aesthetic score descending,
            with "aesthetic_score" added to each item.
        """
        for item in results:
            iid = item["image_id"]
            if iid in image_bytes_map:
                item["aesthetic_score"] = self.score_image_bytes(image_bytes_map[iid])
            else:
                item["aesthetic_score"] = 0.0

        return sorted(results, key=lambda x: x["aesthetic_score"], reverse=True)
