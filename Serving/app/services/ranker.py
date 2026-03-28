import io
import urllib.request

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


class NIMA(nn.Module):
    """Neural Image Assessment model — predicts aesthetic quality score (1-10)."""

    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(base.last_channel, 10),
            nn.Softmax(dim=1),
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_BINS = torch.arange(1, 11, dtype=torch.float32)  # quality buckets 1-10


class AestheticRanker:
    def __init__(self, device_str: str = "auto"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            if device_str == "auto" else device_str
        )
        self.model = NIMA().to(self.device).eval()
        self._bins = _BINS.to(self.device)

    @torch.no_grad()
    def score_image_bytes(self, image_bytes: bytes) -> float:
        """Return aesthetic score (1.0–10.0) for raw image bytes."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0).to(self.device)
        dist = self.model(tensor)           # (1, 10) probability distribution
        score = (dist * self._bins).sum().item()
        return round(score, 4)

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
