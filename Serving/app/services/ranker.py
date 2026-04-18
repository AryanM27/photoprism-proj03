import io
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image


# ── Shared building blocks (copied from Training/src/aesthetic/model.py) ──────

class GeMPool2d(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.pow(1.0 / self.p)


class SEBlock(nn.Module):
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        reduced = max(dim // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(dim, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ── Model architectures ────────────────────────────────────────────────────────

class ResNet18LinearAestheticRegressor(nn.Module):
    """Simple ResNet18 scalar regressor — matches Training resnet18_linear."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


class MobileNetV3LargeFusionAestheticRegressor(nn.Module):
    """Multi-scale MobileNetV3-Large regressor — matches Training mobilenet_v3_large_fusion."""

    def __init__(self, pretrained: bool = True, hidden_dim: int = 512, dropout: float = 0.30):
        super().__init__()
        weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = tv_models.mobilenet_v3_large(weights=weights)
        self.features = backbone.features
        self.pool = GeMPool2d()

        self.mid_dim = 80
        self.final_dim = 960
        fused_dim = self.mid_dim + self.final_dim

        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.fusion_gate = SEBlock(fused_dim, reduction=16)

        self.proj = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mlp_block1 = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.mlp_block2 = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid_feat = None
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx == 10:
                mid_feat = x
        if mid_feat is None:
            raise RuntimeError("MobileNetV3 intermediate feature not captured.")
        mid_vec = self.pool(mid_feat).flatten(1)
        final_vec = self.pool(x).flatten(1)
        feat = torch.cat([mid_vec, final_vec], dim=1)
        feat = self.fusion_norm(feat)
        feat = self.fusion_gate(feat)
        feat = self.proj(feat)
        feat = self.mlp_block1(feat)
        feat = self.mlp_block2(feat)
        return self.out_head(feat).squeeze(-1)


def _build_model(model_type: str, hidden_dim: int = 512) -> nn.Module:
    if model_type == "mobilenet_v3_large_fusion":
        return MobileNetV3LargeFusionAestheticRegressor(hidden_dim=hidden_dim)
    return ResNet18LinearAestheticRegressor()   # default


# ── Transform ──────────────────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Ranker ─────────────────────────────────────────────────────────────────────

class AestheticRanker:
    def __init__(self, device_str: str = "auto", checkpoint_path: str = None,
                 model_type: str = "resnet18_linear"):
        if device_str == "auto":
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            _device = device_str
        self.device = torch.device(_device)

        # Infer hidden_dim from checkpoint so the model architecture matches
        # exactly what was used during training (avoids shape-mismatch errors).
        hidden_dim = 512  # default; overridden below if checkpoint is present
        ckpt = None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            # proj.0 is nn.Linear(fused_dim → hidden_dim); shape[0] is hidden_dim
            if "proj.0.weight" in state:
                hidden_dim = state["proj.0.weight"].shape[0]

        self.model = _build_model(model_type, hidden_dim=hidden_dim).to(self.device).eval()

        if ckpt is not None:
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)

    @torch.no_grad()
    def score_image_bytes(self, image_bytes: bytes) -> float:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0).to(self.device)
        score = self.model(tensor).item()
        return round(float(torch.clamp(torch.tensor(score), 1.0, 10.0).item()), 4)

    @torch.no_grad()
    def score_image_path(self, path: str) -> float:
        with open(path, "rb") as f:
            return self.score_image_bytes(f.read())

    def rerank(self, results: list[dict], image_bytes_map: dict[str, bytes]) -> list[dict]:
        for item in results:
            iid = item["image_id"]
            item["aesthetic_score"] = (
                self.score_image_bytes(image_bytes_map[iid])
                if iid in image_bytes_map else 0.0
            )
        return sorted(results, key=lambda x: x["aesthetic_score"], reverse=True)
