import torch
import torch.nn as nn
import torchvision.models as tv_models


class TinyAestheticRegressor(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)


class ResNet18LinearAestheticRegressor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


class ResNet18MLPAestheticRegressor(nn.Module):
    def __init__(self, pretrained: bool = True, hidden_dim: int = 256):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        out = self.head(feats)
        return out.squeeze(-1)


def build_aesthetic_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "tiny_cnn")

    if model_type == "tiny_cnn":
        return TinyAestheticRegressor(
            embedding_dim=model_cfg.get("embedding_dim", 128)
        )

    if model_type == "resnet18_linear":
        return ResNet18LinearAestheticRegressor(
            pretrained=model_cfg.get("pretrained", True)
        )

    if model_type == "resnet18_mlp":
        return ResNet18MLPAestheticRegressor(
            pretrained=model_cfg.get("pretrained", True),
            hidden_dim=model_cfg.get("hidden_dim", 256),
        )

    raise ValueError(f"Unsupported aesthetic model type: {model_type}")