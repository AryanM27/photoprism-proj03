import torch
import torch.nn as nn
import torchvision.models as tv_models
import torch.nn.functional as F


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
        w = self.fc(x)
        return x * w


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


class ResNet18FusionMLPAestheticRegressor(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.35,
    ):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.pool = GeMPool2d()

        # layer3 output channels = 256
        # layer4 output channels = 512
        layer3_dim = 256
        layer4_dim = 512
        fused_dim = layer3_dim + layer4_dim

        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.fusion_gate = SEBlock(fused_dim, reduction=16)

        # Project fused backbone features into a richer hidden space
        self.proj = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Deeper residual head for stronger nonlinear modeling
        self.mlp_block1 = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.mlp_block2 = ResidualMLPBlock(hidden_dim, dropout=dropout)

        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x3 = self.backbone.layer3(x)
        x4 = self.backbone.layer4(x3)

        feat3 = self.pool(x3).flatten(1)
        feat4 = self.pool(x4).flatten(1)

        # Multi-level fusion: composition/textures + semantics
        feat = torch.cat([feat3, feat4], dim=1)
        feat = self.fusion_norm(feat)
        feat = self.fusion_gate(feat)

        feat = self.proj(feat)
        feat = self.mlp_block1(feat)
        feat = self.mlp_block2(feat)

        out = self.out_head(feat)
        return out.squeeze(-1)


class EfficientNetB0LinearAestheticRegressor(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = tv_models.efficientnet_b0(weights=weights)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        out = self.head(feat)
        return out.squeeze(-1)


class EfficientNetB2MLPAestheticRegressor(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        hidden_dim: int = 384,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = tv_models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = tv_models.efficientnet_b2(weights=weights)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        out = self.head(feat)
        return out.squeeze(-1)


class EfficientNetB2FusionAestheticRegressor(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        hidden_dim: int = 768,
        dropout: float = 0.35,
    ):
        super().__init__()
        weights = tv_models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = tv_models.efficientnet_b2(weights=weights)

        self.features = backbone.features
        self.pool = GeMPool2d()

        # EfficientNet-B2 approximate feature dims:
        # mid-stage chosen around feature block index 5 -> ~120 channels
        # final stage -> 1408 channels
        self.mid_dim = 120
        self.final_dim = 1408
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
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid_feat = None

        # Capture both intermediate and final visual representations
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx == 5:
                mid_feat = x

        final_feat = x

        if mid_feat is None:
            raise RuntimeError("Intermediate EfficientNet feature was not captured.")

        mid_vec = self.pool(mid_feat).flatten(1)
        final_vec = self.pool(final_feat).flatten(1)

        # Fuse multiscale features
        feat = torch.cat([mid_vec, final_vec], dim=1)
        feat = self.fusion_norm(feat)
        feat = self.fusion_gate(feat)

        feat = self.proj(feat)
        feat = self.mlp_block1(feat)
        feat = self.mlp_block2(feat)

        out = self.out_head(feat)
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
    
    if model_type == "resnet18_fusion_mlp":
        return ResNet18FusionMLPAestheticRegressor(
            pretrained=model_cfg.get("pretrained", True),
            hidden_dim=model_cfg.get("hidden_dim", 512),
            dropout=model_cfg.get("dropout", 0.35),
        )

    if model_type == "efficientnet_b0_linear":
        return EfficientNetB0LinearAestheticRegressor(
            pretrained=model_cfg.get("pretrained", True),
            dropout=model_cfg.get("dropout", 0.2),
        )

    if model_type == "efficientnet_b2_mlp":
        return EfficientNetB2MLPAestheticRegressor(
            pretrained=model_cfg.get("pretrained", True),
            hidden_dim=model_cfg.get("hidden_dim", 384),
            dropout=model_cfg.get("dropout", 0.3),
        )

    if model_type == "efficientnet_b2_fusion":
        return EfficientNetB2FusionAestheticRegressor(
            pretrained=model_cfg.get("pretrained", True),
            hidden_dim=model_cfg.get("hidden_dim", 768),
            dropout=model_cfg.get("dropout", 0.35),
        )

    raise ValueError(f"Unsupported aesthetic model type: {model_type}")