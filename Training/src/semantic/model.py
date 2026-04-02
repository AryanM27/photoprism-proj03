# import torch
# from torch import nn


# class TinySemanticModel(nn.Module):
#     """
#     Lightweight semantic training baseline for local validation.
#     Produces image/text embeddings in the same shared space.
#     """

#     def __init__(self, embedding_dim: int = 128):
#         super().__init__()

#         self.image_encoder = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(32, embedding_dim),
#         )

#         self.text_encoder = nn.Sequential(
#             nn.Linear(1, 64),
#             nn.ReLU(),
#             nn.Linear(64, embedding_dim),
#         )

#         self.logit_scale = nn.Parameter(torch.tensor(1.0))

#     def encode_image(self, images: torch.Tensor) -> torch.Tensor:
#         x = self.image_encoder(images)
#         return x / x.norm(dim=-1, keepdim=True)

#     def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
#         x = self.text_encoder(text_features)
#         return x / x.norm(dim=-1, keepdim=True)


# def build_text_features(texts):
#     """
#     Minimal deterministic scalar text feature for local training validation.
#     """
#     values = []
#     for text in texts:
#         score = sum(ord(ch) for ch in text) % 1000
#         values.append([score / 1000.0])

#     return torch.tensor(values, dtype=torch.float32)


import torch
from torch import nn
import torchvision.models as tv_models


class TinySemanticModel(nn.Module):
    """
    Lightweight semantic training baseline.
    Produces image/text embeddings in the same shared space.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, embedding_dim),
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        x = self.image_encoder(images)
        return x / x.norm(dim=-1, keepdim=True)

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        x = self.text_encoder(text_features)
        return x / x.norm(dim=-1, keepdim=True)


class ResNet18ProjectionSemanticModel(nn.Module):
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()

        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.image_backbone = backbone
        self.image_projection = nn.Linear(in_features, embedding_dim)

        self.text_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.image_backbone(images)
        x = self.image_projection(feats)
        return x / x.norm(dim=-1, keepdim=True)

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        x = self.text_encoder(text_features)
        return x / x.norm(dim=-1, keepdim=True)


class ResNet18MLPSemanticModel(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()

        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.image_backbone = backbone
        self.image_projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.image_backbone(images)
        x = self.image_projection(feats)
        return x / x.norm(dim=-1, keepdim=True)

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        x = self.text_encoder(text_features)
        return x / x.norm(dim=-1, keepdim=True)


def build_text_features(texts):
    """
    Minimal deterministic scalar text feature for baseline training.
    """
    values = []
    for text in texts:
        score = sum(ord(ch) for ch in text) % 1000
        values.append([score / 1000.0])

    return torch.tensor(values, dtype=torch.float32)


def build_semantic_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "tiny_cnn")

    if model_type == "tiny_cnn":
        return TinySemanticModel(
            embedding_dim=model_cfg.get("embedding_dim", 128)
        )

    if model_type == "resnet18_projection":
        return ResNet18ProjectionSemanticModel(
            embedding_dim=model_cfg.get("embedding_dim", 128),
            pretrained=model_cfg.get("pretrained", True),
        )

    if model_type == "resnet18_mlp":
        return ResNet18MLPSemanticModel(
            embedding_dim=model_cfg.get("embedding_dim", 128),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            pretrained=model_cfg.get("pretrained", True),
        )

    raise ValueError(f"Unsupported semantic model type: {model_type}")