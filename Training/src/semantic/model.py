import torch
from torch import nn
import torchvision.models as tv_models
import open_clip


class OpenCLIPEnhancedSemanticModel(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        embed_dim: int = 512,
        proj_dim: int = 512,
        dropout: float = 0.2,
        unfreeze_last_n_blocks: int = 1,
    ):
        super().__init__()

        #load openclip
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        #freeze partially
        # for name, param in self.clip_model.named_parameters():
        #     if "visual" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        #freeze everything first
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze only the last few visual transformer blocks
        if hasattr(self.clip_model.visual, "transformer"):
            blocks = self.clip_model.visual.transformer.resblocks
            if unfreeze_last_n_blocks > 0:
                for block in blocks[-unfreeze_last_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True

        # Also unfreeze final visual norm / projection if present
        for attr in ["ln_post", "proj"]:
            if hasattr(self.clip_model.visual, attr):
                module_or_param = getattr(self.clip_model.visual, attr)
                if isinstance(module_or_param, nn.Module):
                    for param in module_or_param.parameters():
                        param.requires_grad = True
                elif isinstance(module_or_param, torch.nn.Parameter):
                    module_or_param.requires_grad = True


        self.visual = self.clip_model.visual
        self.text = self.clip_model.transformer

        #projection heads
        self.image_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        #learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, image):
        img_feat = self.clip_model.encode_image(image)
        img_feat = self.image_proj(img_feat)
        return nn.functional.normalize(img_feat, dim=-1)

    def encode_text(self, text):
        txt_feat = self.clip_model.encode_text(text)
        txt_feat = self.text_proj(txt_feat)
        return nn.functional.normalize(txt_feat, dim=-1)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        logit_scale = self.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return {
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "image_features": image_features,
            "text_features": text_features,
        }

class TinySemanticModel(nn.Module):

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
    
    if model_type == "openclip_enhanced":
        return OpenCLIPEnhancedSemanticModel(
            model_name=model_cfg.get("model_name", "ViT-B-32"),
            pretrained=model_cfg.get("pretrained", "laion2b_s34b_b79k"),
            embed_dim=model_cfg.get("embed_dim", 512),
            proj_dim=model_cfg.get("proj_dim", 512),
            dropout=model_cfg.get("dropout", 0.2),
            unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 1),
        )

    raise ValueError(f"Unsupported semantic model type: {model_type}")