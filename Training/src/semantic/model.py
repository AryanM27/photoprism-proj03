import torch
from torch import nn


class TinySemanticModel(nn.Module):
    """
    Lightweight semantic training baseline for local validation.
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


def build_text_features(texts):
    """
    Minimal deterministic scalar text feature for local training validation.
    """
    values = []
    for text in texts:
        score = sum(ord(ch) for ch in text) % 1000
        values.append([score / 1000.0])

    return torch.tensor(values, dtype=torch.float32)