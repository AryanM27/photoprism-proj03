from typing import Dict, List, Optional

import open_clip
import torch
from torch.utils.data import DataLoader

from src.datasets.semantic_dataset import SemanticRetrievalDataset


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "image_ids": [item["image_id"] for item in batch],
        "texts": [item["text"] for item in batch],
        "image_tensors": torch.stack([item["image_tensor"] for item in batch]),
    }


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(model_name: str, device_str: str):
    device = get_device(device_str)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained="openai",
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.eval()

    return model, tokenizer, preprocess, device

def _build_mock_text_embedding(text: str, embedding_dim: int) -> torch.Tensor:
    base = sum(ord(ch) for ch in text) % 997
    values = torch.tensor(
        [((base + i * 13) % 101) / 100.0 for i in range(embedding_dim)],
        dtype=torch.float32,
    )
    return values / values.norm()


def _build_mock_image_embedding(image_tensor: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    flat_mean = float(image_tensor.mean().item())
    base = int(flat_mean * 1000) % 997
    values = torch.tensor(
        [((base + i * 17) % 103) / 100.0 for i in range(embedding_dim)],
        dtype=torch.float32,
    )
    return values / values.norm()


@torch.no_grad()
def generate_mock_embeddings(
    manifest_path: str,
    config: dict,
    embedding_dim: int = 512,
    batch_size: int = 2,
    split: Optional[str] = None,
) -> Dict:
    dataset = SemanticRetrievalDataset(manifest_path = manifest_path, config = config, image_size=config["model"]["image_size"], split = split,)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_image_embeddings = []
    all_text_embeddings = []
    all_image_ids = []
    all_texts = []

    for batch in loader:
        image_embeddings = []
        text_embeddings = []

        for image_tensor, text in zip(batch["image_tensors"], batch["texts"]):
            image_embeddings.append(_build_mock_image_embedding(image_tensor, embedding_dim))
            text_embeddings.append(_build_mock_text_embedding(text, embedding_dim))

        image_embeddings = torch.stack(image_embeddings, dim=0)
        text_embeddings = torch.stack(text_embeddings, dim=0)

        all_image_embeddings.append(image_embeddings)
        all_text_embeddings.append(text_embeddings)
        all_image_ids.extend(batch["image_ids"])
        all_texts.extend(batch["texts"])

    return {
        "image_embeddings": torch.cat(all_image_embeddings),
        "text_embeddings": torch.cat(all_text_embeddings),
        "image_ids": all_image_ids,
        "texts": all_texts,
        "device": "mock-cpu",
    }


@torch.no_grad()
def generate_openclip_embeddings(
    manifest_path: str,
    config: dict,
    model_name: str,
    device_str: str,
    batch_size: int = 2,
    split: Optional[str] = None,
) -> Dict:
    
    model, tokenizer, preprocess, device = load_model(model_name, device_str)
    dataset = SemanticRetrievalDataset(manifest_path = manifest_path, config = config, image_size=config["model"]["image_size"], split = split, transform_override = preprocess,)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_image_embeddings = []
    all_text_embeddings = []
    all_image_ids = []
    all_texts = []

    for batch in loader:
        images = batch["image_tensors"].to(device)
        texts = tokenizer(batch["texts"]).to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        all_image_embeddings.append(image_features.cpu())
        all_text_embeddings.append(text_features.cpu())
        all_image_ids.extend(batch["image_ids"])
        all_texts.extend(batch["texts"])

    return {
        "image_embeddings": torch.cat(all_image_embeddings),
        "text_embeddings": torch.cat(all_text_embeddings),
        "image_ids": all_image_ids,
        "texts": all_texts,
        "device": str(device),
    }


def generate_embeddings(
    manifest_path: str,
    config: dict,
    model_name: str,
    device_str: str,
    use_mock_inference: bool = False,
    embedding_dim: int = 512,
    batch_size: int = 2,
    split: Optional[str] = None,
) -> Dict:
    if use_mock_inference:
        return generate_mock_embeddings(
            manifest_path=manifest_path,
            config=config,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            split=split,
        )

    return generate_openclip_embeddings(
        manifest_path=manifest_path,
        config=config,
        model_name=model_name,
        device_str=device_str,
        batch_size=batch_size,
        split=split,
    )