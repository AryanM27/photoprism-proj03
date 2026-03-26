from pathlib import Path
from typing import Dict, List

from PIL import Image
from torch.utils.data import Dataset

from src.datasets.manifests import load_and_validate_semantic_manifest
from src.datasets.transforms import get_eval_transforms


class SemanticRetrievalDataset(Dataset):
    def __init__(self, manifest_path: str, image_size: int = 224):
        self.records: List[Dict] = load_and_validate_semantic_manifest(manifest_path)
        self.transforms = get_eval_transforms(image_size=image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]

        image_path = Path(record["image_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transforms(image)

        return {
            "image_id": record["image_id"],
            "image_path": record["image_path"],
            "text": record["text"],
            "split": record["split"],
            "source_dataset": record["source_dataset"],
            "image_tensor": image_tensor,
        }