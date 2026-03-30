import json
from pathlib import Path
from typing import Dict, List

from PIL import Image
from torch.utils.data import Dataset

from src.datasets.transforms import get_eval_transforms
from src.common.paths import resolve_training_path



REQUIRED_AESTHETIC_FIELDS = {
    "image_id",
    "image_path",
    "aesthetic_score",
    "split",
    "source_dataset",
}


def load_and_validate_aesthetic_manifest(manifest_path: str) -> List[Dict]:
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            missing = REQUIRED_AESTHETIC_FIELDS - record.keys()
            if missing:
                raise ValueError(
                    f"Record {line_number} missing fields: {sorted(missing)}"
                )

            # image_path = Path(record["image_path"])
            # if not image_path.exists():
            #     raise FileNotFoundError(
            #         f"Record {line_number} points to missing image: {image_path}"
            #     )
            resolved_image_path = Path(resolve_training_path(record["image_path"]))
            if not resolved_image_path.exists():
                raise FileNotFoundError(
                    f"Record {line_number} points to missing image: {record['image_path']}"
                )

            record["image_path"] = str(resolved_image_path)

            if record["split"] not in {"train", "val", "test"}:
                raise ValueError(
                    f"Record {line_number} has invalid split: {record['split']}"
                )

            record["aesthetic_score"] = float(record["aesthetic_score"])
            records.append(record)

    if not records:
        raise ValueError(f"Manifest file is empty: {path}")

    return records


class AestheticDataset(Dataset):
    def __init__(self, manifest_path: str, split: str, image_size: int = 224):
        all_records = load_and_validate_aesthetic_manifest(manifest_path)
        self.records = [r for r in all_records if r["split"] == split]
        self.transforms = get_eval_transforms(image_size=image_size)

        if not self.records:
            raise ValueError(f"No records found for split='{split}'")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]

        image = Image.open(record["image_path"]).convert("RGB")
        image_tensor = self.transforms(image)

        return {
            "image_id": record["image_id"],
            "image_tensor": image_tensor,
            "aesthetic_score": record["aesthetic_score"],
            "split": record["split"],
            "source_dataset": record["source_dataset"],
        }