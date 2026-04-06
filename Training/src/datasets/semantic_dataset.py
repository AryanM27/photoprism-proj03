from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset
import random

from src.datasets.manifests import load_and_validate_semantic_manifest
from src.datasets.transforms import get_eval_transforms
from src.datasets.uri_resolver import cache_image_from_uri

class SemanticRetrievalDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        config: dict,
        image_size: int = 224,
        split: Optional[str] = None,
        transform_override = None,
        start_index: int = 0,
        max_records: Optional[int] = None,
        subset_seed: Optional[int] = None,
    ):
        self.config = config
        self.split = split
        records: List[Dict] = load_and_validate_semantic_manifest(manifest_path)

        if split is not None:
            records = [record for record in records if record["split"] == split]

        if not records:
            raise ValueError(f"No semantic records found for split={split}")
        
        if subset_seed is not None:
            rng = random.Random(subset_seed)
            rng.shuffle(records)

        if max_records is not None:
            end_index = start_index + max_records
            records = records[start_index:end_index]
        else:
            records = records[start_index:]

        if len(records) == 0:
            raise ValueError(
                f"Empty dataset after slicing: start_index={start_index}, max_records={max_records}"
            )

        self.records = records
        self.transforms = transform_override or get_eval_transforms(image_size=image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]

        image_ref = record.get("image_path") or record.get("image_uri")
        resolved_image_path = cache_image_from_uri(self.config, image_ref)
        record["resolved_image_path"] = resolved_image_path
        image = Image.open(record["resolved_image_path"]).convert("RGB")
        image_tensor = self.transforms(image)

        
        return {
            "record_id": record["record_id"],
            "image_id": record["image_id"],
            "image_ref": image_ref,
            "resolved_image_path": record["resolved_image_path"],
            "text_id": record["text_id"],
            "text": record["text"],
            "split": record["split"],
            "dataset_version": record["dataset_version"],
            "source_dataset": record["source_dataset"],
            "image_tensor": image_tensor,
            "pair_label": record["pair_label"],
        }