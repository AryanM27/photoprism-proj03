import json
from pathlib import Path
from typing import List, Dict

from src.common.paths import resolve_training_path

REQUIRED_SEMANTIC_FIELDS = {
    "record_id",
    "image_id",
    "split",
    "dataset_version",
    "source_dataset",
    "text_id",
    "text",
    "pair_label",
    "num_clicks",
}

REQUIRED_AESTHETIC_FIELDS = {
    "record_id",
    "image_id",
    "split",
    "dataset_version",
    "source_dataset",
    "aesthetic_score",
    "num_favourites",
}

VALID_SPLITS = {"train", "val", "test"}

def load_jsonl_manifest(manifest_path: str) -> List[Dict]:
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}: {e}"
                ) from e

            records.append(record)

    if not records:
        raise ValueError(f"Manifest file is empty: {path}")

    return records

def validate_semantic_manifest(records: List[Dict]) -> None:
    seen_record_ids = set()

    for idx, record in enumerate(records, start=1):
        missing = REQUIRED_SEMANTIC_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Record {idx} missing required fields: {sorted(missing)}")

        if "image_uri" not in record and "image_path" not in record:
            raise ValueError(f"Record {idx} must include either image_uri or image_path")

        for field in ["record_id", "image_id", "text_id", "dataset_version", "source_dataset"]:
            if not str(record[field]).strip():
                raise ValueError(f"Record {idx} has empty field: {field}")

        if record["record_id"] in seen_record_ids:
            raise ValueError(f"Duplicate record_id found: {record['record_id']}")
        seen_record_ids.add(record["record_id"])

        if record["split"] not in VALID_SPLITS:
            raise ValueError(f"Record {idx} has invalid split: {record['split']}")

        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError(f"Record {idx} has empty or invalid text field")
        
        try:
            record["pair_label"] = int(record["pair_label"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Record {idx} has non-integer pair_label: {record['pair_label']}"
            )

        if record["pair_label"] not in {0, 1}:
            raise ValueError(f"Record {idx} has invalid pair_label: {record['pair_label']}")

        if "image_path" in record:
            resolved_image_path = Path(resolve_training_path(record["image_path"]))
            if not resolved_image_path.exists():
                raise FileNotFoundError(
                    f"Record {idx} points to missing image file: {record['image_path']}"
                )
            record["image_path"] = str(resolved_image_path)

def load_and_validate_semantic_manifest(manifest_path: str) -> List[Dict]:
    records = load_jsonl_manifest(manifest_path)
    validate_semantic_manifest(records)
    return records


def validate_aesthetic_manifest(records: List[Dict]) -> None:
    seen_record_ids = set()

    for idx, record in enumerate(records, start=1):
        missing = REQUIRED_AESTHETIC_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Record {idx} missing required fields: {sorted(missing)}")

        if "image_uri" not in record and "image_path" not in record:
            raise ValueError(f"Record {idx} must include either image_uri or image_path")

        for field in ["record_id", "image_id", "dataset_version", "source_dataset"]:
            if not str(record[field]).strip():
                raise ValueError(f"Record {idx} has empty field: {field}")

        if record["record_id"] in seen_record_ids:
            raise ValueError(f"Duplicate record_id found: {record['record_id']}")
        seen_record_ids.add(record["record_id"])

        if record["split"] not in VALID_SPLITS:
            raise ValueError(f"Record {idx} has invalid split: {record['split']}")

        try:
            record["aesthetic_score"] = float(record["aesthetic_score"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Record {idx} has non-numeric aesthetic score: {record['aesthetic_score']}"
            )

        if record["aesthetic_score"] < 0 or record["aesthetic_score"] > 10:
            raise ValueError(f"Record {idx} has invalid aesthetic_score: {record['aesthetic_score']}")

        if "image_path" in record:
            resolved_image_path = Path(resolve_training_path(record["image_path"]))
            if not resolved_image_path.exists():
                raise FileNotFoundError(
                    f"Record {idx} points to missing image file: {record['image_path']}"
                )
            record["image_path"] = str(resolved_image_path)


def load_and_validate_aesthetic_manifest(manifest_path: str) -> List[Dict]:
    records = load_jsonl_manifest(manifest_path)
    validate_aesthetic_manifest(records)
    return records