import json
from pathlib import Path
from typing import List, Dict


REQUIRED_SEMANTIC_FIELDS = {
    "image_id",
    "image_path",
    "text",
    "split",
    "source_dataset",
}


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
    for idx, record in enumerate(records, start=1):
        missing_fields = REQUIRED_SEMANTIC_FIELDS - record.keys()

        if missing_fields:
            raise ValueError(
                f"Record {idx} is missing required fields: {sorted(missing_fields)}"
            )

        image_path = Path(record["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(
                f"Record {idx} points to missing image file: {image_path}"
            )

        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError(f"Record {idx} has empty or invalid text field")

        if record["split"] not in {"train", "val", "test"}:
            raise ValueError(
                f"Record {idx} has invalid split '{record['split']}'. "
                f"Expected one of: train, val, test"
            )


def load_and_validate_semantic_manifest(manifest_path: str) -> List[Dict]:
    records = load_jsonl_manifest(manifest_path)
    validate_semantic_manifest(records)
    return records