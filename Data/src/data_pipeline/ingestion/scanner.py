"""
scanner.py — Scan a local directory for images and emit one ingestion event per file.

Usage:
    python -m src.data_pipeline.ingestion.scanner --source-dir /data/yfcc \
        --source-dataset yfcc --batch-size 100

Flow:
    For each image file found:
      1. Compute image_id (MD5 of original relative path)
      2. Determine train/val split deterministically (int(image_id[-1], 16) < 4 → val)
      3. Publish ingestion event to RabbitMQ `ingestion` queue
"""

import hashlib
import argparse
import logging
from pathlib import Path

from src.data_pipeline.ingestion.publisher import Publisher

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tiff"}


def compute_image_id(relative_path: str) -> str:
    """MD5 hash of the relative file path — stable, deterministic image ID."""
    return hashlib.md5(relative_path.encode()).hexdigest()


def compute_split(image_id: str) -> str:
    """Deterministic train/val split: last hex digit < 4 → val (~25%), else train."""
    return "val" if int(image_id[-1], 16) < 4 else "train"


def scan(source_dir: str, source_dataset: str, batch_size: int = 100) -> int:
    """Scan source_dir recursively and publish one ingestion event per image.

    Returns the total number of events published.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    publisher = Publisher()
    count = 0

    for file_path in source_path.rglob("*"):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative = str(file_path.relative_to(source_path))
        image_id = compute_image_id(relative)
        split = compute_split(image_id)

        event = {
            "image_id": image_id,
            "file_path": str(file_path.resolve()),
            "relative_path": relative,
            "source_dataset": source_dataset,
            "split": split,
        }

        publisher.publish_ingestion(event)
        count += 1

        if count % batch_size == 0:
            logger.info(f"Scanned {count} images...")

    publisher.close()
    logger.info(f"Scan complete. Published {count} ingestion events.")
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Scan image directory and publish ingestion events")
    parser.add_argument("--source-dir", required=True, help="Root directory of images")
    parser.add_argument(
        "--source-dataset",
        required=True,
        choices=["yfcc", "ava_subset"],
        help="Dataset name tag",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Log progress every N images")
    args = parser.parse_args()

    total = scan(args.source_dir, args.source_dataset, args.batch_size)
    print(f"Done. {total} events published.")
