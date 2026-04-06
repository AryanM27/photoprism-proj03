"""
scanner.py — Scan a local directory for images and emit one ingestion event per file.

Usage:
    python -m src.data_pipeline.ingestion.scanner --source-dir /data/yfcc \
        --source-dataset yfcc --batch-size 100

    Supported --source-dataset values: yfcc, ava, ava_subset, flickr30k, user

Flow:
    For each image file found:
      1. Compute image_id (MD5 of original relative path)
      2. Determine train/val/test split deterministically via last hex digit:
            0-2  → val  (~18.75%)
            3-5  → test (~18.75%)
            6-f  → train (~62.5%)
      3. Publish ingestion event to RabbitMQ `ingestion` queue
"""

import hashlib
import argparse
import logging
from pathlib import Path

from src.data_pipeline.workers.celery_app import app

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tiff"}


def compute_image_id(relative_path: str) -> str:
    """MD5 hash of the relative file path — stable, deterministic image ID."""
    return hashlib.md5(relative_path.encode()).hexdigest()


def compute_split(image_id: str) -> str:
    """Deterministic train/val/test split based on last hex digit of image_id.

    Bucket distribution (16 digits total):
        0,1,2 → val  (3/16 ≈ 18.75%)
        3,4,5 → test (3/16 ≈ 18.75%)
        6-f   → train (10/16 ≈ 62.5%)

    Never re-derive this at manifest build time — always read from images.split.
    """
    d = int(image_id[-1], 16)
    if d <= 2:
        return "val"
    if d <= 5:
        return "test"
    return "train"


def scan(source_dir: str, source_dataset: str, batch_size: int = 100) -> int:
    """Scan source_dir recursively and dispatch one Celery ingestion task per image.

    Returns the total number of tasks dispatched.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Import here to avoid circular import at module load time
    from src.data_pipeline.workers.ingestion_worker import process_ingestion_event

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

        process_ingestion_event.delay(event)
        count += 1

        if count % batch_size == 0:
            logger.info(f"Scanned {count} images...")

    logger.info(f"Scan complete. Dispatched {count} ingestion tasks.")
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Scan image directory and publish ingestion events")
    parser.add_argument("--source-dir", required=True, help="Root directory of images")
    parser.add_argument(
        "--source-dataset",
        required=True,
        choices=["yfcc", "ava", "ava_subset", "flickr30k", "user"],
        help="Dataset name tag",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Log progress every N images")
    args = parser.parse_args()

    total = scan(args.source_dir, args.source_dataset, args.batch_size)
    print(f"Done. {total} events published.")
