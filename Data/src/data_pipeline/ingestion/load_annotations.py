"""
load_annotations.py — Load ground-truth annotation CSVs into the PostgreSQL database.

Two CSVs are supported:

1. Flickr30k captions (results.csv)
   - Pipe-separated file where each image has 5 captions (comment_number 0–4).
   - All 5 captions are concatenated with ' | ' and stored in image_metadata.text.
   - Performs UPDATE on existing ImageMetadata rows (rows must already exist from
     the validation worker).  Rows whose image_id is not found in the DB are skipped.

2. AVA aesthetic scores (ground_truth_dataset.csv)
   - Comma-separated file with columns: image_num, vote_1 … vote_10.
   - vote_i values are score-distribution fractions that sum to ~1.0.
   - Weighted mean  = sum((i+1) * vote_{i+1} for i in range(10))   → range [1, 10]
   - Normalized     = (weighted_mean - 1) / 9.0                     → range [0, 1]
   - One FeedbackEvent row is INSERTed per image with user_id and query_id both
     set to "ava_ground_truth".  Rows whose image_id is not found in image_metadata
     are skipped.

image_id convention (must match scanner.py):
   image_id = hashlib.md5(filename.encode()).hexdigest()
   - Flickr30k : filename = image_name.strip()          e.g. "1000092795.jpg"
   - AVA       : filename = f"{image_num}.jpg"           e.g. "953417.jpg"
"""

import argparse
import csv
import hashlib
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone

from src.data_pipeline.db.models import FeedbackEvent, ImageMetadata
from src.data_pipeline.db.session import SessionLocal

logger = logging.getLogger(__name__)

_COMMIT_BATCH = 5_000


def _md5_id(filename: str) -> str:
    """Return the MD5 hex digest of *filename* — matches scanner.py convention."""
    return hashlib.md5(filename.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Flickr30k
# ---------------------------------------------------------------------------

def load_flickr30k(csv_path: str) -> tuple[int, int]:
    """Read *results.csv* and update image_metadata.text for each Flickr30k image.

    Parameters
    ----------
    csv_path:
        Absolute or relative path to the pipe-separated results.csv file.

    Returns
    -------
    (updated, skipped)
        updated — number of ImageMetadata rows successfully updated.
        skipped — number of rows whose image_id was not found in the DB.
    """
    # Group all captions by image_name first (avoid one DB query per row).
    captions: dict[str, list[tuple[int, str]]] = defaultdict(list)

    logger.info("Reading Flickr30k CSV: %s", csv_path)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="|")
        header = next(reader)  # skip header line
        logger.debug("Header: %s", header)
        for row in reader:
            if len(row) < 3:
                continue
            image_name = row[0].strip()
            try:
                comment_number = int(row[1].strip())
            except ValueError:
                continue
            comment = row[2].strip()
            captions[image_name].append((comment_number, comment))

    logger.info("Loaded captions for %d unique images from CSV.", len(captions))

    updated = 0
    skipped = 0

    db = SessionLocal()
    try:
        for image_name, cap_list in captions.items():
            image_id = _md5_id(image_name)
            row_obj = db.get(ImageMetadata, image_id)
            if row_obj is None:
                logger.debug("image_id not found, skipping: %s (%s)", image_id, image_name)
                skipped += 1
                continue

            # Sort by comment_number and concatenate
            cap_list.sort(key=lambda t: t[0])
            combined_text = " | ".join(cap for _, cap in cap_list)
            row_obj.text = combined_text
            updated += 1

        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Error during Flickr30k load — rolled back.")
        raise
    finally:
        db.close()

    logger.info("Flickr30k done: updated=%d skipped=%d", updated, skipped)
    return updated, skipped


# ---------------------------------------------------------------------------
# AVA
# ---------------------------------------------------------------------------

def load_ava(csv_path: str) -> tuple[int, int]:
    """Read *ground_truth_dataset.csv* and insert FeedbackEvent rows for AVA images.

    Parameters
    ----------
    csv_path:
        Absolute or relative path to the comma-separated ground_truth_dataset.csv.

    Returns
    -------
    (inserted, skipped)
        inserted — number of FeedbackEvent rows successfully inserted.
        skipped  — number of rows whose image_id was not found in image_metadata.
    """
    inserted = 0
    skipped = 0

    logger.info("Reading AVA CSV: %s", csv_path)

    db = SessionLocal()
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            batch: list[FeedbackEvent] = []

            for row in reader:
                image_num = row["image_num"].strip()
                filename = f"{image_num}.jpg"
                image_id = _md5_id(filename)

                # Verify the image exists in image_metadata
                if db.get(ImageMetadata, image_id) is None:
                    logger.debug(
                        "image_id not found in image_metadata, skipping: %s (%s)",
                        image_id, filename,
                    )
                    skipped += 1
                    continue

                # Idempotency: skip if an AVA ground-truth event already exists
                already_exists = (
                    db.query(FeedbackEvent)
                    .filter_by(image_id=image_id, user_id="ava_ground_truth")
                    .first()
                    is not None
                )
                if already_exists:
                    logger.debug("AVA event already exists, skipping: %s", image_id)
                    skipped += 1
                    continue

                # Compute weighted mean over vote_1 … vote_10
                weighted_sum = 0.0
                for i in range(10):
                    col = f"vote_{i + 1}"
                    weighted_sum += (i + 1) * float(row[col])
                # Normalize to [0, 1] and clamp against malformed vote distributions
                score_normalized = round(
                    max(0.0, min(1.0, (weighted_sum - 1.0) / 9.0)), 6
                )

                event = FeedbackEvent(
                    event_id=str(uuid.uuid4()),
                    user_id="ava_ground_truth",
                    query_id="ava_ground_truth",
                    image_id=image_id,
                    aesthetic_score=score_normalized,
                    timestamp=datetime.now(timezone.utc),
                )
                batch.append(event)

                if len(batch) >= _COMMIT_BATCH:
                    db.add_all(batch)
                    db.commit()
                    inserted += len(batch)
                    logger.info("Committed batch — inserted so far: %d", inserted)
                    batch = []

            # Commit any remaining rows
            if batch:
                db.add_all(batch)
                db.commit()
                inserted += len(batch)

    except Exception:
        db.rollback()
        logger.exception("Error during AVA load — rolled back.")
        raise
    finally:
        db.close()

    logger.info("AVA done: inserted=%d skipped=%d", inserted, skipped)
    return inserted, skipped


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load Flickr30k and/or AVA annotations into the PostgreSQL database."
    )
    parser.add_argument(
        "--flickr30k",
        metavar="PATH",
        help="Path to Flickr30k results.csv (pipe-separated).",
    )
    parser.add_argument(
        "--ava",
        metavar="PATH",
        help="Path to AVA ground_truth_dataset.csv (comma-separated).",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if not args.flickr30k and not args.ava:
        parser.error("At least one of --flickr30k or --ava must be provided.")

    if args.flickr30k:
        f_updated, f_skipped = load_flickr30k(args.flickr30k)
        print(f"Flickr30k: updated={f_updated} skipped={f_skipped}")

    if args.ava:
        a_inserted, a_skipped = load_ava(args.ava)
        print(f"AVA: inserted={a_inserted} skipped={a_skipped}")
