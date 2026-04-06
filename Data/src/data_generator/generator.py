"""
generator.py — Synthetic traffic generator.

Reads image IDs from a JSONL manifest and fires simulated query sessions
at the service's feedback endpoint at a configurable rate.

Usage:
    python -m src.data_generator.generator \\
        --manifest data/manifests/semantic_train.jsonl \\
        --duration 300 \\
        --rate 2
"""

import argparse
import json
import os
import random
import time

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data_pipeline.db.models import Image
from src.data_pipeline.feedback.synthetic import generate_feedback_session

SAMPLE_QUERIES = [
    "sunset over water",
    "mountain landscape",
    "street photography",
    "portrait close-up",
    "food photography",
    "architecture",
    "cityscape at night",
    "beach and ocean",
    "forest trail",
    "wildlife",
]


def load_image_ids_from_manifest(manifest_path: str) -> list[str]:
    """Read image_ids from a JSONL manifest file."""
    ids = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["image_id"])
    if not ids:
        raise ValueError(f"No image IDs found in manifest: {manifest_path}")
    return ids


def load_image_ids_from_db() -> list[str]:
    """Query Postgres for all image IDs where status='valid'."""
    print("No manifest found, loading image IDs from database...")
    database_url = os.environ["DATABASE_URL"]
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        ids = [row.image_id for row in db.query(Image.image_id).filter(Image.status == "valid").all()]
    finally:
        db.close()
    if not ids:
        raise ValueError("No valid images found in database (status='valid'). Cannot run generator.")
    return ids


def run_generator(
    service_url: str,
    image_ids: list[str],
    duration_seconds: int,
    rate_per_second: float,
) -> int:
    """Fire synthetic sessions until duration_seconds elapses.

    Returns total number of sessions generated.
    """
    end_time = time.time() + duration_seconds
    session_count = 0
    interval = 1.0 / rate_per_second

    while time.time() < end_time:
        query = random.choice(SAMPLE_QUERIES)
        user_id = f"synthetic_user_{random.randint(1, 1000)}"
        events = generate_feedback_session(user_id, query, image_ids)

        try:
            requests.post(
                f"{service_url}/api/v1/feedback",
                json=events,
                timeout=5,
            )
        except Exception:
            pass  # generator never blocks on service availability

        session_count += 1
        time.sleep(interval)

    print(f"Generated {session_count} sessions over {duration_seconds}s")
    return session_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic traffic generator")
    parser.add_argument(
        "--service-url",
        default=os.getenv("SERVICE_URL", "http://localhost:2342"),
        help="Base URL of the serving endpoint",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to JSONL manifest to load image IDs from. If not provided or file does not exist, falls back to querying Postgres for all images with status='valid'.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="How long to run in seconds",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Sessions per second",
    )
    args = parser.parse_args()

    if args.manifest and os.path.exists(args.manifest):
        image_ids = load_image_ids_from_manifest(args.manifest)
        print(f"Loaded {len(image_ids)} image IDs from {args.manifest}")
    else:
        image_ids = load_image_ids_from_db()
        print(f"Loaded {len(image_ids)} image IDs from database.")
    run_generator(args.service_url, image_ids, args.duration, args.rate)
