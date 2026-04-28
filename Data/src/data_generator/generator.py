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

from src.data_pipeline.db.models import FeedbackEvent, Image
from src.data_pipeline.feedback.synthetic import generate_feedback_session
from src.data_generator.photoprism.client import PhotoprismClient

_FALLBACK_USERS = [
    "user_alice",
    "user_bob",
    "user_carol",
    "user_dave",
    "user_eve",
    "user_frank",
    "user_grace",
    "user_henry",
    "user_iris",
    "user_jack",
]


def load_users_from_photoprism() -> list[str]:
    """Fetch usernames from PhotoPrism using admin credentials from env vars.

    Falls back to _FALLBACK_USERS if the env vars are not set or the call fails.
    """
    photoprism_url = os.getenv("PHOTOPRISM_URL")
    username = os.getenv("PHOTOPRISM_ADMIN_USER", os.getenv("PHOTOPRISM_USERNAME", "admin"))
    password = os.getenv("PHOTOPRISM_ADMIN_PASSWORD", os.getenv("PHOTOPRISM_PASSWORD"))
    if not photoprism_url or not password:
        return _FALLBACK_USERS
    client = PhotoprismClient(base_url=photoprism_url, username=username, password=password)
    try:
        client.login()
        users = client.get_users()
        names = [u["Name"] for u in users if u.get("Name")]
        client.logout()
        return names if names else _FALLBACK_USERS
    except Exception:
        return _FALLBACK_USERS

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
    """Query Postgres for all image IDs where status='validated'."""
    database_url = os.environ["DATABASE_URL"]
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        ids = [row.image_id for row in db.query(Image.image_id).filter(Image.status == "validated").all()]
    finally:
        db.close()
    if not ids:
        raise ValueError("No validated images found in database (status='validated'). Cannot run generator.")
    return ids


def run_generator(
    service_url: str,
    image_ids: list[str],
    duration_seconds: int,
    rate_per_second: float,
    users: list[str] | None = None,
) -> int:
    """Fire synthetic sessions until duration_seconds elapses.

    Returns total number of sessions generated.
    """
    database_url = os.environ["DATABASE_URL"]
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    db = Session()

    active_users = users if users else _FALLBACK_USERS

    end_time = time.time() + duration_seconds
    session_count = 0
    total_events = 0
    interval = 1.0 / rate_per_second

    try:
        while time.time() < end_time:
            query = random.choice(SAMPLE_QUERIES)
            user_id = random.choice(active_users)
            events = generate_feedback_session(user_id, query, image_ids)

            try:
                requests.post(
                    f"{service_url}/api/v1/feedback",
                    json=events,
                    timeout=5,
                )
            except Exception:
                pass  # generator never blocks on service availability

            for e in events:
                db.add(FeedbackEvent(**e))
            db.commit()

            session_count += 1
            total_events += len(events)
            time.sleep(interval)
    finally:
        db.close()

    print(f"Generated {session_count} sessions ({total_events} events) over {duration_seconds}s")
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

    users = load_users_from_photoprism()
    print(f"Using {len(users)} user(s): {users}")

    run_generator(args.service_url, image_ids, args.duration, args.rate, users=users)
