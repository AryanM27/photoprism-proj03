"""
build.py — Build versioned dataset manifests from validated images in Postgres.

Produces four JSONL manifest files per version and uploads them to MinIO:
    manifests/v<N>/semantic_train.jsonl
    manifests/v<N>/semantic_val.jsonl
    manifests/v<N>/aesthetic_train.jsonl
    manifests/v<N>/aesthetic_val.jsonl

Records a DatasetSnapshot row in Postgres for each version built.

Usage:
    python -m src.data_pipeline.manifests.build --version v1 --dataset yfcc
    python -m src.data_pipeline.manifests.build --version v1  # all datasets
"""

from __future__ import annotations

import os
import io
import uuid
import json
import logging
import argparse
from datetime import datetime, timezone

import boto3
from botocore.client import Config
from sqlalchemy import select

from src.data_pipeline.db.session import SessionLocal
from src.data_pipeline.db.models import Image, ImageMetadata, DatasetSnapshot

logger = logging.getLogger(__name__)

BUCKET        = os.environ.get("MINIO_BUCKET", "photoprism-proj03")
S3_ENDPOINT   = os.environ.get("S3_ENDPOINT_URL", "http://minio:9000")
S3_ACCESS_KEY = os.environ.get("MINIO_USER", "minioadmin")
S3_SECRET_KEY = os.environ.get("MINIO_PASSWORD", "minioadmin")

SPLIT_STRATEGY = "hash_hex_75_25"  # int(image_id[-1], 16) < 4 → val (~25%), else train (~75%)


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def _upload_jsonl(s3, key: str, records: list[dict]) -> int:
    """Serialise records to JSONL and upload to MinIO. Returns record count."""
    body = "\n".join(json.dumps(r) for r in records) + "\n"
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=body.encode(),
        ContentType="application/x-ndjson",
    )
    logger.info(f"Uploaded {key} ({len(records)} records)")
    return len(records)


def _record_snapshot(db, version: str, manifest_path: str, record_count: int) -> None:
    snap = DatasetSnapshot(
        snapshot_id=str(uuid.uuid4()),
        version_tag=version,
        manifest_path=manifest_path,
        record_count=record_count,
        split_strategy=SPLIT_STRATEGY,
        created_at=datetime.now(timezone.utc),
    )
    db.add(snap)
    db.commit()
    logger.info(f"Recorded snapshot: {version} — {manifest_path} ({record_count} rows)")


def build_semantic_manifests(version: str, source_dataset: str | None = None) -> dict:
    """Build semantic manifests (image_id, image_uri, text, split, source_dataset).

    Includes only validated images that have a non-null text in image_metadata.
    Anti-leakage: split is deterministic from image_id hash, not assigned at build time.

    Returns dict with counts per split.
    """
    db = SessionLocal()
    s3 = _s3_client()
    try:
        query = (
            db.query(Image, ImageMetadata)
            .join(ImageMetadata, Image.image_id == ImageMetadata.image_id)
            .filter(Image.status == "validated")
            .filter(Image.split.isnot(None))       # exclude rows with no split assigned
            .filter(ImageMetadata.text.isnot(None))
        )
        if source_dataset:
            query = query.filter(Image.source_dataset == source_dataset)

        train_records, val_records = [], []
        for image, meta in query.all():
            record = {
                "image_id": image.image_id,
                "image_uri": image.image_uri,
                "text": meta.text,
                "split": image.split,
                "source_dataset": image.source_dataset,
            }
            if image.split == "val":
                val_records.append(record)
            else:
                train_records.append(record)

        prefix = f"manifests/{version}"
        train_key = f"{prefix}/semantic_train.jsonl"
        val_key   = f"{prefix}/semantic_val.jsonl"

        train_count = _upload_jsonl(s3, train_key, train_records)
        val_count   = _upload_jsonl(s3, val_key,   val_records)

        # Store prefix URI — callers append /semantic_train.jsonl or /semantic_val.jsonl
        manifest_path = f"s3://{BUCKET}/{prefix}/"
        _record_snapshot(db, version, manifest_path, train_count + val_count)

        return {"train": train_count, "val": val_count}
    finally:
        db.close()


def build_aesthetic_manifests(version: str, source_dataset: str | None = None) -> dict:
    """Build aesthetic manifests (image_id, image_uri, aesthetic_score, split, source_dataset).

    Includes only validated images that have a non-null aesthetic_score in feedback_events.
    Uses median aesthetic_score per image across all feedback events as the label.
    Anti-leakage: split is deterministic from image_id hash, not assigned at build time.

    Returns dict with counts per split.
    """
    from src.data_pipeline.db.models import FeedbackEvent
    from sqlalchemy import func

    db = SessionLocal()
    s3 = _s3_client()
    try:
        # Aggregate median-like score: use AVG as a proxy (Postgres lacks MEDIAN)
        score_subq = (
            db.query(
                FeedbackEvent.image_id,
                func.avg(FeedbackEvent.aesthetic_score).label("aesthetic_score"),
            )
            .filter(FeedbackEvent.aesthetic_score.isnot(None))
            .group_by(FeedbackEvent.image_id)
            .subquery()
        )

        query = (
            db.query(Image, score_subq.c.aesthetic_score)
            .join(score_subq, Image.image_id == score_subq.c.image_id)
            .filter(Image.status == "validated")
            .filter(Image.split.isnot(None))       # exclude rows with no split assigned
            .filter(score_subq.c.aesthetic_score.isnot(None))
        )
        if source_dataset:
            query = query.filter(Image.source_dataset == source_dataset)

        train_records, val_records = [], []
        for image, score in query.all():
            if score is None:
                logger.warning(f"Skipping {image.image_id}: null aesthetic_score after join")
                continue
            record = {
                "image_id": image.image_id,
                "image_uri": image.image_uri,
                "aesthetic_score": round(float(score), 4),
                "split": image.split,
                "source_dataset": image.source_dataset,
            }
            if image.split == "val":
                val_records.append(record)
            else:
                train_records.append(record)

        prefix = f"manifests/{version}"
        train_key = f"{prefix}/aesthetic_train.jsonl"
        val_key   = f"{prefix}/aesthetic_val.jsonl"

        train_count = _upload_jsonl(s3, train_key, train_records)
        val_count   = _upload_jsonl(s3, val_key,   val_records)

        # Store prefix URI — callers append /aesthetic_train.jsonl or /aesthetic_val.jsonl
        manifest_path = f"s3://{BUCKET}/{prefix}/"
        _record_snapshot(db, version, manifest_path, train_count + val_count)

        return {"train": train_count, "val": val_count}
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build versioned dataset manifests")
    parser.add_argument("--version", required=True, help="Version tag e.g. v1")
    parser.add_argument("--dataset", default=None, choices=["yfcc", "ava_subset"],
                        help="Filter to a specific source dataset (default: all)")
    parser.add_argument("--type", default="both", choices=["semantic", "aesthetic", "both"],
                        help="Which manifest type to build")
    args = parser.parse_args()

    if args.type in ("semantic", "both"):
        counts = build_semantic_manifests(args.version, args.dataset)
        print(f"Semantic manifests: train={counts['train']} val={counts['val']}")

    if args.type in ("aesthetic", "both"):
        counts = build_aesthetic_manifests(args.version, args.dataset)
        print(f"Aesthetic manifests: train={counts['train']} val={counts['val']}")
