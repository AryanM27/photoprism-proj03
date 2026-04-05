"""
build.py — Build versioned dataset manifests from validated images in Postgres.

Produces six JSONL manifest files per version and uploads them to MinIO:
    manifests/v<N>/semantic_train.jsonl
    manifests/v<N>/semantic_val.jsonl
    manifests/v<N>/semantic_test.jsonl
    manifests/v<N>/aesthetic_train.jsonl
    manifests/v<N>/aesthetic_val.jsonl
    manifests/v<N>/aesthetic_test.jsonl

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

# Aesthetic scores from FeedbackEvent are stored in 0–1 range.
# Training lead expects 0–10 scale.
_AESTHETIC_SCALE = 10.0

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

SPLIT_STRATEGY = "hash_hex_62_19_19"  # last hex digit: 0-2→val, 3-5→test, 6-f→train


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
    """Build semantic manifests per training lead schema.

    Each record: record_id, image_id, image_uri, split, dataset_version,
                 source_dataset, text_id, text, pair_label.

    - image_uri: stable s3:// URI (set at ingest time, never a local path).
    - text_id: deterministic UUID5 from text content — same text always maps to same ID.
    - pair_label: 1 for all records (matched image-text pairs only).
    - Anti-leakage: split is read from images.split (hash-deterministic at ingest), never re-derived here.

    Returns dict with counts per split (train, val, test).
    """
    db = SessionLocal()
    s3 = _s3_client()
    try:
        query = (
            db.query(Image, ImageMetadata)
            .join(ImageMetadata, Image.image_id == ImageMetadata.image_id)
            .filter(Image.status == "validated")
            .filter(Image.split.isnot(None))
            .filter(ImageMetadata.text.isnot(None))
        )
        if source_dataset:
            query = query.filter(Image.source_dataset == source_dataset)

        train_records, val_records, test_records = [], [], []
        for image, meta in query.all():
            record = {
                "record_id":       str(uuid.uuid4()),
                "image_id":        image.image_id,
                "image_uri":       image.image_uri,
                "split":           image.split,
                "dataset_version": version,
                "source_dataset":  image.source_dataset,
                "text_id":         str(uuid.uuid5(uuid.NAMESPACE_DNS, meta.text)),
                "text":            meta.text,
                "pair_label":      1,
            }
            if image.split == "val":
                val_records.append(record)
            elif image.split == "test":
                test_records.append(record)
            else:
                train_records.append(record)

        prefix = f"manifests/{version}"
        train_count = _upload_jsonl(s3, f"{prefix}/semantic_train.jsonl", train_records)
        val_count   = _upload_jsonl(s3, f"{prefix}/semantic_val.jsonl",   val_records)
        test_count  = _upload_jsonl(s3, f"{prefix}/semantic_test.jsonl",  test_records)

        manifest_path = f"s3://{BUCKET}/{prefix}/"
        _record_snapshot(db, version, manifest_path, train_count + val_count + test_count)

        return {"train": train_count, "val": val_count, "test": test_count}
    finally:
        db.close()


def build_aesthetic_manifests(version: str, source_dataset: str | None = None) -> dict:
    """Build aesthetic manifests per training lead schema.

    Each record: record_id, image_id, image_uri, split, dataset_version,
                 source_dataset, aesthetic_score (0–10 scale).

    - aesthetic_score is averaged across all FeedbackEvent rows per image,
      then multiplied by 10 to normalise from the stored 0–1 range to 0–10.
    - image_uri: stable s3:// URI, never a local path.
    - Anti-leakage: split is read from images.split, never re-derived here.

    Returns dict with counts per split.
    """
    from src.data_pipeline.db.models import FeedbackEvent
    from sqlalchemy import func

    db = SessionLocal()
    s3 = _s3_client()
    try:
        # AVG as proxy for median (Postgres lacks MEDIAN).
        # Scores are stored in 0–1 range; normalised to 0–10 at record build time.
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
            .filter(Image.split.isnot(None))
            .filter(score_subq.c.aesthetic_score.isnot(None))
        )
        if source_dataset:
            query = query.filter(Image.source_dataset == source_dataset)

        train_records, val_records, test_records = [], [], []
        for image, score in query.all():
            if score is None:
                logger.warning(f"Skipping {image.image_id}: null aesthetic_score after join")
                continue
            raw_score = float(score)
            if not (0.0 <= raw_score <= 1.0):
                logger.warning(
                    f"aesthetic_score {raw_score} for {image.image_id} is outside [0,1]; clamping."
                )
                raw_score = max(0.0, min(1.0, raw_score))
            record = {
                "record_id":       str(uuid.uuid4()),
                "image_id":        image.image_id,
                "image_uri":       image.image_uri,
                "split":           image.split,
                "dataset_version": version,
                "source_dataset":  image.source_dataset,
                "aesthetic_score": round(raw_score * _AESTHETIC_SCALE, 4),
            }
            if image.split == "val":
                val_records.append(record)
            elif image.split == "test":
                test_records.append(record)
            else:
                train_records.append(record)

        prefix = f"manifests/{version}"
        train_count = _upload_jsonl(s3, f"{prefix}/aesthetic_train.jsonl", train_records)
        val_count   = _upload_jsonl(s3, f"{prefix}/aesthetic_val.jsonl",   val_records)
        test_count  = _upload_jsonl(s3, f"{prefix}/aesthetic_test.jsonl",  test_records)

        manifest_path = f"s3://{BUCKET}/{prefix}/"
        _record_snapshot(db, version, manifest_path, train_count + val_count + test_count)

        return {"train": train_count, "val": val_count, "test": test_count}
    finally:
        db.close()


def build_version_metadata(
    version: str,
    semantic_counts: dict | None,
    aesthetic_counts: dict | None,
    source_dataset: str | None = None,
) -> str:
    """Upload a metadata.json for a dataset version to MinIO.

    Pass None for semantic_counts or aesthetic_counts if that manifest type was
    not built in this run — those URIs and counts will be omitted from the output
    rather than shown as zero (which would falsely imply an empty built manifest).

    Schema:
        version, built_at, split_strategy, source_datasets,
        manifest_uris { only types that were built },
        counts { only types that were built, plus total }

    Returns the s3:// URI of the uploaded metadata file.
    """
    s3 = _s3_client()
    prefix = f"manifests/{version}"

    manifest_uris = {}
    counts = {}
    if semantic_counts is not None:
        manifest_uris["semantic_train"] = f"s3://{BUCKET}/{prefix}/semantic_train.jsonl"
        manifest_uris["semantic_val"]   = f"s3://{BUCKET}/{prefix}/semantic_val.jsonl"
        manifest_uris["semantic_test"]  = f"s3://{BUCKET}/{prefix}/semantic_test.jsonl"
        counts["semantic_train"] = semantic_counts["train"]
        counts["semantic_val"]   = semantic_counts["val"]
        counts["semantic_test"]  = semantic_counts["test"]
    if aesthetic_counts is not None:
        manifest_uris["aesthetic_train"] = f"s3://{BUCKET}/{prefix}/aesthetic_train.jsonl"
        manifest_uris["aesthetic_val"]   = f"s3://{BUCKET}/{prefix}/aesthetic_val.jsonl"
        manifest_uris["aesthetic_test"]  = f"s3://{BUCKET}/{prefix}/aesthetic_test.jsonl"
        counts["aesthetic_train"] = aesthetic_counts["train"]
        counts["aesthetic_val"]   = aesthetic_counts["val"]
        counts["aesthetic_test"]  = aesthetic_counts["test"]
    # total = unique images (same images appear in both semantic and aesthetic,
    # so sum only one type's splits to avoid double-counting)
    if semantic_counts is not None:
        counts["total"] = sum(semantic_counts.values())
    elif aesthetic_counts is not None:
        counts["total"] = sum(aesthetic_counts.values())
    else:
        counts["total"] = 0

    meta = {
        "version":         version,
        "built_at":        datetime.now(timezone.utc).isoformat(),
        "split_strategy":  SPLIT_STRATEGY,
        "source_datasets": [source_dataset] if source_dataset else ["all"],
        "manifest_uris":   manifest_uris,
        "counts":          counts,
    }
    key = f"{prefix}/metadata.json"
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(meta, indent=2).encode(),
        ContentType="application/json",
    )
    uri = f"s3://{BUCKET}/{key}"
    logger.info(f"Uploaded version metadata: {uri}")
    return uri


def generate_drift_report(
    reference_manifest: str, current_manifest: str, output_path: str
) -> None:
    """Compare current dataset distribution against reference — detect drift.

    Args:
        reference_manifest: Path to reference JSONL manifest (e.g. previous semantic_train.jsonl).
        current_manifest:   Path to current JSONL manifest to compare against.
        output_path:        File path to write the HTML drift report.
    """
    import pandas as pd
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    ref_df = pd.read_json(reference_manifest, lines=True)
    cur_df = pd.read_json(current_manifest, lines=True)

    if len(ref_df) == 0 or len(cur_df) == 0:
        raise ValueError("One or both manifests are empty — cannot generate drift report.")

    preferred_cols = ["aesthetic_score", "width", "height", "pair_label"]
    numeric_cols = [
        c for c in preferred_cols
        if c in ref_df.columns and c in cur_df.columns
    ]
    if not numeric_cols:
        # Fall back to any numeric column present in both
        numeric_cols = [
            c for c in ref_df.select_dtypes(include="number").columns
            if c in cur_df.columns
        ]
    if not numeric_cols:
        raise ValueError(
            "No shared numeric columns found between reference and current manifest. "
            "Expected at least one of: aesthetic_score, width, height, pair_label."
        )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df[numeric_cols], current_data=cur_df[numeric_cols])

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    report.save_html(output_path)
    logger.info("Drift report saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build versioned dataset manifests")
    parser.add_argument("--version", required=True, help="Version tag e.g. v1")
    parser.add_argument("--dataset", default=None, choices=["yfcc", "ava_subset"],
                        help="Filter to a specific source dataset (default: all)")
    parser.add_argument("--type", default="both", choices=["semantic", "aesthetic", "both"],
                        help="Which manifest type to build")
    args = parser.parse_args()

    # None = not built this run (excluded from metadata counts rather than shown as 0)
    semantic_counts  = None
    aesthetic_counts = None

    if args.type in ("semantic", "both"):
        semantic_counts = build_semantic_manifests(args.version, args.dataset)
        print(f"Semantic manifests:  train={semantic_counts['train']} val={semantic_counts['val']} test={semantic_counts['test']}")

    if args.type in ("aesthetic", "both"):
        aesthetic_counts = build_aesthetic_manifests(args.version, args.dataset)
        print(f"Aesthetic manifests: train={aesthetic_counts['train']} val={aesthetic_counts['val']} test={aesthetic_counts['test']}")

    metadata_uri = build_version_metadata(
        args.version, semantic_counts, aesthetic_counts, args.dataset
    )
    print(f"Version metadata: {metadata_uri}")
