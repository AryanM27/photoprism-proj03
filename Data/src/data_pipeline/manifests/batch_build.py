"""
batch_build.py — DVC `build_manifests` stage entrypoint.

Wraps build_semantic_manifests, build_aesthetic_manifests, and
build_version_metadata from build.py into a single CLI-callable script.

Usage:
    python -m src.data_pipeline.manifests.batch_build --version-tag dataset-v1.0
    python -m src.data_pipeline.manifests.batch_build --version-tag dataset-v1.0 --dataset yfcc

Required environment variables:
    DATABASE_URL, S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

from src.data_pipeline.manifests.build import (
    build_aesthetic_manifests,
    build_semantic_manifests,
    build_version_metadata,
)

# Replaced by Chameleon native S3 (CHI@TACC)
_REQUIRED_ENV_VARS = [
    "DATABASE_URL",
    "S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "S3_BUCKET",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build versioned dataset manifests (DVC build_manifests stage)"
    )
    parser.add_argument(
        "--version-tag",
        required=True,
        help="Dataset version tag, e.g. dataset-v1.0",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=["yfcc", "ava", "ava_subset", "flickr30k", "user"],
        help="Filter to a specific source dataset (default: all)",
    )
    args = parser.parse_args()

    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        print(
            f"ERROR: the following required environment variables are not set: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    version_tag: str = args.version_tag
    dataset: str | None = args.dataset

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    semantic_counts = build_semantic_manifests(version_tag, dataset, ts=ts)
    print(
        f"Semantic  — train+val: {semantic_counts['train']}  "
        f"test: {semantic_counts['test']}"
    )

    aesthetic_counts = build_aesthetic_manifests(version_tag, dataset, ts=ts)
    print(
        f"Aesthetic — train+val: {aesthetic_counts['train']}  "
        f"test: {aesthetic_counts['test']}"
    )

    metadata_uri = build_version_metadata(version_tag, semantic_counts, aesthetic_counts, dataset, ts=ts)
    print(f"Metadata  — {metadata_uri}")


if __name__ == "__main__":
    main()
