"""
promote_model.py

Downloads training summaries for new semantic and aesthetic model candidates
from S3, compares against current production metrics in config/current_production.yaml,
and promotes if both models pass their thresholds.

Usage:
    python scripts/promote_model.py \
        --semantic-path artifacts/semantic/openclip_enhanced_real_v2/training_summary.txt \
        --aesthetic-path artifacts/aesthetic/mobilenet_v3_large_fusion_real_v2/training_summary.txt

Exit codes:
    0 - promoted
    1 - rejected (metrics did not improve enough)
    2 - error (missing files, bad format, etc.)
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import boto3
from botocore.config import Config
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_PRODUCTION_PATH = REPO_ROOT / "config" / "current_production.yaml"
PROMOTION_CONFIG_PATH = REPO_ROOT / "config" / "promotion.yaml"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_summary_from_s3(s3_client, bucket: str, key: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        print(f"Downloading s3://{bucket}/{key} ...")
        s3_client.download_file(bucket, key, tmp_path)
        summary = {}
        with open(tmp_path) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                try:
                    summary[k] = float(v)
                except ValueError:
                    summary[k] = v
        return summary
    finally:
        os.unlink(tmp_path)


def check_semantic(new: dict, current: dict, thresholds: dict) -> tuple[bool, str]:
    new_recall = new.get("test_recall_at_1_mean")
    if new_recall is None:
        return False, "new semantic summary missing test_recall_at_1_mean"

    current_recall = current["semantic"]["recall_at_1"]
    min_improvement = thresholds["semantic"]["min_improvement"]
    absolute_minimum = thresholds["semantic"]["absolute_minimum"]

    print(f"Semantic  — current recall@1: {current_recall:.4f} | new: {new_recall:.4f} | "
          f"required improvement: +{min_improvement} | absolute min: {absolute_minimum}")

    if new_recall < absolute_minimum:
        return False, f"new semantic recall@1 {new_recall:.4f} below absolute minimum {absolute_minimum}"

    if new_recall < current_recall + min_improvement:
        return False, (f"new semantic recall@1 {new_recall:.4f} does not beat "
                       f"current {current_recall:.4f} by required {min_improvement}")

    return True, "semantic passed"


def check_aesthetic(new: dict, current: dict, thresholds: dict) -> tuple[bool, str]:
    new_mae = new.get("test_mae")
    if new_mae is None:
        return False, "new aesthetic summary missing test_mae"

    current_mae = current["aesthetic"]["mae"]
    min_improvement = thresholds["aesthetic"]["min_improvement"]
    absolute_maximum = thresholds["aesthetic"]["absolute_maximum"]

    print(f"Aesthetic — current MAE: {current_mae:.4f} | new: {new_mae:.4f} | "
          f"required improvement: -{min_improvement} | absolute max: {absolute_maximum}")

    if new_mae > absolute_maximum:
        return False, f"new aesthetic MAE {new_mae:.4f} exceeds absolute maximum {absolute_maximum}"

    if new_mae > current_mae - min_improvement:
        return False, (f"new aesthetic MAE {new_mae:.4f} does not improve on "
                       f"current {current_mae:.4f} by required {min_improvement}")

    return True, "aesthetic passed"


def update_production_yaml(
    current: dict,
    new_semantic: dict,
    new_aesthetic: dict,
    semantic_s3_path: str,
    aesthetic_s3_path: str,
):
    current["semantic"]["recall_at_1"] = new_semantic["test_recall_at_1_mean"]
    current["semantic"]["recall_at_5"] = new_semantic.get("test_i2t_recall_at_5", current["semantic"]["recall_at_5"])
    current["semantic"]["contrastive_loss"] = new_semantic.get("best_val_contrastive_loss", current["semantic"]["contrastive_loss"])
    current["semantic"]["model_version"] = new_semantic.get("model_version", current["semantic"]["model_version"])
    current["semantic"]["candidate_name"] = new_semantic.get("candidate_name", current["semantic"]["candidate_name"])
    current["semantic"]["s3_artifact_path"] = semantic_s3_path

    current["aesthetic"]["mae"] = new_aesthetic["test_mae"]
    current["aesthetic"]["mse"] = new_aesthetic.get("test_mse_loss", current["aesthetic"]["mse"])
    current["aesthetic"]["rmse"] = new_aesthetic.get("test_rmse", current["aesthetic"]["rmse"])
    current["aesthetic"]["model_version"] = new_aesthetic.get("model_version", current["aesthetic"]["model_version"])
    current["aesthetic"]["candidate_name"] = new_aesthetic.get("candidate_name", current["aesthetic"]["candidate_name"])
    current["aesthetic"]["s3_artifact_path"] = aesthetic_s3_path

    with open(CURRENT_PRODUCTION_PATH, "w") as f:
        yaml.dump(current, f, default_flow_style=False, sort_keys=False)

    print(f"Updated {CURRENT_PRODUCTION_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Promote new models to production if metrics improve")
    parser.add_argument("--semantic-path", required=True,
                        help="S3 key for new semantic training_summary.txt")
    parser.add_argument("--aesthetic-path", required=True,
                        help="S3 key for new aesthetic training_summary.txt")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check metrics but do not update current_production.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Skip metric gate and force promotion regardless of scores")
    args = parser.parse_args()

    promotion_cfg = load_yaml(PROMOTION_CONFIG_PATH)
    current = load_yaml(CURRENT_PRODUCTION_PATH)

    s3_cfg = promotion_cfg["s3"]
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_cfg["endpoint_url"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
    )

    try:
        new_semantic = download_summary_from_s3(s3, s3_cfg["bucket"], args.semantic_path)
        new_aesthetic = download_summary_from_s3(s3, s3_cfg["bucket"], args.aesthetic_path)
    except Exception as e:
        print(f"ERROR downloading summaries: {e}")
        sys.exit(2)

    sem_ok, sem_msg = check_semantic(new_semantic, current, promotion_cfg)
    aes_ok, aes_msg = check_aesthetic(new_aesthetic, current, promotion_cfg)

    print(f"\nSemantic:  {'PASS' if sem_ok else 'FAIL'} — {sem_msg}")
    print(f"Aesthetic: {'PASS' if aes_ok else 'FAIL'} — {aes_msg}")

    if args.force:
        print("\nForce flag set — skipping metric gate. Promoting to production.")
        if not args.dry_run:
            update_production_yaml(current, new_semantic, new_aesthetic,
                                   args.semantic_path, args.aesthetic_path)
        sys.exit(0)

    if sem_ok and aes_ok:
        print("\nBoth models passed. Promoting to production.")
        if not args.dry_run:
            update_production_yaml(current, new_semantic, new_aesthetic,
                                   args.semantic_path, args.aesthetic_path)
        else:
            print("Dry run — skipping update.")
        sys.exit(0)
    else:
        print("\nPromotion rejected. Keeping current production model.")
        sys.exit(1)


if __name__ == "__main__":
    main()
