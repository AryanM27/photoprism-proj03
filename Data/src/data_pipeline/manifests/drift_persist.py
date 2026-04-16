"""
drift_persist.py — Persist manifest drift metrics to the drift_metrics table.

Called from batch_build.py after manifests are built. For each manifest kind
(semantic, aesthetic):
  1. Finds the previous dated snapshot from dataset_snapshots (reference baseline).
  2. Downloads both the reference and current manifest from S3.
  3. Joins with image_metadata to enrich with width/height/source_dataset.
  4. Runs Evidently DataDriftPreset and extracts the dict result.
  5. Persists one DriftMetric row per column into Postgres.
  6. Also saves an HTML report to /tmp/drift_<kind>_<ts>.html.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile

import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)

BUCKET        = os.environ.get("S3_BUCKET", "training-module-proj03")
S3_PREFIX     = os.environ.get("S3_PREFIX", "data_arm9337")
S3_ENDPOINT   = os.environ.get("S3_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
S3_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
S3_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def _download_s3_jsonl(s3, key: str) -> bytes:
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, key, buf)
    return buf.getvalue()


def _previous_snapshot(db, version_tag: str, manifest_kind: str):
    """Return the second-most-recent DatasetSnapshot for this kind, or None."""
    from src.data_pipeline.db.models import DatasetSnapshot
    rows = (
        db.query(DatasetSnapshot)
        .filter(DatasetSnapshot.manifest_type == manifest_kind)
        .order_by(DatasetSnapshot.created_at.desc())
        .all()
    )
    # Skip current version rows; take the first row that belongs to a different version
    for row in rows:
        if row.version_tag != version_tag:
            return row
    return None


def _run_drift(ref_df, cur_df, numeric_cols: list[str]) -> dict:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df[numeric_cols], current_data=cur_df[numeric_cols])
    return report.as_dict()


def _extract_column_metrics(drift_dict: dict) -> list[dict]:
    """Pull per-column drift results out of Evidently's as_dict() output."""
    results = []
    try:
        metrics = drift_dict.get("metrics", [])
        for metric in metrics:
            result = metric.get("result", {})
            # DataDriftPreset wraps results under drift_by_columns
            drift_by_columns = result.get("drift_by_columns", {})
            for col_name, col_data in drift_by_columns.items():
                results.append({
                    "column_name": col_name,
                    "drift_score": col_data.get("drift_score"),
                    "drift_detected": col_data.get("drift_detected"),
                    "stattest_name": col_data.get("stattest"),
                })
    except Exception as exc:
        logger.warning("Could not parse Evidently drift dict: %s", exc)
    return results


def persist_drift_metrics(version_tag: str, ts: str) -> None:
    """Compute and persist drift metrics for both manifest kinds."""
    import pandas as pd
    from src.data_pipeline.db.session import SessionLocal
    from src.data_pipeline.db.models import DriftMetric

    db = SessionLocal()
    s3 = _s3_client()

    for kind in ("semantic", "aesthetic"):
        try:
            _persist_one_kind(db, s3, version_tag, ts, kind)
        except Exception as exc:
            logger.warning("Drift persistence skipped for %s/%s: %s", version_tag, kind, exc)

    db.close()


def _persist_one_kind(db, s3, version_tag: str, ts: str, kind: str) -> None:
    import pandas as pd
    from src.data_pipeline.db.models import DriftMetric, ImageMetadata
    from sqlalchemy import text

    prev = _previous_snapshot(db, version_tag, kind)
    if prev is None:
        logger.info("No previous snapshot found for %s/%s — skipping drift", version_tag, kind)
        return

    # Derive S3 key for current and reference manifests (train split only for drift)
    cur_prefix  = f"{S3_PREFIX}/manifests/{version_tag}_{ts}"
    cur_key     = f"{cur_prefix}/{kind}_train.jsonl"

    # Reference path: strip s3://bucket/ prefix from manifest_path
    ref_path = prev.manifest_path  # e.g. s3://bucket/prefix/
    ref_path_stripped = ref_path.replace(f"s3://{BUCKET}/", "").rstrip("/")
    ref_key = f"{ref_path_stripped}/{kind}_train.jsonl"

    try:
        cur_bytes = _download_s3_jsonl(s3, cur_key)
        ref_bytes = _download_s3_jsonl(s3, ref_key)
    except Exception as exc:
        logger.warning("Could not download manifests for drift (%s): %s", kind, exc)
        return

    cur_df = pd.read_json(io.BytesIO(cur_bytes), lines=True)
    ref_df = pd.read_json(io.BytesIO(ref_bytes), lines=True)

    if cur_df.empty or ref_df.empty:
        logger.warning("Empty manifest for %s/%s — skipping drift", version_tag, kind)
        return

    # Enrich with image_metadata columns via SQL join (build a lookup from Postgres)
    image_ids = list(set(cur_df["image_id"].tolist() + ref_df["image_id"].tolist()))
    if image_ids:
        meta_rows = (
            db.query(ImageMetadata.image_id, ImageMetadata.width, ImageMetadata.height, ImageMetadata.source_dataset)
            .filter(ImageMetadata.image_id.in_(image_ids))
            .all()
        )
        meta_df = pd.DataFrame(meta_rows, columns=["image_id", "width", "height", "source_dataset"])
        cur_df = cur_df.merge(meta_df, on="image_id", how="left")
        ref_df = ref_df.merge(meta_df, on="image_id", how="left")

    preferred = ["aesthetic_score", "width", "height", "pair_label"]
    numeric_cols = [
        c for c in preferred
        if c in ref_df.columns and c in cur_df.columns
        and ref_df[c].notna().any() and cur_df[c].notna().any()
    ]
    if not numeric_cols:
        numeric_cols = [
            c for c in ref_df.select_dtypes(include="number").columns
            if c in cur_df.columns and ref_df[c].notna().any()
        ]
    if not numeric_cols:
        logger.warning("No shared numeric columns for drift (%s) — skipping", kind)
        return

    drift_dict = _run_drift(ref_df, cur_df, numeric_cols)

    col_metrics = _extract_column_metrics(drift_dict)

    for cm in col_metrics:
        db.add(DriftMetric(
            version_tag=version_tag,
            reference_version_tag=prev.version_tag,
            manifest_kind=kind,
            column_name=cm["column_name"],
            drift_score=cm.get("drift_score"),
            drift_detected=cm.get("drift_detected"),
            stattest_name=cm.get("stattest_name"),
        ))
    db.commit()
    logger.info("Persisted %d drift metric rows for %s/%s", len(col_metrics), version_tag, kind)

    # Save HTML report to /tmp for debugging
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df[numeric_cols], current_data=cur_df[numeric_cols])
        html_path = f"/tmp/drift_{kind}_{version_tag}.html"
        report.save_html(html_path)
        logger.info("HTML drift report: %s", html_path)
    except Exception as exc:
        logger.warning("Could not save HTML drift report: %s", exc)
