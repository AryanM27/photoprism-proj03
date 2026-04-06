"""
normalizer.py — Extract and normalise image metadata from S3 object storage (Chameleon CHI@TACC).

extract_metadata(storage_path, image_id) downloads the image, reads EXIF
data via Pillow, and returns a populated ImageMetadata ORM object ready to
be added to the Postgres session.

EXIF extraction is best-effort: missing tags are stored as None.
"""

from __future__ import annotations

import os
import io
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import boto3
from botocore.client import Config
from PIL import Image, ExifTags

from src.data_pipeline.db.models import ImageMetadata

logger = logging.getLogger(__name__)

# Replaced by Chameleon native S3 (CHI@TACC)
BUCKET_NAME   = "training-module-proj03"
BUCKET        = os.environ.get("S3_BUCKET", BUCKET_NAME)
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


def _download(storage_path: str) -> bytes:
    s3 = _s3_client()
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, storage_path, buf)
    return buf.getvalue()


def _parse_exif(img: Image.Image) -> dict:
    """Return a flat dict of readable EXIF tag name → value.

    Uses getexif() which works across JPEG, PNG, WEBP, and TIFF.
    """
    try:
        raw = img.getexif()  # Pillow 8+ universal API (not JPEG-only _getexif)
    except Exception:
        return {}
    if not raw:
        return {}
    result = {}
    for tag_id, value in raw.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        # Skip large binary blobs (e.g. MakerNote, thumbnail data)
        if isinstance(value, bytes) and len(value) > 256:
            continue
        try:
            json.dumps(value)  # keep only JSON-serialisable values
            result[tag_name] = value
        except (TypeError, ValueError):
            result[tag_name] = str(value)
    return result


def _parse_captured_at(exif: dict) -> Optional[datetime]:
    """Parse DateTimeOriginal or DateTime from EXIF into a UTC datetime."""
    raw = exif.get("DateTimeOriginal") or exif.get("DateTime")
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def extract_metadata(
    storage_path: str,
    image_id: str,
    source_dataset: Optional[str] = None,
) -> ImageMetadata:
    """Download image from S3 and extract metadata.

    Args:
        storage_path:   S3 key e.g. raw/<image_id>.jpg
        image_id:       primary key for the ImageMetadata row
        source_dataset: passed through from the Image record (yfcc / ava_subset)

    Returns:
        ImageMetadata ORM object (not yet committed to session)

    Raises:
        Exception if download or image open fails (caller handles retry)
    """
    try:
        data = _download(storage_path)
    except Exception as exc:
        raise RuntimeError(f"download_failed for {image_id}: {exc}") from exc

    try:
        img = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise RuntimeError(f"image_open_failed for {image_id}: {exc}") from exc

    width, height = img.size
    fmt = img.format  # always set after a successful Image.open

    exif = {}
    try:
        exif = _parse_exif(img)
    except Exception as exc:
        logger.warning(f"EXIF extraction failed for {image_id}: {exc}")

    captured_at = _parse_captured_at(exif)
    exif_json = json.dumps(exif) if exif else None

    logger.info(f"Extracted metadata for {image_id}: {fmt} {width}x{height}")

    return ImageMetadata(
        image_id=image_id,
        source_dataset=source_dataset,
        width=width,
        height=height,
        format=fmt,
        exif_json=exif_json,
        # tags populated downstream by a tagging worker (not part of ingestion)
        captured_at=captured_at,
        normalized_at=datetime.now(timezone.utc),
    )
