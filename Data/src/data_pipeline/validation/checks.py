"""
checks.py — Quality checks for images stored in S3 object storage (Chameleon CHI@TACC).

run_checks(storage_path) downloads the image from S3 object storage (Chameleon CHI@TACC) into memory and runs:
  1. Format check   — file extension matches actual image format
  2. Integrity check — Pillow can fully decode the image (detects corruption)
  3. Min resolution  — width and height >= MIN_DIMENSION px
  4. Max file size   — raw bytes <= MAX_FILE_BYTES

Returns (passed: bool, reason: str). reason is empty string on pass.
"""

import os
import io
import logging

import boto3
from botocore.client import Config
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Replaced by Chameleon native S3 (CHI@TACC)
BUCKET_NAME    = "training-module-proj03"
BUCKET         = os.environ.get("S3_BUCKET", BUCKET_NAME)
S3_ENDPOINT    = os.environ.get("S3_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
S3_ACCESS_KEY  = os.environ.get("AWS_ACCESS_KEY_ID", "")
S3_SECRET_KEY  = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

MIN_DIMENSION  = 32      # pixels — reject thumbnails / icons
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "TIFF"}


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def _download(storage_path: str) -> bytes:
    """Download object from S3 and return raw bytes."""
    s3 = _s3_client()
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, storage_path, buf)
    return buf.getvalue()


def run_checks(storage_path: str) -> tuple[bool, str]:
    """Run all quality checks on an image in S3 (Chameleon CHI@TACC).

    Args:
        storage_path: S3 key e.g. raw/<image_id>.jpg

    Returns:
        (True, "") on pass
        (False, reason) on failure
    """
    try:
        data = _download(storage_path)
    except Exception as exc:
        return False, f"download_failed: {exc}"

    # File size check
    if len(data) > MAX_FILE_BYTES:
        return False, f"file_too_large: {len(data)} bytes > {MAX_FILE_BYTES}"

    # Format + integrity check via Pillow
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # detects truncation / corruption without full decode
    except UnidentifiedImageError:
        return False, "unrecognised_format"
    except Exception as exc:
        return False, f"corrupt_image: {exc}"

    # Re-open after verify() (verify() closes the file handle)
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        width, height = img.size
    except Exception as exc:
        return False, f"decode_failed: {exc}"

    # Supported format check
    if fmt not in SUPPORTED_FORMATS:
        return False, f"unsupported_format: {fmt}"

    # Minimum resolution check
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        return False, f"too_small: {width}x{height} < {MIN_DIMENSION}px"

    logger.debug(f"checks passed: {storage_path} ({fmt} {width}x{height})")
    return True, ""
