"""
upload_notify.py — POST /upload/notify

Called by PhotoPrism's ProcessUserUpload (Go) synchronously before imp.Start()
moves files out of staging. Reads image files from the shared storage volume,
uploads each to S3, registers in Postgres, and dispatches the validation Celery task.
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.ingestion_bridge import ingest_staged_files

logger = logging.getLogger(__name__)

router = APIRouter()

STORAGE_PATH = os.environ.get("PHOTOPRISM_STORAGE_PATH", "/photoprism/storage")


class UploadNotifyRequest(BaseModel):
    user_id: str
    staging_path: str


class UploadNotifyResponse(BaseModel):
    processed: int
    failed: int


@router.post("/upload/notify", response_model=UploadNotifyResponse)
def upload_notify(req: UploadNotifyRequest) -> UploadNotifyResponse:
    # Reject paths that escape the known storage root (path traversal guard).
    try:
        resolved = Path(req.staging_path).resolve()
        storage_root = Path(STORAGE_PATH).resolve()
        resolved.relative_to(storage_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="staging_path is outside storage root")

    if not resolved.is_dir():
        raise HTTPException(
            status_code=400,
            detail="staging_path does not exist or is not a directory",
        )

    processed, failed = ingest_staged_files(req.user_id, str(resolved))
    logger.info(
        "upload/notify: user=%s staging=%s processed=%d failed=%d",
        req.user_id,
        resolved,
        processed,
        failed,
    )
    return UploadNotifyResponse(processed=processed, failed=failed)
