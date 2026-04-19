"""Serve an image by image_id, fetching from S3 via image_fetcher."""
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.services.image_fetcher import fetch_image_bytes, resolve_storage_path

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/image/{image_id}")
def get_image(image_id: str):
    storage_path = resolve_storage_path(image_id, {})
    if not storage_path:
        raise HTTPException(status_code=404, detail="image not found")
    try:
        data = fetch_image_bytes(storage_path)
    except Exception as exc:
        logger.warning("Failed to fetch image %s from S3: %s", image_id, exc)
        raise HTTPException(status_code=502, detail="failed to fetch image from storage")
    return Response(content=data, media_type="image/jpeg")
