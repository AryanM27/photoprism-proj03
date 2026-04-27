import io
import logging

from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from pydantic import BaseModel

from app.services.image_fetcher import fetch_image_bytes

logger = logging.getLogger(__name__)

router = APIRouter()


class CaptionRequest(BaseModel):
    storage_path: str


class CaptionResponse(BaseModel):
    caption: str
    model_version: str


@router.post("/caption/image", response_model=CaptionResponse)
def caption_image(req: CaptionRequest, request: Request):
    captioner = request.app.state.captioner
    try:
        image_bytes = fetch_image_bytes(req.storage_path)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        caption = captioner.caption(image)
    except Exception as exc:
        logger.error("Failed to caption image %s: %s", req.storage_path, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return CaptionResponse(caption=caption, model_version=captioner.model_name)
