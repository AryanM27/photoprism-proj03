import io
import logging

from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from pydantic import BaseModel

from app.services.image_fetcher import fetch_image_bytes

logger = logging.getLogger(__name__)

router = APIRouter()


class EmbedImageRequest(BaseModel):
    storage_path: str


class EmbedImageResponse(BaseModel):
    embedding: list[float]
    model_version: str


@router.post("/embed/image", response_model=EmbedImageResponse)
def embed_image(req: EmbedImageRequest, request: Request):
    embedder = request.app.state.embedder
    try:
        image_bytes = fetch_image_bytes(req.storage_path)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        embedding = embedder.embed_image(image)
    except Exception as exc:
        logger.error("Failed to embed image %s: %s", req.storage_path, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return EmbedImageResponse(embedding=embedding, model_version=embedder.model_name)
