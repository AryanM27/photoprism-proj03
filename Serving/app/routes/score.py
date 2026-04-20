"""POST /score/aesthetic — score a single image's aesthetic quality."""
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.services.image_fetcher import fetch_image_bytes

logger = logging.getLogger(__name__)

router = APIRouter()


class ScoreRequest(BaseModel):
    s3_path: str


class ScoreResponse(BaseModel):
    s3_path: str
    aesthetic_score: float


@router.post("/score/aesthetic", response_model=ScoreResponse)
async def score_aesthetic(req: ScoreRequest, request: Request):
    ranker = request.app.state.ranker
    if ranker is None:
        raise HTTPException(status_code=503, detail="Aesthetic ranker not loaded")

    try:
        image_bytes = fetch_image_bytes(req.s3_path)
    except Exception as exc:
        logger.warning("Failed to fetch image %s: %s", req.s3_path, exc)
        raise HTTPException(status_code=502, detail=f"Failed to fetch image: {exc}")

    try:
        score = ranker.score_image_bytes(image_bytes)
    except Exception as exc:
        logger.warning("Failed to score image %s: %s", req.s3_path, exc)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

    return ScoreResponse(s3_path=req.s3_path, aesthetic_score=score)
