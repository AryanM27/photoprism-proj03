"""Feedback endpoints for recording user likes on semantic search results."""
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import feedback as fb

logger = logging.getLogger(__name__)

router = APIRouter()


class LikeRequest(BaseModel):
    image_id: str
    query: str = ""
    score: float = 0.0


@router.post("/feedback/like", status_code=204)
def like_image(req: LikeRequest):
    if fb._Session is None:
        raise HTTPException(status_code=503, detail="feedback DB not available")
    try:
        session = fb._Session()
        session.execute(
            __import__("sqlalchemy").text("""
                INSERT INTO feedback_events
                    (event_id, user_id, query_id, image_id, shown_rank,
                     clicked, favorited, semantic_score,
                     model_version, timestamp)
                VALUES
                    (:event_id, :user_id, :query_id, :image_id, :shown_rank,
                     :clicked, :favorited, :semantic_score,
                     :model_version, :timestamp)
            """),
            {
                "event_id": str(uuid.uuid4()),
                "user_id": "anonymous",
                "query_id": str(uuid.uuid4()),
                "image_id": req.image_id,
                "shown_rank": 0,
                "clicked": True,
                "favorited": True,
                "semantic_score": req.score,
                "model_version": fb.MODEL_VERSION,
                "timestamp": datetime.utcnow(),
            },
        )
        session.commit()
        session.close()
    except Exception as exc:
        logger.error("Failed to record like for %s: %s", req.image_id, exc)
        raise HTTPException(status_code=500, detail="failed to record like")
