"""Feedback endpoints for recording user likes and clicks on semantic search results."""
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from app.services import feedback as fb

logger = logging.getLogger(__name__)

router = APIRouter()


class FeedbackRequest(BaseModel):
    image_id: str
    query: str = ""
    score: float = 0.0


def _insert_feedback(image_id: str, query: str, score: float, clicked: bool, favorited: bool):
    if fb._Session is None:
        raise HTTPException(status_code=503, detail="feedback DB not available")
    try:
        session = fb._Session()
        session.execute(
            text("""
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
                "image_id": image_id,
                "shown_rank": 0,
                "clicked": clicked,
                "favorited": favorited,
                "semantic_score": float(score),
                "model_version": fb.MODEL_VERSION,
                "timestamp": datetime.utcnow(),
            },
        )
        session.commit()
        session.close()
    except Exception as exc:
        logger.error("Failed to record feedback for %s: %s", image_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/feedback/like")
def like_image(req: FeedbackRequest):
    _insert_feedback(req.image_id, req.query, req.score, clicked=True, favorited=True)
    logger.info("Recorded like for image %s", req.image_id)
    return {"liked": True, "image_id": req.image_id}


@router.post("/feedback/click")
def click_image(req: FeedbackRequest):
    _insert_feedback(req.image_id, req.query, req.score, clicked=True, favorited=False)
    logger.info("Recorded click for image %s", req.image_id)
    return {"clicked": True, "image_id": req.image_id}
