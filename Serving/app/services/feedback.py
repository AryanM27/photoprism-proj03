import logging
import os
import uuid
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", "clip-ViT-B-32-v1")

_engine = None
_Session = None


def init_db():
    global _engine, _Session
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set — feedback logging disabled")
        return
    _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    _Session = sessionmaker(bind=_engine)
    logger.info("Feedback DB connected: %s", DATABASE_URL)


def log_search(query: str, results: list[dict]):
    if _Session is None:
        return
    query_id = str(uuid.uuid4())
    try:
        session = _Session()
        for rank, item in enumerate(results):
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
                    "query_id": query_id,
                    "image_id": item.get("image_id", ""),
                    "shown_rank": rank,
                    "clicked": False,
                    "favorited": False,
                    "semantic_score": float(item.get("score", 0.0)),
                    "model_version": MODEL_VERSION,
                    "timestamp": datetime.utcnow(),
                },
            )
        session.commit()
        session.close()
    except Exception as e:
        logger.error("Failed to log feedback: %s", e)
