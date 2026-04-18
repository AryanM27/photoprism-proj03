import logging

import numpy as np

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.services import qdrant_client as qdrant
from app.services.qdrant_client import get_client
from app.services import feedback
from app.services.image_fetcher import fetch_image_bytes, resolve_storage_path
from app.metrics import (
    INFERENCE_SEARCH_TOTAL,
    INFERENCE_EMPTY_RESULTS_TOTAL,
    INFERENCE_QUERY_LENGTH_CHARS,
    INFERENCE_QUERY_EMBEDDING_NORM,
    INFERENCE_TOP_K_SCORE,
    INFERENCE_SCORE_SPREAD,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    rerank: bool = True


class SearchResult(BaseModel):
    image_id: str
    score: float
    aesthetic_score: float | None = None
    payload: dict


@router.post("/search", response_model=list[SearchResult])
def search(req: TextSearchRequest, request: Request):
    embedder = request.app.state.embedder
    ranker = request.app.state.ranker
    client = get_client()

    # Checkpoint 3 — query characteristics
    INFERENCE_QUERY_LENGTH_CHARS.observe(len(req.query))

    try:
        query_embedding = embedder.embed_text(req.query)
        # Embedding norm should be ~1.0 if normalised; drift here signals upstream change
        INFERENCE_QUERY_EMBEDDING_NORM.observe(float(np.linalg.norm(query_embedding)))

        results = qdrant.search_photos(client, query_embedding, top_k=req.top_k)

        # Record vector-stage scores
        for item in results:
            INFERENCE_TOP_K_SCORE.labels(stage="vector").observe(float(item["score"]))

        if not results:
            INFERENCE_EMPTY_RESULTS_TOTAL.inc()
            INFERENCE_SEARCH_TOTAL.labels(result="empty").inc()
            feedback.log_search(req.query, results)
            return results

        if req.rerank and results:
            image_bytes_map = {}
            for item in results:
                sp = resolve_storage_path(item["image_id"], item["payload"])
                if sp:
                    try:
                        image_bytes_map[item["image_id"]] = fetch_image_bytes(sp)
                    except Exception as exc:
                        logger.warning("Could not fetch image %s from S3: %s", item["image_id"], exc)
            results = ranker.rerank(results, image_bytes_map)

        # Record rerank-stage scores and spread
        scores = [float(item["score"]) for item in results]
        for s in scores:
            INFERENCE_TOP_K_SCORE.labels(stage="rerank").observe(s)
        if len(scores) >= 2:
            INFERENCE_SCORE_SPREAD.observe(scores[0] - scores[-1])

        INFERENCE_SEARCH_TOTAL.labels(result="ok").inc()
        feedback.log_search(req.query, results)
        return results

    except Exception:
        INFERENCE_SEARCH_TOTAL.labels(result="error").inc()
        raise
