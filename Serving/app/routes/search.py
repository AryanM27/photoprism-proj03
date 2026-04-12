from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.services import qdrant_client as qdrant
from app.services.qdrant_client import get_client
from app.services import feedback

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

    query_embedding = embedder.embed_text(req.query)
    results = qdrant.search_photos(client, query_embedding, top_k=req.top_k)

    if req.rerank and results:
        image_bytes_map = {}
        for item in results:
            path = item["payload"].get("filepath") or item["payload"].get("filename")
            if path:
                try:
                    with open(path, "rb") as f:
                        image_bytes_map[item["image_id"]] = f.read()
                except (FileNotFoundError, OSError):
                    pass
        results = ranker.rerank(results, image_bytes_map)

    feedback.log_search(req.query, results)

    return results
