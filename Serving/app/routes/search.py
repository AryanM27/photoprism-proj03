from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.services import qdrant_client as qdrant
from app.services.qdrant_client import get_client

router = APIRouter()


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    image_id: str
    score: float
    payload: dict


@router.post("/search", response_model=list[SearchResult])
def search(req: TextSearchRequest, request: Request):
    embedder = request.app.state.embedder
    client = get_client()

    query_embedding = embedder.embed_text(req.query)
    results = qdrant.search_photos(client, query_embedding, top_k=req.top_k)

    return results
