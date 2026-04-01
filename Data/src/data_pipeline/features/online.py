"""
online.py — PhotoPrism Features API.

Endpoints:
    POST /search    — text query → Qdrant ANN → ranked image results
    GET  /healthz   — liveness probe
    POST /features  — image → CLIP embedding + aesthetic score
"""
import base64
import io
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from PIL import Image

from src.data_pipeline.embeddings.encoder import CLIPEncoder
from src.data_pipeline.embeddings.qdrant_store import QdrantStore

app = FastAPI(title="PhotoPrism Features API")

_encoder: CLIPEncoder | None = None
_store: QdrantStore | None = None


def _get_encoder() -> CLIPEncoder:
    global _encoder
    if _encoder is None:
        _encoder = CLIPEncoder(model_name=os.environ.get("EMBEDDING_MODEL", "clip-ViT-B-32"))
    return _encoder


def _get_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore(
            host=os.environ["QDRANT_HOST"],
            port=int(os.environ["QDRANT_PORT"]),
            collection=os.environ["QDRANT_COLLECTION"],
        )
        _store.ensure_collection()
    return _store


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()


class SearchResult(BaseModel):
    image_id: str
    score: float
    aesthetic_score: float | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Text-to-image semantic search via Qdrant ANN."""
    vec = _get_encoder().encode_text(req.query)
    hits = _get_store().search(vec, top_k=req.top_k)
    return SearchResponse(results=[SearchResult(**h) for h in hits])


@app.get("/healthz")
def health():
    return {"status": "ok"}


class ImagePayload(BaseModel):
    image_id:  str
    image_b64: str | None = None
    image_url: str | None = None


class FeatureResponse(BaseModel):
    image_id:        str
    embedding:       list[float]
    aesthetic_score: float
    latency_ms:      float


@app.post("/features", response_model=FeatureResponse)
def compute_features(payload: ImagePayload) -> FeatureResponse:
    """Image → CLIP embedding + aesthetic score."""
    start = time.time()
    if not payload.image_b64 and not payload.image_url:
        raise HTTPException(status_code=400, detail="Provide image_b64 or image_url")
    if payload.image_b64:
        img = Image.open(io.BytesIO(base64.b64decode(payload.image_b64)))
    else:
        try:
            import requests as http
            response = http.get(payload.image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image_url: {e}")
    return FeatureResponse(
        image_id=payload.image_id,
        embedding=_compute_embedding(img),
        aesthetic_score=_compute_aesthetic_score(img),
        latency_ms=(time.time() - start) * 1000,
    )


def _compute_embedding(img: Image.Image) -> list[float]:
    # Stub: returns zero vector until CLIP inference is wired in
    return [0.0] * 512


def _compute_aesthetic_score(img: Image.Image) -> float:
    # Stub: returns 0.5 until aesthetic model is wired in
    return 0.5
