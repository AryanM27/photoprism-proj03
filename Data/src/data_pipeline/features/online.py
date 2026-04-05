"""
online.py — PhotoPrism Features API.

Endpoints:
    POST /search    — text query → Qdrant ANN → ranked image results
    GET  /healthz   — liveness probe
    POST /features  — image → CLIP embedding + aesthetic score

Note: CLIPEncoder is stubbed until training lead provides model via MLflow artifacts.
      The stub returns deterministic normalised vectors for shape correctness.
"""
import base64
import hashlib
import io
import os
import time
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class _StubEncoder:
    """Placeholder until training lead provides model via MLflow artifacts.
    Returns a deterministic normalised 512-dim vector derived from the input hash.
    Ensures /search returns results in correct shape without loading sentence-transformers.
    """
    VECTOR_DIM = 512

    def encode_text(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.random(self.VECTOR_DIM).astype("float32")
        return vec / np.linalg.norm(vec)


class _StubStore:
    """Thin Qdrant client wrapper for /search only — no sentence-transformers dependency."""

    def __init__(self, host: str, port: int, collection: str):
        self._client = QdrantClient(host=host, port=port)
        self.collection = collection

    def ensure_collection(self, vector_size: int = 512) -> None:
        from qdrant_client.http.exceptions import UnexpectedResponse
        try:
            self._client.get_collection(self.collection)
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            else:
                raise

    def search(self, query_vector: np.ndarray, top_k: int = 10,
               filter_: dict | None = None) -> list[dict]:
        hits = self._client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=filter_,
            with_payload=True,
        )
        return [
            {"image_id": h.payload.get("image_id"), "score": h.score, **h.payload}
            for h in hits
        ]


app = FastAPI(title="PhotoPrism Features API")

_encoder: _StubEncoder | None = None
_store: _StubStore | None = None


def _get_encoder() -> _StubEncoder:
    global _encoder
    if _encoder is None:
        _encoder = _StubEncoder()
    return _encoder


def _get_store() -> _StubStore:
    global _store
    if _store is None:
        _store = _StubStore(
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
            response = requests.get(payload.image_url, timeout=10)
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
    # Stub: returns a normalised uniform vector until CLIP model is provided via MLflow
    vec = np.ones(512, dtype="float32")
    return (vec / np.linalg.norm(vec)).tolist()


def _compute_aesthetic_score(img: Image.Image) -> float:
    # Stub: returns 0.5 until aesthetic model is wired in
    return 0.5
