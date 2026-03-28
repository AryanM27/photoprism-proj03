import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.routes import health, index, search
from app.services.embedder import Embedder
from app.services.ranker import AestheticRanker
from app.services.qdrant_client import ensure_collection, get_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once on startup
    app.state.embedder = Embedder(
        model_name=os.getenv("MODEL_NAME", "ViT-B-32"),
        device_str=os.getenv("DEVICE", "auto"),
        checkpoint_path=os.getenv("CHECKPOINT_PATH", None),
    )
    app.state.ranker = AestheticRanker(device_str=os.getenv("DEVICE", "auto"))
    # Make sure Qdrant collection exists
    ensure_collection(get_client())
    yield


app = FastAPI(title="PhotoPrism Semantic Search", lifespan=lifespan)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

app.include_router(health.router)
app.include_router(search.router)
app.include_router(index.router)
