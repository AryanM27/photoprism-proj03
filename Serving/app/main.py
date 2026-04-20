import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.routes import health, image, index, search, webhook
from app.routes import feedback as feedback_routes
from app.services.checkpoint_resolver import resolve_checkpoint
from app.services.embedder import Embedder
from app.services.embedder_onnx import OnnxEmbedder
from app.services.ranker import AestheticRanker
from app.services.qdrant_client import ensure_collection, get_client
from app.services import feedback
import app.metrics  # noqa: F401  — registers custom Prometheus metrics at import time


@asynccontextmanager
async def lifespan(app: FastAPI):
    feedback.init_db()
    use_onnx = os.getenv("USE_ONNX", "false").lower() == "true"
    model_name = os.getenv("MODEL_NAME", "clip-ViT-B-32")

    semantic_ckpt = resolve_checkpoint(os.getenv("CHECKPOINT_PATH"))
    aesthetic_ckpt = resolve_checkpoint(os.getenv("AESTHETIC_CHECKPOINT_PATH"))

    if use_onnx:
        app.state.embedder = OnnxEmbedder(
            model_name=model_name,
            text_onnx_path=os.getenv("ONNX_TEXT_PATH", "checkpoints/text_encoder.onnx"),
            image_onnx_path=os.getenv("ONNX_IMAGE_PATH", "checkpoints/image_encoder.onnx"),
        )
    else:
        app.state.embedder = Embedder(
            model_name=model_name,
            device_str=os.getenv("DEVICE", "auto"),
            checkpoint_path=semantic_ckpt,
        )

    app.state.ranker = AestheticRanker(
        device_str=os.getenv("DEVICE", "auto"),
        checkpoint_path=aesthetic_ckpt,
        model_type=os.getenv("AESTHETIC_MODEL_TYPE", "resnet18_linear"),
    )
    ensure_collection(get_client())
    yield


app = FastAPI(title="PhotoPrism Semantic Search", lifespan=lifespan)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

app.include_router(health.router)
app.include_router(search.router)
app.include_router(index.router)
app.include_router(image.router)
app.include_router(feedback_routes.router)
app.include_router(webhook.router)
