import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.routes import health, index, search
from app.services.embedder import Embedder
from app.services.embedder_onnx import OnnxEmbedder
from app.services.ranker import AestheticRanker
from app.services.qdrant_client import ensure_collection, get_client
from app.services import feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    feedback.init_db()
    use_onnx = os.getenv("USE_ONNX", "false").lower() == "true"
    model_name = os.getenv("MODEL_NAME", "clip-ViT-B-32")

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
            checkpoint_path=os.getenv("CHECKPOINT_PATH", None),
        )

    app.state.ranker = AestheticRanker(
        device_str=os.getenv("DEVICE", "auto"),
        checkpoint_path=os.getenv("AESTHETIC_CHECKPOINT_PATH", None),
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
