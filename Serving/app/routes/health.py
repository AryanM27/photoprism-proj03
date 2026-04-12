from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    embedder = getattr(request.app.state, "embedder", None)
    model_name = embedder.model_name if embedder else "unknown"
    return {"status": "ok", "model": model_name}
