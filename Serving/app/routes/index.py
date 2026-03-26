import io
from fastapi import APIRouter, Request, UploadFile, File, Form
from pydantic import BaseModel

from app.services import qdrant_client as qdrant
from app.services.qdrant_client import get_client
from PIL import Image

router = APIRouter()


class IndexResponse(BaseModel):
    image_id: str
    status: str


@router.post("/index", response_model=IndexResponse)
async def index_photo(
    request: Request,
    image_id: str = Form(...),
    file: UploadFile = File(...),
):
    embedder = request.app.state.embedder
    client = get_client()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    embedding = embedder.embed_image(image)

    qdrant.ensure_collection(client)
    qdrant.upsert_photo(
        client=client,
        image_id=image_id,
        embedding=embedding,
        payload={"filename": file.filename},
    )

    return {"image_id": image_id, "status": "indexed"}
