from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from vision.bubble.detect import get_detector
from vision.ocr.manga import get_ocr

from ..images import decode_image_b64, decode_image_bytes, encode_png_b64
from ..schemas import PipelineResponse
from .clean import CLEAN_STATUS

router = APIRouter(prefix="/v1", tags=["pipeline"])


@router.post("/pipeline", response_model=PipelineResponse)
async def pipeline(
    file: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    page_id: str | None = Form(None),
) -> PipelineResponse:
    """bubbles → ocr → clean(no-op). Single call for editor/batch clients."""
    if file is None and not image_b64:
        raise HTTPException(400, "Provide multipart `file` or form field `image_b64`")

    try:
        if file is not None:
            img = decode_image_bytes(await file.read())
        else:
            img = decode_image_b64(image_b64 or "")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    bubbles = get_detector().predict(img)
    bubbles = get_ocr().recognize(img, bubbles)

    return PipelineResponse(
        page_id=page_id or str(uuid.uuid4()),
        width=int(img.shape[1]),
        height=int(img.shape[0]),
        bubbles=bubbles,
        cleaned_image=encode_png_b64(img),
        editor_regions=[],
        clean=CLEAN_STATUS,
    )
