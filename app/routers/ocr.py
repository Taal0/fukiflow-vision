from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from vision.ocr.manga import get_ocr
from vision.types import Bubble

from ..images import decode_image_b64, decode_image_bytes
from ..schemas import OcrResponse

router = APIRouter(prefix="/v1", tags=["ocr"])


def _parse_bubbles_json(raw: str) -> List[Bubble]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("bubbles_json must be a JSON array")
    return [Bubble.model_validate(b) for b in data]


@router.post("/ocr", response_model=OcrResponse)
async def run_ocr(
    file: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    bubbles_json: str = Form(...),
    page_id: str | None = Form(None),
) -> OcrResponse:
    """OCR each bubble crop. Provide page image + bubbles_json from /v1/bubbles."""
    if file is None and not image_b64:
        raise HTTPException(400, "Provide multipart `file` or form field `image_b64`")

    try:
        if file is not None:
            img = decode_image_bytes(await file.read())
        else:
            img = decode_image_b64(image_b64 or "")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    try:
        bubbles = _parse_bubbles_json(bubbles_json)
    except Exception as e:
        raise HTTPException(400, f"Invalid bubbles_json: {e}") from e

    ocr = get_ocr()
    filled = ocr.recognize(img, bubbles)
    return OcrResponse(
        page_id=page_id,
        width=int(img.shape[1]),
        height=int(img.shape[0]),
        bubbles=filled,
    )
