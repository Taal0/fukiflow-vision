from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from vision.bubble.detect import get_detector
from vision.bubble.ink_filter import classify, settings_from_app

from ..images import decode_image_b64, decode_image_bytes
from ..schemas import BubblesResponse
from ..settings import get_settings

router = APIRouter(prefix="/v1", tags=["bubbles"])


@router.post("/bubbles", response_model=BubblesResponse)
async def detect_bubbles(
    file: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    page_id: str | None = Form(None),
) -> BubblesResponse:
    if file is None and not image_b64:
        raise HTTPException(400, "Provide multipart `file` or form field `image_b64`")

    try:
        if file is not None:
            img = decode_image_bytes(await file.read())
        else:
            img = decode_image_b64(image_b64 or "")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    detector = get_detector()
    raw = detector.predict(img)
    ink_settings = settings_from_app(get_settings())
    bubbles, _force_editor = classify(img, raw, ink_settings)
    return BubblesResponse(
        page_id=page_id or str(uuid.uuid4()),
        width=int(img.shape[1]),
        height=int(img.shape[0]),
        bubbles=bubbles,
    )
