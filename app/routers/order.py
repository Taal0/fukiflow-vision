from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from vision.bubble.detect import get_detector
from vision.bubble.draw import draw_bubbles
from vision.bubble.ink_filter import classify, settings_from_app
from vision.order.service import get_service as get_order_service

from ..images import decode_image_b64, decode_image_bytes, encode_png_b64
from ..schemas import OrderResponse
from ..settings import get_settings

router = APIRouter(prefix="/v1", tags=["order"])


@router.post("/order", response_model=OrderResponse)
async def reading_order(
    file: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    page_id: str | None = Form(None),
    include_overlay: bool = Form(True),
) -> OrderResponse:
    """bubbles → ink_filter → panel detect + RSP reading order."""
    if file is None and not image_b64:
        raise HTTPException(400, "Provide multipart `file` or form field `image_b64`")

    try:
        if file is not None:
            img = decode_image_bytes(await file.read())
        else:
            img = decode_image_b64(image_b64 or "")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    raw = get_detector().predict(img)
    ink_settings = settings_from_app(get_settings())
    bubbles, _force_editor = classify(img, raw, ink_settings)
    result = get_order_service().order(img, bubbles)

    overlay_b64 = None
    if include_overlay:
        overlay = draw_bubbles(img, result.bubbles, panels=result.panels)
        overlay_b64 = encode_png_b64(overlay)

    return OrderResponse(
        page_id=page_id or str(uuid.uuid4()),
        width=int(img.shape[1]),
        height=int(img.shape[0]),
        bubbles=result.bubbles,
        panels=result.panels,
        order_confidence=result.order_confidence,
        needs_review=result.needs_review,
        bubbles_overlay=overlay_b64,
    )
