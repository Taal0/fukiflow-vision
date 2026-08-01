from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from vision.bubble.detect import get_detector
from vision.bubble.draw import draw_bubbles
from vision.bubble.ink_filter import classify, settings_from_app
from vision.clean.fill import get_cleaner
from vision.layers.build import clean_layers
from vision.order.service import get_service as get_order_service

from ..images import decode_image_b64, decode_image_bytes, encode_png_b64
from ..schemas import PipelineResponse
from ..settings import get_settings

router = APIRouter(prefix="/v1", tags=["pipeline"])


@router.post("/pipeline", response_model=PipelineResponse)
async def pipeline(
    file: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    page_id: str | None = Form(None),
    include_layers: bool = Form(False),
) -> PipelineResponse:
    """bubbles → ink_filter → reading order → inset-fill clean."""
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
    bubbles, force_editor = classify(img, raw, ink_settings)
    result = get_order_service().order(img, bubbles)
    pid = page_id or str(uuid.uuid4())
    outcome = get_cleaner().clean(
        img, result.bubbles, force_editor=force_editor
    )
    overlay = draw_bubbles(img, outcome.bubbles, panels=result.panels)
    layers = clean_layers(outcome.fills) if include_layers else []

    return PipelineResponse(
        page_id=pid,
        width=int(img.shape[1]),
        height=int(img.shape[0]),
        bubbles=outcome.bubbles,
        panels=result.panels,
        order_confidence=result.order_confidence,
        needs_review=result.needs_review,
        cleaned_image=encode_png_b64(outcome.image),
        bubbles_overlay=encode_png_b64(overlay),
        editor_regions=outcome.editor_regions,
        clean=outcome.status,
        clean_layers=layers,
    )
