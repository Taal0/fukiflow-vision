from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from vision.layers.build import text_layers
from vision.types import Bubble, BubbleText
from vision.typeset.draw import draw_typeset
from vision.typeset.service import get_service as get_typeset_service

from ..images import decode_image_b64, decode_image_bytes, encode_png_b64
from ..schemas import TypesetResponse
from ..settings import get_settings

router = APIRouter(prefix="/v1", tags=["typeset"])


def _parse_bubbles_json(raw: str) -> List[Bubble]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("bubbles_json must be a JSON array")
    return [Bubble.model_validate(b) for b in data]


def _parse_texts_json(raw: str) -> List[BubbleText]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("texts_json must be a JSON array")
    return [BubbleText.model_validate(t) for t in data]


@router.post("/typeset", response_model=TypesetResponse)
async def typeset(
    file: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    bubbles_json: str = Form(...),
    texts_json: str = Form(...),
    page_id: str | None = Form(None),
    include_debug: bool = Form(False),
    include_layers: bool = Form(False),
) -> TypesetResponse:
    """Layout JSON for cleaned bubbles + texts. Optional debug PNG / text layers."""
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
        texts = _parse_texts_json(texts_json)
    except Exception as e:
        raise HTTPException(400, f"Invalid bubbles_json/texts_json: {e}") from e

    result = get_typeset_service().typeset(img, bubbles, texts)

    debug_b64 = None
    if include_debug:
        settings = get_settings()
        debug = draw_typeset(
            img,
            result.layouts,
            bubbles,
            font_path=str(settings.typeset_font),
        )
        debug_b64 = encode_png_b64(debug)

    layers = text_layers(result.layouts, bubbles) if include_layers else []

    return TypesetResponse(
        page_id=page_id,
        width=int(img.shape[1]),
        height=int(img.shape[0]),
        typeset=result.status,
        typeset_regions=result.typeset_regions,
        layouts=result.layouts,
        text_layers=layers,
        debug_image=debug_b64,
    )
