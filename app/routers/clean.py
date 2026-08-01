from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from vision.clean.fill import get_cleaner
from vision.layers.build import clean_layers
from vision.types import Bubble

from ..images import decode_image_b64, decode_image_bytes, encode_png_b64, encode_png_bytes
from ..schemas import CleanResponse

router = APIRouter(prefix="/v1", tags=["clean"])


def _parse_bubbles_json(raw: str) -> List[Bubble]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("bubbles_json must be a JSON array")
    return [Bubble.model_validate(b) for b in data]


@router.post("/clean")
async def clean(
    file: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    bubbles_json: str = Form(...),
    page_id: str | None = Form(None),
    as_base64: bool = Form(False),
    include_layers: bool = Form(False),
):
    """Inset-fill clean from bubble polygons. Returns PNG + editor_regions + status."""
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

    outcome = get_cleaner().clean(img, bubbles)
    layers = clean_layers(outcome.fills) if include_layers else []

    if as_base64 or include_layers:
        return CleanResponse(
            page_id=page_id,
            cleaned_image=encode_png_b64(outcome.image),
            editor_regions=outcome.editor_regions,
            clean=outcome.status,
            clean_layers=layers,
        )
    return Response(
        content=encode_png_bytes(outcome.image),
        media_type="image/png",
        headers={
            "X-Clean": outcome.status,
            "X-Editor-Regions": json.dumps(outcome.editor_regions),
            "X-Editor-Bubble-Ids": json.dumps(outcome.editor_bubble_ids),
        },
    )
