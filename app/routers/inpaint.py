from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..engine import get_engine
from ..images import (
    boxes_to_mask,
    decode_image_b64,
    decode_image_bytes,
    decode_mask_b64,
    encode_png_b64,
    encode_png_bytes,
)
from ..schemas import Box, InpaintJsonRequest, InpaintResponse

router = APIRouter(prefix="/v1", tags=["inpaint"])


@router.post("/inpaint")
async def inpaint(
    file: UploadFile | None = File(None),
    mask: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    mask_b64: str | None = Form(None),
    boxes_json: str | None = Form(None),
    page_id: str | None = Form(None),
    as_base64: bool = Form(False),
):
    """Inpaint text regions.

    Provide the page image via `file` or `image_b64`, and either:
    - `mask` / `mask_b64` (PNG from /detect-ocr), or
    - `boxes_json` (JSON list of Box objects; mask is rasterized from polygons).
    """
    if file is None and not image_b64:
        raise HTTPException(400, "Provide multipart `file` or form field `image_b64`")

    try:
        if file is not None:
            img = decode_image_bytes(await file.read())
        else:
            img = decode_image_b64(image_b64 or "")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    mask_arr = None
    if mask is not None:
        try:
            from ..images import decode_image_bytes as _dec
            import cv2
            import numpy as np

            raw = await mask.read()
            arr = np.frombuffer(raw, dtype=np.uint8)
            mask_arr = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if mask_arr is None:
                raise ValueError("decode failed")
        except Exception as e:
            raise HTTPException(400, f"Invalid mask file: {e}") from e
    elif mask_b64:
        try:
            mask_arr = decode_mask_b64(mask_b64)
        except Exception as e:
            raise HTTPException(400, f"Invalid mask_b64: {e}") from e
    elif boxes_json:
        try:
            boxes = [Box.model_validate(b) for b in json.loads(boxes_json)]
            mask_arr = boxes_to_mask(img.shape[:2], boxes)
        except Exception as e:
            raise HTTPException(400, f"Invalid boxes_json: {e}") from e
    else:
        raise HTTPException(400, "Provide `mask`/`mask_b64` or `boxes_json`")

    eng = get_engine()
    try:
        cleaned = await eng.inpaint(img, mask_arr)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    if as_base64:
        return InpaintResponse(page_id=page_id, cleaned_image=encode_png_b64(cleaned))
    return Response(content=encode_png_bytes(cleaned), media_type="image/png")


@router.post("/inpaint/json", response_model=InpaintResponse)
async def inpaint_json(body: InpaintJsonRequest) -> InpaintResponse:
    try:
        img = decode_image_b64(body.image_b64)
    except Exception as e:
        raise HTTPException(400, f"Invalid image_b64: {e}") from e

    if body.mask_b64:
        mask_arr = decode_mask_b64(body.mask_b64)
    elif body.boxes:
        mask_arr = boxes_to_mask(img.shape[:2], body.boxes)
    else:
        raise HTTPException(400, "Provide mask_b64 or boxes")

    eng = get_engine()
    cleaned = await eng.inpaint(img, mask_arr)
    return InpaintResponse(page_id=body.page_id, cleaned_image=encode_png_b64(cleaned))
