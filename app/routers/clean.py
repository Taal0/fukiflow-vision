from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..images import decode_image_b64, decode_image_bytes, encode_png_b64, encode_png_bytes
from ..schemas import CleanResponse

router = APIRouter(prefix="/v1", tags=["clean"])

# Passive stub: real inset-polygon clean is a follow-up revision.
CLEAN_STATUS = "disabled"


@router.post("/clean")
async def clean(
    file: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    bubbles_json: str | None = Form(None),
    page_id: str | None = Form(None),
    as_base64: bool = Form(False),
):
    """No-op clean: returns the original image. editor_regions always [].

    `bubbles_json` is accepted for forward compatibility but ignored.
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

    # Validate optional bubbles_json shape early so clients get 400 not silent drop.
    if bubbles_json:
        try:
            data = json.loads(bubbles_json)
            if not isinstance(data, list):
                raise ValueError("must be a JSON array")
        except Exception as e:
            raise HTTPException(400, f"Invalid bubbles_json: {e}") from e

    editor_regions: list = []
    if as_base64:
        return CleanResponse(
            page_id=page_id,
            cleaned_image=encode_png_b64(img),
            editor_regions=editor_regions,
            clean=CLEAN_STATUS,
        )
    return Response(
        content=encode_png_bytes(img),
        media_type="image/png",
        headers={
            "X-Clean": CLEAN_STATUS,
            "X-Editor-Regions": json.dumps(editor_regions),
        },
    )
