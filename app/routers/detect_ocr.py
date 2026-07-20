from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..engine import get_engine
from ..images import decode_image_b64, decode_image_bytes, encode_png_b64
from ..schemas import DetectOcrResponse

router = APIRouter(prefix="/v1", tags=["detect-ocr"])


@router.post("/detect-ocr", response_model=DetectOcrResponse)
async def detect_ocr(
    file: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    page_id: str | None = Form(None),
) -> DetectOcrResponse:
    if file is None and not image_b64:
        raise HTTPException(400, "Provide multipart `file` or form field `image_b64`")

    try:
        if file is not None:
            data = await file.read()
            img = decode_image_bytes(data)
        else:
            img = decode_image_b64(image_b64 or "")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    eng = get_engine()
    result = await eng.detect_ocr(img, page_id=page_id)
    return DetectOcrResponse(
        page_id=result.page_id,
        boxes=result.boxes,
        mask=encode_png_b64(result.mask),
        width=result.width,
        height=result.height,
    )
