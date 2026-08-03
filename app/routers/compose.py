from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from vision.document import PageDoc
from vision.layers.compose import compose
from vision.typeset.catalog import resolve_font_path

from ..images import decode_image_b64, decode_image_bytes, encode_png_b64, encode_png_bytes
from ..settings import get_settings

router = APIRouter(prefix="/v1", tags=["compose"])


def _resolve_compose_font(doc: PageDoc, font: str | None) -> str | None:
    """Request font > first text layer font_file > global default."""
    settings = get_settings()
    if font and str(font).strip():
        try:
            return str(resolve_font_path(font, default=settings.typeset_font))
        except FileNotFoundError as e:
            raise HTTPException(400, str(e)) from e

    for layer in doc.layers:
        if layer.kind == "text" and layer.layout and layer.layout.font_file:
            path = Path(layer.layout.font_file)
            if path.is_file():
                return str(path.resolve())
            # Relative / basename fallback via catalog
            try:
                return str(
                    resolve_font_path(path.name, default=settings.typeset_font)
                )
            except FileNotFoundError:
                continue

    default = settings.typeset_font
    return str(default) if default.is_file() else None


@router.post("/compose")
async def compose_page(
    file: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    doc_json: str = Form(...),
    as_base64: bool = Form(False),
    off: str | None = Form(None, description="Comma-separated layer ids to disable"),
    only_kind: str | None = Form(None, description="clean | text"),
    font: str | None = Form(None, description="Catalog key or font filename override"),
):
    """Compose source image + PageDoc layers → preview PNG."""
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
        doc = PageDoc.model_validate(json.loads(doc_json))
    except Exception as e:
        raise HTTPException(400, f"Invalid doc_json: {e}") from e

    off_ids = {s.strip() for s in (off or "").split(",") if s.strip()}
    kind = only_kind.strip() if only_kind and only_kind.strip() else None
    if kind and kind not in {"clean", "text"}:
        raise HTTPException(400, "only_kind must be 'clean' or 'text'")

    font_path = _resolve_compose_font(doc, font)
    try:
        preview = compose(
            img,
            doc,
            font_path=font_path,
            off=off_ids or None,
            only_kind=kind,
        )
    except Exception as e:
        raise HTTPException(400, f"Compose failed: {e}") from e

    if as_base64:
        return {
            "page_id": doc.page_id,
            "width": int(preview.shape[1]),
            "height": int(preview.shape[0]),
            "image": encode_png_b64(preview),
        }
    return Response(content=encode_png_bytes(preview), media_type="image/png")
