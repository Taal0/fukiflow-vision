from __future__ import annotations

from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field


Point = List[float]  # [x, y]
Quad = List[Point]  # 4 points


class Box(BaseModel):
    """One text region after detect+OCR (+ textline merge)."""

    polygon: Union[Quad, List[Quad]] = Field(
        ...,
        description="Single quad [[x,y]x4] or list of quads (one per textline)",
    )
    bbox: Tuple[int, int, int, int] = Field(..., description="Axis-aligned [x1, y1, x2, y2]")
    text_ja: str = ""
    confidence: float = 0.0


class DetectOcrResponse(BaseModel):
    page_id: str
    boxes: List[Box]
    mask: str = Field(..., description="Base64-encoded PNG mask for /v1/inpaint")
    width: int
    height: int


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool


class InpaintJsonRequest(BaseModel):
    image_b64: str
    mask_b64: Optional[str] = None
    boxes: Optional[List[Box]] = None
    page_id: Optional[str] = None


class InpaintResponse(BaseModel):
    page_id: Optional[str] = None
    cleaned_image: str = Field(..., description="Base64-encoded PNG")
    editor_regions: List[Tuple[int, int, int, int]] = Field(
        default_factory=list,
        description="Bboxes [x1,y1,x2,y2] left for the editor (solidfill skip)",
    )
