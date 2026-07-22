"""Shared types for bubble-first vision pipeline (app-agnostic)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


Point = List[float]  # [x, y]
BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2
EditorRegion = BBox  # skipped clean region [x1, y1, x2, y2]


class Bubble(BaseModel):
    """One speech bubble from YOLO-seg (+ optional OCR fields)."""

    id: int
    polygon: List[Point] = Field(..., description="Mask contour [[x,y], ...] in image coords")
    bbox: BBox = Field(..., description="Axis-aligned [x1, y1, x2, y2]")
    confidence: float = 0.0
    cls: int = 0
    text_ja: Optional[str] = None
    ocr_confidence: Optional[float] = None
