"""Shared types for bubble-first vision pipeline (app-agnostic)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


Point = List[float]  # [x, y]
BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2
EditorRegion = BBox  # skipped clean region [x1, y1, x2, y2]


class Panel(BaseModel):
    """One manga panel from YOLO detect (bbox-only)."""

    id: int
    bbox: BBox = Field(..., description="Axis-aligned [x1, y1, x2, y2]")
    confidence: float = 0.0
    reading_order: int = 0  # 1-based among panels
    is_synthetic: bool = False  # full-page fallback when none detected


class Bubble(BaseModel):
    """One speech bubble from YOLO-seg."""

    id: int
    polygon: List[Point] = Field(..., description="Mask contour [[x,y], ...] in image coords")
    bbox: BBox = Field(..., description="Axis-aligned [x1, y1, x2, y2]")
    confidence: float = 0.0
    cls: int = 0
    # Optional reading-order fields (defaults preserve /v1/clean bubbles_json compat)
    panel_id: Optional[int] = None
    assign_mode: str = "unassigned"  # contained|partial|orphan|ambiguous|no_panel
    reading_order: Optional[int] = None  # 1..N global
    order_source: str = "unordered"  # algo|ai|manual
    order_confidence: float = 0.0
