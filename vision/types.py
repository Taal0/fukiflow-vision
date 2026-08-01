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
    clean_status: str = "unknown"  # filled | editor | unknown


class TypesetLine(BaseModel):
    """One hard-broken line inside a bubble text layer."""

    text: str
    x: float
    baseline_y: float
    width: float


class BubbleLayout(BaseModel):
    """Per-bubble typeset result (layout JSON; PNG is debug-only).

    Export notes (PSD live text / IDML text frame):
    - ``lines[]`` are hard breaks with explicit leading; export as hard breaks.
    - When ``hyphenated`` is True, hyphens are already embedded in ``lines[].text`` —
      export must disable hyphenation so engines do not re-hyphenate.
    - Coordinates are pixel, top-left origin, y down (PSD-native; IDML converts to pt).
    """

    bubble_id: int
    status: str = "ok"  # ok | editor
    reason: Optional[str] = None  # clean_editor | multi_lobe | no_fit | no_text | empty_mask
    font_family: Optional[str] = None
    font_px: Optional[float] = None
    font_pt: Optional[float] = None  # font_px * 72 / dpi
    font_ratio: Optional[float] = None
    leading_px: Optional[float] = None
    leading_pt: Optional[float] = None  # leading_px * 72 / dpi
    font_postscript_name: Optional[str] = None
    font_file: Optional[str] = None
    align: str = "center"
    lines: List[TypesetLine] = Field(default_factory=list)
    hyphenated: bool = False
    frame: Optional[BBox] = None  # text block rect from line boxes (IDML frame)


class BubbleText(BaseModel):
    """Human-supplied text for one bubble (fixture until translation layer exists)."""

    bubble_id: int
    text: str = ""
