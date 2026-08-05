"""Page document + layer model (clean/text/paint; PSD/IDML export later)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from vision.types import BBox, Bubble, BubbleLayout, BubbleText, Point


class SourceRef(BaseModel):
    """Immutable page source reference."""

    file: str = ""
    width: int
    height: int
    dpi: float = 350.0
    color_mode: str = "rgb"  # rgb | cmyk (cmyk conversion deferred to export)


class MaskSpec(BaseModel):
    """Parametric or raster mask. Default: inset_polygon (reproducible via erode)."""

    type: str = "inset_polygon"  # inset_polygon | rle | png_alpha
    polygon: List[Point] = Field(default_factory=list)
    inset: int = 10
    size: Optional[Tuple[int, int]] = None  # (h, w) for rle
    bbox: Optional[BBox] = None
    counts: Optional[str] = None  # rle
    file: Optional[str] = None  # png_alpha


class Stroke(BaseModel):
    """Parametric brush stroke (no raster payload)."""

    id: str
    mode: str  # fill | restore
    color: Optional[Tuple[int, int, int]] = None  # required for fill
    radius_px: float
    points: List[Point] = Field(default_factory=list)
    bubble_id: Optional[int] = None


class Layer(BaseModel):
    """One composable page layer (clean fill, text layout, or paint strokes)."""

    id: str  # "cln_3" / "txt_3" / "pnt_3" — stable
    kind: str  # clean | text | paint  (later: sfx | note)
    name: str  # "Clean / bubble 3" — PS layer name
    group: Optional[str] = None  # "bubble_3" — PS group / IDML group
    bubble_id: Optional[int] = None
    enabled: bool = True
    opacity: float = 1.0
    blend: str = "normal"
    z: int = 0  # clean 100+, paint 150+, text 200+
    source_step: str = ""  # "clean.inset_fill.v1" | "typeset.v1" | "paint.v1"
    bbox: BBox
    # kind == clean
    mask: Optional[MaskSpec] = None
    fill_color: Optional[Tuple[int, int, int]] = None
    raster_file: Optional[str] = None
    # kind == text
    layout: Optional[BubbleLayout] = None
    frame: Optional[BBox] = None  # IDML text frame rectangle
    # kind == paint
    strokes: List[Stroke] = Field(default_factory=list)


class DocFlags(BaseModel):
    """Pipeline status summary for the page."""

    clean: str = "ok"  # ok | partial
    typeset: str = "ok"  # ok | partial | skipped
    editor_bubble_ids: List[int] = Field(default_factory=list)
    typeset_editor_bubble_ids: List[int] = Field(default_factory=list)


class PageDoc(BaseModel):
    """Composable page: immutable source + bubbles + enable/disable layers."""

    schema_version: int = 1
    page_id: str
    source: SourceRef
    bubbles: List[Bubble] = Field(default_factory=list)
    texts: List[BubbleText] = Field(default_factory=list)
    layers: List[Layer] = Field(default_factory=list)
    flags: DocFlags = Field(default_factory=DocFlags)
