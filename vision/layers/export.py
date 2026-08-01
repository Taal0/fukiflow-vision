"""Normalized export items (PSD/IDML writers bind here, not to PageDoc)."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from vision.document import PageDoc
from vision.types import BBox


class TextRun(BaseModel):
    """One hard-broken line ready for live-text export."""

    text: str
    x: float
    baseline_y: float
    width: float


class ExportItem(BaseModel):
    """Normalized export unit for a future PSD/IDML writer."""

    id: str
    kind: Literal["raster", "text"]
    name: str
    group: Optional[str] = None
    bubble_id: Optional[int] = None
    enabled: bool = True
    z: int = 0
    # raster
    bbox: Optional[BBox] = None
    rgba_file: Optional[str] = None  # filled by export writer / EXPORT_RASTER
    fill_color: Optional[Tuple[int, int, int]] = None
    # text (live)
    frame: Optional[BBox] = None
    runs: List[TextRun] = Field(default_factory=list)
    font_postscript_name: Optional[str] = None
    font_file: Optional[str] = None
    font_pt: Optional[float] = None
    leading_pt: Optional[float] = None
    align: str = "center"
    hyphenated: bool = False  # if True, export must disable hyphenation


def to_export_items(doc: PageDoc) -> List[ExportItem]:
    """Flatten enabled-aware layers into export items (all layers; respect enabled)."""
    items: List[ExportItem] = []
    for layer in sorted(doc.layers, key=lambda L: (L.z, L.id)):
        if layer.kind == "clean":
            items.append(
                ExportItem(
                    id=layer.id,
                    kind="raster",
                    name=layer.name,
                    group=layer.group,
                    bubble_id=layer.bubble_id,
                    enabled=layer.enabled,
                    z=layer.z,
                    bbox=layer.bbox,
                    rgba_file=layer.raster_file,
                    fill_color=layer.fill_color,
                )
            )
        elif layer.kind == "text":
            lay = layer.layout
            runs: List[TextRun] = []
            if lay and lay.lines:
                runs = [
                    TextRun(
                        text=line.text,
                        x=line.x,
                        baseline_y=line.baseline_y,
                        width=line.width,
                    )
                    for line in lay.lines
                ]
            items.append(
                ExportItem(
                    id=layer.id,
                    kind="text",
                    name=layer.name,
                    group=layer.group,
                    bubble_id=layer.bubble_id,
                    enabled=layer.enabled,
                    z=layer.z,
                    bbox=layer.bbox,
                    frame=layer.frame or (lay.frame if lay else None),
                    runs=runs,
                    font_postscript_name=lay.font_postscript_name if lay else None,
                    font_file=lay.font_file if lay else None,
                    font_pt=lay.font_pt if lay else None,
                    leading_pt=lay.leading_pt if lay else None,
                    align=(lay.align if lay else "center"),
                    hyphenated=bool(lay.hyphenated) if lay else False,
                )
            )
    return items
