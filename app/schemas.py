from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from vision.types import Bubble, Panel


class BubblesResponse(BaseModel):
    page_id: str
    width: int
    height: int
    bubbles: List[Bubble]
    panels: List[Panel] = Field(default_factory=list)
    order_confidence: float = 1.0
    needs_review: bool = False


class OrderResponse(BaseModel):
    page_id: str
    width: int
    height: int
    bubbles: List[Bubble]
    panels: List[Panel] = Field(default_factory=list)
    order_confidence: float = 1.0
    needs_review: bool = False
    bubbles_overlay: Optional[str] = Field(
        None,
        description="Base64 PNG: panels + bubble reading_order overlay",
    )


class CleanResponse(BaseModel):
    page_id: Optional[str] = None
    cleaned_image: str = Field(..., description="Base64-encoded PNG after inset-fill clean")
    editor_regions: List[Tuple[int, int, int, int]] = Field(
        default_factory=list,
        description="BBoxes skipped (textured / too small / empty after inset)",
    )
    clean: str = Field(
        "ok",
        description="'ok' if all bubbles filled; 'partial' if any skipped",
    )


class PipelineResponse(BaseModel):
    page_id: str
    width: int
    height: int
    bubbles: List[Bubble]
    panels: List[Panel] = Field(default_factory=list)
    order_confidence: float = 1.0
    needs_review: bool = False
    cleaned_image: str = Field(..., description="Base64 PNG after inset-fill clean")
    bubbles_overlay: str = Field(
        ...,
        description="Base64 PNG: original with panels + bubble reading_order labels",
    )
    editor_regions: List[Tuple[int, int, int, int]] = Field(default_factory=list)
    clean: str = Field("ok", description="'ok' or 'partial'")


class HealthResponse(BaseModel):
    status: str
    device: str
    models: Dict[str, bool]
