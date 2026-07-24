from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from vision.types import Bubble


class BubblesResponse(BaseModel):
    page_id: str
    width: int
    height: int
    bubbles: List[Bubble]


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
    cleaned_image: str = Field(..., description="Base64 PNG after inset-fill clean")
    bubbles_overlay: str = Field(
        ...,
        description="Base64 PNG: original with YOLO polygons + 1-based labels",
    )
    editor_regions: List[Tuple[int, int, int, int]] = Field(default_factory=list)
    clean: str = Field("ok", description="'ok' or 'partial'")


class HealthResponse(BaseModel):
    status: str
    device: str
    models: Dict[str, bool]
