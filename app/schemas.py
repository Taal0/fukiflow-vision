from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from vision.types import Bubble


class BubblesResponse(BaseModel):
    page_id: str
    width: int
    height: int
    bubbles: List[Bubble]


class OcrResponse(BaseModel):
    page_id: Optional[str] = None
    width: int
    height: int
    bubbles: List[Bubble]


class CleanResponse(BaseModel):
    page_id: Optional[str] = None
    cleaned_image: str = Field(..., description="Base64-encoded PNG (original when clean disabled)")
    editor_regions: List[Tuple[int, int, int, int]] = Field(default_factory=list)
    clean: str = Field("disabled", description="'disabled' while clean is a no-op stub")


class PipelineResponse(BaseModel):
    page_id: str
    width: int
    height: int
    bubbles: List[Bubble]
    cleaned_image: str = Field(..., description="Base64 PNG; equals original while clean disabled")
    editor_regions: List[Tuple[int, int, int, int]] = Field(default_factory=list)
    clean: str = "disabled"


class HealthResponse(BaseModel):
    status: str
    device: str
    models: Dict[str, bool]
