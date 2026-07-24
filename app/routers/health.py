from __future__ import annotations

from fastapi import APIRouter

from vision.bubble.detect import get_detector
from vision.clean.fill import get_cleaner

from ..schemas import HealthResponse
from ..settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    bubble_ok = False
    clean_ok = False
    try:
        bubble_ok = get_detector().ready
    except RuntimeError:
        pass
    try:
        clean_ok = get_cleaner().ready
    except RuntimeError:
        pass
    return HealthResponse(
        status="ok",
        device=settings.device,
        models={"bubble": bubble_ok, "clean": clean_ok},
    )
