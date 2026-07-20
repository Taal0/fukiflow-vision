from __future__ import annotations

from fastapi import APIRouter

from ..engine import get_engine
from ..schemas import HealthResponse
from ..settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    try:
        eng = get_engine()
        loaded = eng.ready
    except RuntimeError:
        loaded = False
    return HealthResponse(status="ok", device=settings.device, models_loaded=loaded)
