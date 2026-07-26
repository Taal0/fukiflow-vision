from __future__ import annotations

from fastapi import APIRouter

from vision.bubble.detect import get_detector
from vision.clean.fill import get_cleaner
from vision.order.service import get_service as get_order_service
from vision.panel.detect import get_detector as get_panel_detector

from ..schemas import HealthResponse
from ..settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    bubble_ok = False
    panel_ok = False
    order_ok = False
    clean_ok = False
    try:
        bubble_ok = get_detector().ready
    except RuntimeError:
        pass
    try:
        panel_ok = get_panel_detector().ready
    except RuntimeError:
        pass
    try:
        order_ok = get_order_service().ready
    except RuntimeError:
        pass
    try:
        clean_ok = get_cleaner().ready
    except RuntimeError:
        pass
    return HealthResponse(
        status="ok",
        device=settings.device,
        models={
            "bubble": bubble_ok,
            "panel": panel_ok,
            "order": order_ok,
            "clean": clean_ok,
        },
    )
