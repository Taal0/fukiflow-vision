"""VISION_LIGHT guards for YOLO-dependent routes."""

from __future__ import annotations

from fastapi import HTTPException

from .settings import get_settings

LIGHT_UNAVAILABLE = (
    "Unavailable in VISION_LIGHT mode (YOLO models not loaded). "
    "Unset VISION_LIGHT to enable /v1/bubbles, /v1/order, and /v1/pipeline."
)


def require_full_vision() -> None:
    if get_settings().vision_light:
        raise HTTPException(status_code=503, detail=LIGHT_UNAVAILABLE)
