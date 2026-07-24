"""Model holders for FastAPI lifespan (bubble + clean)."""

from __future__ import annotations

from typing import Optional

from vision.bubble.detect import BubbleDetector, get_detector, set_detector
from vision.clean.fill import CleanService, get_cleaner, set_cleaner

from .settings import Settings, get_settings


def create_bubble_detector(settings: Optional[Settings] = None) -> BubbleDetector:
    s = settings or get_settings()
    return BubbleDetector(
        weight=s.bubble_weight,
        conf=s.bubble_conf,
        imgsz=s.bubble_imgsz,
        device=s.device,
    )


def create_clean_service(settings: Optional[Settings] = None) -> CleanService:
    s = settings or get_settings()
    return CleanService(
        inset=s.clean_inset,
        ring=s.clean_ring,
        uniform_tol=s.clean_uniform_tol,
        uniform_frac=s.clean_uniform_frac,
        min_ring_px=s.clean_min_ring_px,
    )


__all__ = [
    "BubbleDetector",
    "CleanService",
    "create_bubble_detector",
    "create_clean_service",
    "get_detector",
    "set_detector",
    "get_cleaner",
    "set_cleaner",
]
