"""Model holders for FastAPI lifespan (bubble + OCR + clean)."""

from __future__ import annotations

from typing import Optional

from vision.bubble.detect import BubbleDetector, get_detector, set_detector
from vision.clean.fill import CleanService, get_cleaner, set_cleaner
from vision.ocr.manga import MangaOcrService, get_ocr, set_ocr

from .settings import Settings, get_settings


def create_bubble_detector(settings: Optional[Settings] = None) -> BubbleDetector:
    s = settings or get_settings()
    return BubbleDetector(
        weight=s.bubble_weight,
        conf=s.bubble_conf,
        imgsz=s.bubble_imgsz,
        device=s.device,
    )


def create_ocr_service(settings: Optional[Settings] = None) -> MangaOcrService:
    s = settings or get_settings()
    return MangaOcrService(
        model=s.ocr_model,
        device=s.device,
        crop_pad=s.ocr_crop_pad,
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
    "MangaOcrService",
    "CleanService",
    "create_bubble_detector",
    "create_ocr_service",
    "create_clean_service",
    "get_detector",
    "set_detector",
    "get_ocr",
    "set_ocr",
    "get_cleaner",
    "set_cleaner",
]
