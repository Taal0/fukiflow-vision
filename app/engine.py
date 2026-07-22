"""Model holders for FastAPI lifespan (bubble + OCR). CTD VisionEngine removed."""

from __future__ import annotations

from typing import Optional

from vision.bubble.detect import BubbleDetector, get_detector, set_detector
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


__all__ = [
    "BubbleDetector",
    "MangaOcrService",
    "create_bubble_detector",
    "create_ocr_service",
    "get_detector",
    "set_detector",
    "get_ocr",
    "set_ocr",
]
