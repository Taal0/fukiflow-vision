"""Model holders for FastAPI lifespan (bubble + panel + order + clean)."""

from __future__ import annotations

from typing import Optional

from vision.bubble.detect import BubbleDetector, get_detector, set_detector
from vision.clean.fill import CleanService, get_cleaner, set_cleaner
from vision.order.service import (
    OrderService,
    OrderSettings,
    get_service as get_order_service,
    set_service as set_order_service,
)
from vision.panel.detect import (
    PanelDetector,
    get_detector as get_panel_detector,
    set_detector as set_panel_detector,
)

from .settings import Settings, get_settings


def create_bubble_detector(settings: Optional[Settings] = None) -> BubbleDetector:
    s = settings or get_settings()
    return BubbleDetector(
        weight=s.bubble_weight,
        conf=s.bubble_conf,
        imgsz=s.bubble_imgsz,
        device=s.device,
    )


def create_panel_detector(settings: Optional[Settings] = None) -> PanelDetector:
    s = settings or get_settings()
    return PanelDetector(
        weight=s.panel_weight,
        conf=s.panel_conf,
        imgsz=s.panel_imgsz,
        device=s.device,
    )


def create_order_service(
    settings: Optional[Settings] = None,
    panel_detector: Optional[PanelDetector] = None,
) -> OrderService:
    s = settings or get_settings()
    return OrderService(
        settings=OrderSettings(
            tol_frac=s.order_tol_frac,
            nms_iou=s.panel_nms_iou,
            iob_contained=s.order_iob_contained,
            iob_partial=s.order_iob_partial,
            ambig_delta=s.order_ambig_delta,
            conf_min=s.order_conf_min,
        ),
        panel_detector=panel_detector,
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
    "PanelDetector",
    "OrderService",
    "CleanService",
    "create_bubble_detector",
    "create_panel_detector",
    "create_order_service",
    "create_clean_service",
    "get_detector",
    "set_detector",
    "get_panel_detector",
    "set_panel_detector",
    "get_order_service",
    "set_order_service",
    "get_cleaner",
    "set_cleaner",
]
