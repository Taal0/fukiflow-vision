"""Inset-polygon solid fill for speech bubbles (no text mask, no hull)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from vision.masks import (
    as_bbox,
    erode_mask,
    polygon_mask,
    ring_band,
    ring_is_uniform,
    ring_median,
)
from vision.types import Bubble, EditorRegion

_cleaner: Optional["CleanService"] = None


def set_cleaner(service: Optional["CleanService"]) -> None:
    global _cleaner
    _cleaner = service


def get_cleaner() -> "CleanService":
    if _cleaner is None:
        raise RuntimeError("CleanService not initialized")
    return _cleaner


class CleanService:
    """Fill uniform bubble interiors from YOLO-seg polygons; skip textured ones."""

    def __init__(
        self,
        inset: int = 10,
        ring: int = 6,
        uniform_tol: float = 18.0,
        uniform_frac: float = 0.85,
        min_ring_px: int = 30,
    ) -> None:
        self.inset = inset
        self.ring = ring
        self.uniform_tol = uniform_tol
        self.uniform_frac = uniform_frac
        self.min_ring_px = min_ring_px
        self.ready = False

    def load(self) -> None:
        """No-op: no model weights; keeps the bubble load() interface."""
        self.ready = True

    def clean(
        self,
        img_rgb: np.ndarray,
        bubbles: Sequence[Bubble],
        force_editor: Optional[Set[int]] = None,
    ) -> Tuple[np.ndarray, List[EditorRegion], str]:
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB, got shape {img_rgb.shape}")

        out = img_rgb.copy()
        editor_regions: List[EditorRegion] = []
        if not bubbles:
            return out, editor_regions, "ok"

        force = force_editor or set()
        h, w = img_rgb.shape[:2]
        inset_px = max(0, int(self.inset))
        ring_px = max(1, int(self.ring))
        tol = float(self.uniform_tol)
        frac_need = float(self.uniform_frac)
        min_ring = int(self.min_ring_px)

        for b in bubbles:
            bbox = as_bbox(b.bbox)
            if int(b.id) in force:
                editor_regions.append(bbox)
                continue

            poly_mask = polygon_mask((h, w), b.polygon, bbox)
            if not poly_mask.any():
                editor_regions.append(bbox)
                continue

            fill_mask = erode_mask(poly_mask, inset_px)
            if not fill_mask.any():
                editor_regions.append(bbox)
                continue

            band = ring_band(fill_mask, ring_px)
            ring_count = int(band.sum())
            if ring_count < min_ring:
                editor_regions.append(bbox)
                continue

            med = ring_median(out, band)
            if med is None:
                editor_regions.append(bbox)
                continue

            if not ring_is_uniform(out, band, med, tol=tol, frac=frac_need):
                editor_regions.append(bbox)
                continue

            color = (
                int(np.clip(round(float(med[0])), 0, 255)),
                int(np.clip(round(float(med[1])), 0, 255)),
                int(np.clip(round(float(med[2])), 0, 255)),
            )
            out[fill_mask > 0] = color

        status = "partial" if editor_regions else "ok"
        return out, editor_regions, status
