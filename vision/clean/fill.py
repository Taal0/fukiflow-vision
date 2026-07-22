"""Inset-polygon solid fill for speech bubbles (no text mask, no hull)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

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
        """No-op: no model weights; keeps the bubble/ocr load() interface."""
        self.ready = True

    def clean(
        self,
        img_rgb: np.ndarray,
        bubbles: Sequence[Bubble],
    ) -> Tuple[np.ndarray, List[EditorRegion], str]:
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB, got shape {img_rgb.shape}")

        out = img_rgb.copy()
        editor_regions: List[EditorRegion] = []
        if not bubbles:
            return out, editor_regions, "ok"

        h, w = img_rgb.shape[:2]
        inset_px = max(0, int(self.inset))
        ring_px = max(1, int(self.ring))
        tol = float(self.uniform_tol)
        frac_need = float(self.uniform_frac)
        min_ring = int(self.min_ring_px)

        inset_kernel = None
        if inset_px > 0:
            ik = 2 * inset_px + 1
            inset_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ik, ik))

        rk = 2 * ring_px + 1
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rk, rk))

        for b in bubbles:
            bbox = _as_editor_region(b.bbox)
            poly_mask = _polygon_mask((h, w), b.polygon, bbox)
            if not poly_mask.any():
                editor_regions.append(bbox)
                continue

            if inset_kernel is not None:
                fill_mask = cv2.erode(poly_mask, inset_kernel)
            else:
                fill_mask = poly_mask

            if not fill_mask.any():
                editor_regions.append(bbox)
                continue

            eroded_inner = cv2.erode(fill_mask, ring_kernel)
            ring_band = (fill_mask > 0) & (eroded_inner == 0)
            ring_count = int(ring_band.sum())
            if ring_count < min_ring:
                editor_regions.append(bbox)
                continue

            samples = out[ring_band].astype(np.float32)
            med = np.median(samples, axis=0)
            within = (np.abs(samples - med) <= tol).all(axis=1)
            frac = float(within.mean())
            if frac < frac_need:
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


def _as_editor_region(bbox: Sequence[int]) -> EditorRegion:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    return (x1, y1, x2, y2)


def _polygon_mask(
    shape: Tuple[int, int],
    polygon: Sequence[Sequence[float]],
    bbox: EditorRegion,
) -> np.ndarray:
    """Rasterize bubble polygon; fall back to bbox if polygon is too short."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts: List[List[float]] = []
    for p in polygon or []:
        if p is None or len(p) < 2:
            continue
        pts.append([float(p[0]), float(p[1])])

    if len(pts) >= 3:
        arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [arr], 1)
        return mask

    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask
