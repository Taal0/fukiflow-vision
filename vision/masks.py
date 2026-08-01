"""Shared polygon / ring / ink helpers for bubble detect + clean."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vision.types import EditorRegion

BBoxLike = Sequence[int]


def as_bbox(bbox: BBoxLike) -> EditorRegion:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    return (x1, y1, x2, y2)


def polygon_mask(
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


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Morphological erode by ``px``; ``px<=0`` returns mask unchanged."""
    if px <= 0:
        return mask
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, kernel)


def ring_band(fill_mask: np.ndarray, ring_px: int) -> np.ndarray:
    """Inner-edge band of ``fill_mask`` with width ``ring_px``."""
    ring_px = max(1, int(ring_px))
    eroded_inner = erode_mask(fill_mask, ring_px)
    return (fill_mask > 0) & (eroded_inner == 0)


def ring_median(
    img_rgb: np.ndarray,
    band: np.ndarray,
) -> Optional[np.ndarray]:
    """Per-channel median of RGB pixels in ``band``; None if empty."""
    if not band.any():
        return None
    samples = img_rgb[band].astype(np.float32)
    return np.median(samples, axis=0)


def ring_is_uniform(
    img_rgb: np.ndarray,
    band: np.ndarray,
    med: np.ndarray,
    *,
    tol: float = 18.0,
    frac: float = 0.85,
) -> bool:
    """True if enough ring pixels sit within ``tol`` of ``med`` on all channels."""
    if not band.any():
        return False
    samples = img_rgb[band].astype(np.float32)
    within = (np.abs(samples - med) <= float(tol)).all(axis=1)
    return float(within.mean()) >= float(frac)


@dataclass(frozen=True)
class InkStructure:
    """Ink presence metrics inside an inset bubble mask."""

    dark_px: int
    dark_frac: float
    cc_count: int
    max_cc_area: int
    interior_px: int
    ring_ok: bool
    ring_uniform: bool
    text_present: bool


def _empty_structure(*, ring_ok: bool = False, ring_uniform: bool = False) -> InkStructure:
    return InkStructure(
        dark_px=0,
        dark_frac=0.0,
        cc_count=0,
        max_cc_area=0,
        interior_px=0,
        ring_ok=ring_ok,
        ring_uniform=ring_uniform,
        text_present=False,
    )


def dark_mask(
    img_rgb: np.ndarray,
    fill_mask: np.ndarray,
    med: np.ndarray,
    *,
    tol: float = 40.0,
) -> np.ndarray:
    """Pixels in ``fill_mask`` that deviate from ring-median on any channel."""
    out = np.zeros(fill_mask.shape, dtype=np.uint8)
    interior = fill_mask > 0
    if not interior.any():
        return out
    pixels = img_rgb[interior].astype(np.float32)
    deviate = (np.abs(pixels - med) > float(tol)).any(axis=1)
    out[interior] = deviate.astype(np.uint8)
    return out


def connected_component_stats(mask: np.ndarray) -> Tuple[int, int]:
    """Return (cc_count, max_cc_area) for non-zero 8-connected components."""
    areas = component_areas(mask)
    if not areas:
        return 0, 0
    return int(len(areas)), int(areas[0])


def component_areas(mask: np.ndarray) -> List[int]:
    """Areas of non-background 8-connected components, descending."""
    if not mask.any():
        return []
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return []
    areas = [int(a) for a in stats[1:, cv2.CC_STAT_AREA]]
    areas.sort(reverse=True)
    return areas


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Zero out all but the largest 8-connected component."""
    if not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        # 0 = background, 1 = sole foreground component
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_label = int(1 + int(np.argmax(areas)))
    out = np.zeros_like(mask)
    out[labels == keep_label] = 1
    return out


def ink_structure(
    img_rgb: np.ndarray,
    polygon: Sequence[Sequence[float]],
    bbox: BBoxLike,
    *,
    inset: int = 10,
    ring: int = 6,
    tol: float = 40.0,
    min_ring_px: int = 30,
    min_px: int = 50,
    min_cc_area: int = 12,
    uniform_tol: float = 18.0,
    uniform_frac: float = 0.85,
) -> InkStructure:
    """Measure ink *structure* (not area ratio) inside an inset bubble.

    ``dark`` = inset pixels that deviate from the ring-median background.
    ``text_present`` if any CC area ≥ ``min_cc_area`` OR total dark ≥ ``min_px``.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {img_rgb.shape}")

    h, w = img_rgb.shape[:2]
    box = as_bbox(bbox)
    poly = polygon_mask((h, w), polygon, box)
    if not poly.any():
        return _empty_structure()

    fill = erode_mask(poly, max(0, int(inset)))
    if not fill.any():
        return _empty_structure()

    band = ring_band(fill, ring)
    ring_count = int(band.sum())
    interior_px = int((fill > 0).sum())
    if ring_count < int(min_ring_px):
        # Can't sample background — keep for editor (not DROP).
        return InkStructure(
            dark_px=0,
            dark_frac=0.0,
            cc_count=0,
            max_cc_area=0,
            interior_px=interior_px,
            ring_ok=False,
            ring_uniform=False,
            text_present=True,
        )

    med = ring_median(img_rgb, band)
    if med is None:
        return InkStructure(
            dark_px=0,
            dark_frac=0.0,
            cc_count=0,
            max_cc_area=0,
            interior_px=interior_px,
            ring_ok=False,
            ring_uniform=False,
            text_present=True,
        )

    uniform = ring_is_uniform(
        img_rgb, band, med, tol=uniform_tol, frac=uniform_frac
    )
    dmask = dark_mask(img_rgb, fill, med, tol=tol)
    dark_px = int(dmask.sum())
    dark_frac = float(dark_px / interior_px) if interior_px else 0.0
    cc_count, max_cc_area = connected_component_stats(dmask)
    text_present = (max_cc_area >= int(min_cc_area)) or (dark_px >= int(min_px))

    return InkStructure(
        dark_px=dark_px,
        dark_frac=dark_frac,
        cc_count=cc_count,
        max_cc_area=max_cc_area,
        interior_px=interior_px,
        ring_ok=True,
        ring_uniform=uniform,
        text_present=text_present,
    )


def ink_ratio(
    img_rgb: np.ndarray,
    polygon: Sequence[Sequence[float]],
    bbox: BBoxLike,
    *,
    inset: int = 10,
    ring: int = 6,
    tol: float = 40.0,
    min_ring_px: int = 30,
) -> float:
    """Fraction of inset-mask pixels that deviate from ring-median background.

    Kept for diagnostics; filter decisions use :func:`ink_structure`.
    """
    s = ink_structure(
        img_rgb,
        polygon,
        bbox,
        inset=inset,
        ring=ring,
        tol=tol,
        min_ring_px=min_ring_px,
        min_px=0,
        min_cc_area=0,
    )
    return float(s.dark_frac)
