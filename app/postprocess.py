from __future__ import annotations

import cv2
import numpy as np


def remove_inpaint_stains(
    cleaned_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    mask_dilate: int = 2,
    gray_max: int = 220,
    min_area: int = 3,
    max_area: int = 500,
    neighbor_median_min: int = 230,
    neighbor_ring: int = 4,
    paint_dilate: int = 8,
) -> np.ndarray:
    """Rule-based cleanup of small dark/gray stains left after LaMa inpaint.

    Scans a narrow ROI around the inpaint mask (small dilate). Accepted stain
    components are then expanded (paint_dilate) before painting white, so soft
    ghosts get covered without scanning near bubble/art edges.
    """
    if cleaned_rgb.ndim != 3 or cleaned_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {cleaned_rgb.shape}")
    if mask.shape[:2] != cleaned_rgb.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape[:2]} does not match image {cleaned_rgb.shape[:2]}"
        )

    out = cleaned_rgb.copy()
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return out

    dilate_px = max(0, int(mask_dilate))
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        roi = cv2.dilate(binary, kernel)
    else:
        roi = binary

    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    candidates = ((gray <= gray_max) & (roi > 0)).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates, connectivity=8
    )
    if n_labels <= 1:
        return out

    ring = max(1, int(neighbor_ring))
    ring_k = 2 * ring + 1
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_k, ring_k))

    paint = np.zeros(gray.shape, dtype=bool)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        comp = labels == label
        dilated = cv2.dilate(comp.astype(np.uint8), ring_kernel) > 0
        ring_mask = dilated & ~comp
        outside = ring_mask & (roi == 0)

        if not outside.any():
            paint |= comp
            continue

        if int(np.median(gray[outside])) >= neighbor_median_min:
            paint |= comp

    if not paint.any():
        return out

    expand = max(0, int(paint_dilate))
    if expand > 0:
        pk = 2 * expand + 1
        paint_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pk, pk))
        paint = cv2.dilate(paint.astype(np.uint8), paint_kernel) > 0
        # Never paint outside the scan ROI — keeps bubble edges safe.
        paint &= roi > 0

    out[paint] = (255, 255, 255)
    return out
