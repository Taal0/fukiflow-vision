from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

EditorRegion = Tuple[int, int, int, int]


def _box_points(box) -> np.ndarray:
    """Collect polygon vertices from a Box; fall back to bbox corners."""
    poly = getattr(box, "polygon", None)
    pts: List[List[float]] = []
    if poly:
        first = poly[0]
        # List[Quad] → first element is a list of points; Quad → first is [x, y].
        if first and isinstance(first[0], (list, tuple)):
            for quad in poly:
                for p in quad:
                    pts.append([float(p[0]), float(p[1])])
        else:
            for p in poly:
                pts.append([float(p[0]), float(p[1])])

    if len(pts) < 3:
        x1, y1, x2, y2 = (int(v) for v in box.bbox)
        return np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        )
    return np.asarray(pts, dtype=np.float32)


def _hull_mask(shape: Tuple[int, int], box) -> np.ndarray:
    h, w = shape
    points = _box_points(box)
    hull = cv2.convexHull(points)
    mask = np.zeros((h, w), dtype=np.uint8)
    if hull is None or len(hull) < 3:
        x1, y1, x2, y2 = (int(v) for v in box.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
    return mask


def solid_fill_clean(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    boxes: Optional[Sequence] = None,
    *,
    ring: int = 6,
    uniform_tol: float = 18.0,
    uniform_frac: float = 0.85,
    fill_dilate: int = 2,
    min_ring_px: int = 30,
) -> Tuple[np.ndarray, List[EditorRegion]]:
    """Fill uniform-background detection regions; skip textured ones for the editor.

    Decision unit is a detect-ocr box (balloon-scale), not a mask connected
    component. For each box, sample a ring around its convex hull (excluding
    other regions). If enough ring pixels are within ``uniform_tol`` of the
    median color, fill the (slightly dilated) hull with that median. Otherwise
    leave pixels untouched and report one editor bbox per region.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {img_rgb.shape}")
    if mask.shape[:2] != img_rgb.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape[:2]} does not match image {img_rgb.shape[:2]}"
        )

    out = img_rgb.copy()
    editor_regions: List[EditorRegion] = []
    if not boxes:
        return out, editor_regions

    h, w = img_rgb.shape[:2]
    binary = (mask > 0).astype(np.uint8)

    hull_masks = [_hull_mask((h, w), box) for box in boxes]
    all_hulls = np.zeros((h, w), dtype=np.uint8)
    for hm in hull_masks:
        all_hulls = np.maximum(all_hulls, hm)

    ring_px = max(1, int(ring))
    ring_k = 2 * ring_px + 1
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_k, ring_k))

    fill_px = max(0, int(fill_dilate))
    fill_kernel = None
    if fill_px > 0:
        fill_k = 2 * fill_px + 1
        fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_k, fill_k))

    tol = float(uniform_tol)
    frac_need = float(uniform_frac)

    for box, hm in zip(boxes, hull_masks):
        bbox_t = tuple(int(v) for v in box.bbox)
        if len(bbox_t) != 4:
            continue
        bbox: EditorRegion = (bbox_t[0], bbox_t[1], bbox_t[2], bbox_t[3])

        if not hm.any():
            editor_regions.append(bbox)
            continue

        other = (all_hulls > 0) & (hm == 0)
        other_mask = (binary > 0) & other

        dilated = cv2.dilate(hm, ring_kernel)
        ring_mask = (dilated > 0) & (hm == 0) & ~other & ~other_mask
        ring_count = int(ring_mask.sum())
        if ring_count < int(min_ring_px):
            editor_regions.append(bbox)
            continue

        samples = out[ring_mask].astype(np.float32)
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
        if fill_kernel is not None:
            fill = cv2.dilate(hm, fill_kernel) > 0
        else:
            fill = hm > 0
        out[fill] = color

    return out, editor_regions
