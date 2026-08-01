"""Raster width profile from bubble polygon (erode + per-row runs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from vision.masks import (
    as_bbox,
    component_areas,
    erode_mask,
    keep_largest_component,
    polygon_mask,
)
from vision.types import Bubble

Run = Tuple[int, int]  # [x_start, x_end) inclusive-exclusive

# Secondary component must be at least this fraction of the primary
# to count as a real second lobe (speech-tail residue is typically << this).
DEFAULT_MULTI_LOBE_MIN_AREA_RATIO = 0.20


@dataclass
class WidthProfile:
    """Per-row contiguous runs inside the typographic (eroded) mask."""

    mask: np.ndarray
    cx: float
    cy: float
    y0: int
    y1: int
    row_runs: List[List[Run]] = field(default_factory=list)
    multi_lobe: bool = False
    neck_depth: float = 0.0  # 1 - min_w/max_w over single-run body rows
    empty: bool = False
    # Diagnostics for multi-lobe threshold tuning / ignored tail fragments.
    cc_count: int = 0
    primary_area: int = 0
    secondary_area: int = 0
    secondary_area_ratio: float = 0.0
    ignored_fragments: int = 0


def _row_runs(row: np.ndarray) -> List[Run]:
    """Contiguous True runs via padded diff (vectorized per row)."""
    if row.size == 0 or not row.any():
        return []
    padded = np.concatenate(([0], row.view(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _neck_depth(row_runs: Sequence[Sequence[Run]], *, min_width: int = 4) -> float:
    """Sharpest single-run narrowing: 1 - min/max width (0 if undefined)."""
    widths: List[int] = []
    for runs in row_runs:
        if len(runs) != 1:
            continue
        w = runs[0][1] - runs[0][0]
        if w >= min_width:
            widths.append(w)
    if len(widths) < 2:
        return 0.0
    lo, hi = min(widths), max(widths)
    if hi <= 0:
        return 0.0
    return float(1.0 - (lo / hi))


def _lobe_decision(
    areas: Sequence[int],
    *,
    min_area_ratio: float,
) -> Tuple[bool, int, float, int]:
    """Return (multi_lobe, secondary_area, secondary_ratio, ignored_fragments)."""
    if not areas:
        return False, 0, 0.0, 0
    primary = int(areas[0])
    if primary <= 0:
        return False, 0, 0.0, 0
    threshold = primary * float(min_area_ratio)
    significant = [int(a) for a in areas if a >= threshold]
    ignored = len(areas) - len(significant)
    secondary = int(areas[1]) if len(areas) > 1 else 0
    ratio = float(secondary) / float(primary) if primary else 0.0
    return len(significant) > 1, secondary, ratio, ignored


def build_width_profile(
    shape: Tuple[int, int],
    bubble: Bubble,
    *,
    clean_inset: int,
    margin_px: int,
    multi_lobe_min_area_ratio: float = DEFAULT_MULTI_LOBE_MIN_AREA_RATIO,
) -> WidthProfile:
    """Rasterize polygon, erode by clean_inset + typographic margin, extract runs.

    Multi-lobe uses connected components on the eroded mask: a second component
    only escalates if its area is ≥ ``multi_lobe_min_area_ratio`` × primary.
    Smaller fragments (speech-tail residue) are ignored and the profile keeps
    only the primary component.
    """
    h, w = shape
    bbox = as_bbox(bubble.bbox)
    poly = polygon_mask((h, w), bubble.polygon, bbox)
    if not poly.any():
        return WidthProfile(
            mask=poly,
            cx=0.0,
            cy=0.0,
            y0=0,
            y1=0,
            empty=True,
        )

    erode_px = max(0, int(clean_inset)) + max(0, int(margin_px))
    mask = erode_mask(poly, erode_px)
    if not mask.any():
        return WidthProfile(
            mask=mask,
            cx=0.0,
            cy=0.0,
            y0=0,
            y1=0,
            empty=True,
        )

    areas = component_areas(mask)
    multi, secondary, sec_ratio, ignored = _lobe_decision(
        areas, min_area_ratio=multi_lobe_min_area_ratio
    )
    primary = int(areas[0]) if areas else 0
    cc_count = len(areas)

    # Drop sub-threshold fragments so row-run bands aren't poisoned by tails.
    if not multi and cc_count > 1:
        mask = keep_largest_component(mask)
        if not mask.any():
            return WidthProfile(
                mask=mask,
                cx=0.0,
                cy=0.0,
                y0=0,
                y1=0,
                empty=True,
                cc_count=cc_count,
                primary_area=primary,
                secondary_area=secondary,
                secondary_area_ratio=sec_ratio,
                ignored_fragments=ignored,
            )

    ys, xs = np.where(mask > 0)
    cy = float(ys.mean())
    cx = float(xs.mean())
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    row_runs: List[List[Run]] = []
    for y in range(y0, y1):
        runs = _row_runs(mask[y])
        # Ignore tiny speck runs (jagged residue after erode).
        runs = [(a, b) for a, b in runs if (b - a) >= 2]
        row_runs.append(runs)

    return WidthProfile(
        mask=mask,
        cx=cx,
        cy=cy,
        y0=y0,
        y1=y1,
        row_runs=row_runs,
        multi_lobe=multi,
        neck_depth=_neck_depth(row_runs),
        empty=False,
        cc_count=cc_count,
        primary_area=primary,
        secondary_area=secondary,
        secondary_area_ratio=sec_ratio,
        ignored_fragments=ignored,
    )


def band_min_width(
    profile: WidthProfile,
    y_top: float,
    y_bot: float,
    *,
    safety: float = 0.98,
) -> Tuple[float, float, float]:
    """Narrowest usable run in [y_top, y_bot): (width*safety, x_left, x_right).

    Returns (0, 0, 0) if the band has no single-run coverage.
    """
    if profile.empty or y_bot <= y_top:
        return 0.0, 0.0, 0.0

    y_a = max(profile.y0, int(np.floor(y_top)))
    y_b = min(profile.y1, int(np.ceil(y_bot)))
    if y_b <= y_a:
        return 0.0, 0.0, 0.0

    best_w = None
    best_left = 0
    best_right = 0
    for y in range(y_a, y_b):
        runs = profile.row_runs[y - profile.y0]
        if len(runs) == 0:
            # Peripheral row (erode residue filtered out): skip, don't block.
            continue
        if len(runs) > 1:
            # Genuinely multi-run: band crosses a lobe boundary, unusable.
            return 0.0, 0.0, 0.0
        left, right = runs[0]
        w = right - left
        if best_w is None or w < best_w:
            best_w = w
            best_left = left
            best_right = right

    if best_w is None or best_w <= 0:
        return 0.0, 0.0, 0.0
    usable = float(best_w) * float(safety)
    # Keep centered in the narrowest run's interval.
    mid = 0.5 * (best_left + best_right)
    half = 0.5 * usable
    return usable, mid - half, mid + half
