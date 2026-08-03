"""Font-fit ladder + vertical window iteration for one bubble."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from vision.types import BBox, BubbleLayout, TypesetLine
from vision.typeset.font import FontFace
from vision.typeset.profile import WidthProfile, band_min_width
from vision.typeset.wrap import Hyphenator, wrap_lines


def px_to_pt(px: float, dpi: float) -> float:
    return float(px) * 72.0 / float(dpi) if dpi else float(px)


def layout_frame(
    lines: Sequence[TypesetLine],
    *,
    ascent: float,
    descent: float,
) -> Optional[BBox]:
    """Axis-aligned text block from line boxes (IDML text frame)."""
    if not lines:
        return None
    x1 = min(line.x for line in lines)
    x2 = max(line.x + line.width for line in lines)
    y1 = min(line.baseline_y - ascent for line in lines)
    y2 = max(line.baseline_y + descent for line in lines)
    return (
        int(math.floor(x1)),
        int(math.floor(y1)),
        int(math.ceil(x2)),
        int(math.ceil(y2)),
    )


def enrich_layout_export(
    layout: BubbleLayout,
    font: FontFace,
    *,
    dpi: float,
) -> BubbleLayout:
    """Attach pt sizes, font file/PS name, and frame for PSD/IDML export."""
    updates: dict = {
        "font_file": str(font.path),
        "font_postscript_name": font.postscript_name,
        "font_family": layout.font_family or font.family,
    }
    if layout.font_px is not None:
        updates["font_pt"] = px_to_pt(layout.font_px, dpi)
        m = font.metrics(layout.font_px)
        if layout.lines:
            updates["frame"] = layout_frame(
                layout.lines, ascent=m.ascent, descent=m.descent
            )
    if layout.leading_px is not None:
        updates["leading_pt"] = px_to_pt(layout.leading_px, dpi)
    return layout.model_copy(update=updates)


@dataclass(frozen=True)
class FitSettings:
    leading: float = 1.15
    width_safety: float = 0.98
    max_iters: int = 3
    align: str = "center"


def font_ladder(
    page_h: int,
    *,
    min_ratio: float,
    max_ratio: float,
    steps: int,
) -> List[float]:
    """Page-height-relative font sizes, ascending."""
    steps = max(1, int(steps))
    lo = float(min_ratio) * float(page_h)
    hi = float(max_ratio) * float(page_h)
    if steps == 1:
        return [hi]
    return [float(v) for v in np.linspace(lo, hi, steps)]


def _line_bands(
    cy: float,
    n: int,
    lh: float,
) -> List[Tuple[float, float]]:
    """Vertical bands for N lines centered on cy: [(y_top, y_bot), ...]."""
    block_h = n * lh
    y0 = cy - block_h / 2.0
    return [(y0 + i * lh, y0 + (i + 1) * lh) for i in range(n)]


def _place_lines(
    texts: Sequence[str],
    bands: Sequence[Tuple[float, float]],
    widths_meta: Sequence[Tuple[float, float, float]],
    *,
    font: FontFace,
    size_px: float,
    align: str,
) -> List[TypesetLine]:
    """Compute x / baseline_y for wrapped lines (center or left within band)."""
    m = font.metrics(size_px)
    out: List[TypesetLine] = []
    for text, (y_top, y_bot), (usable, x_left, x_right) in zip(
        texts, bands, widths_meta
    ):
        tw = font.measure(text, size_px)
        if align == "left":
            x = x_left
        else:
            # center within the usable interval
            x = 0.5 * (x_left + x_right) - 0.5 * tw
        baseline = y_top + (y_bot - y_top) * (m.ascent / m.em_height) if m.em_height else y_top
        out.append(
            TypesetLine(
                text=text,
                x=float(x),
                baseline_y=float(baseline),
                width=float(tw),
            )
        )
    return out


def try_fit_size(
    text: str,
    profile: WidthProfile,
    font: FontFace,
    size_px: float,
    *,
    settings: FitSettings,
    hyphenator: Optional[Hyphenator] = None,
) -> Optional[Tuple[List[TypesetLine], bool, float]]:
    """Try one font size. Returns (lines, hyphenated, leading_px) or None."""
    if profile.empty or not text.strip():
        return None

    lh = font.line_height(size_px, settings.leading)
    if lh <= 0:
        return None

    # Initial N from available vertical span.
    span = max(1, profile.y1 - profile.y0)
    n = max(1, int(span / lh))
    measure = lambda s, sz=size_px: font.measure(s, sz)

    for _ in range(settings.max_iters):
        bands = _line_bands(profile.cy, n, lh)
        widths: List[float] = []
        metas: List[Tuple[float, float, float]] = []
        for y_top, y_bot in bands:
            usable, x_left, x_right = band_min_width(
                profile, y_top, y_bot, safety=settings.width_safety
            )
            widths.append(usable)
            metas.append((usable, x_left, x_right))

        wrapped, hyphenated = wrap_lines(
            text, widths, measure, hyphenator=hyphenator
        )
        if wrapped is None:
            return None
        n_prime = len(wrapped)
        if n_prime == 0:
            return [], False, lh
        if n_prime == n:
            # Recompute metas for the settled bands (same n).
            lines = _place_lines(
                wrapped, bands, metas, font=font, size_px=size_px, align=settings.align
            )
            return lines, hyphenated, lh
        n = n_prime

    # Did not settle within max_iters → treat as no-fit at this size.
    return None


def _force_place(
    text: str,
    profile: WidthProfile,
    font: FontFace,
    size_px: float,
    *,
    settings: FitSettings,
) -> Tuple[List[TypesetLine], bool, float]:
    """Best-effort placement when forced size does not fit (overflow)."""
    lh = font.line_height(size_px, settings.leading)
    if lh <= 0:
        lh = float(size_px) * float(settings.leading)
    words = text.split() or [text]
    # One word per line, centered on profile; may exceed band width.
    n = max(1, len(words))
    bands = _line_bands(profile.cy, n, lh)
    metas: List[Tuple[float, float, float]] = []
    for y_top, y_bot in bands:
        usable, x_left, x_right = band_min_width(
            profile, y_top, y_bot, safety=settings.width_safety
        )
        metas.append((usable, x_left, x_right))
    # Pad metas if profile bands shorter than words (cy-centered may clip).
    while len(metas) < len(words):
        if metas:
            metas.append(metas[-1])
            bands.append(bands[-1])
        else:
            half = 40.0
            metas.append((2 * half, float(profile.cx) - half, float(profile.cx) + half))
            bands.append((float(profile.y0), float(profile.y1)))
    lines = _place_lines(
        words, bands[: len(words)], metas[: len(words)], font=font, size_px=size_px, align=settings.align
    )
    return lines, False, lh


def fit_bubble(
    text: str,
    profile: WidthProfile,
    font: FontFace,
    ladder: Sequence[float],
    page_h: int,
    *,
    bubble_id: int,
    settings: Optional[FitSettings] = None,
    hyphenator: Optional[Hyphenator] = None,
    force_size_px: Optional[float] = None,
    font_scale: Optional[float] = None,
) -> BubbleLayout:
    """Binary-search the ladder for the largest size that fits.

    ``force_size_px`` skips the ladder (overflow=True if it does not fit).
    ``font_scale`` multiplies the ladder-chosen size, then re-fits.
    """
    settings = settings or FitSettings()
    text = text.strip()
    if not text:
        return BubbleLayout(bubble_id=bubble_id, status="editor", reason="no_text")
    if profile.empty:
        return BubbleLayout(bubble_id=bubble_id, status="editor", reason="empty_mask")
    if profile.multi_lobe:
        return BubbleLayout(bubble_id=bubble_id, status="editor", reason="multi_lobe")

    override_source: Optional[str] = None
    overflow = False

    if force_size_px is not None:
        size = max(1.0, float(force_size_px))
        override_source = "font_px"
        fitted = try_fit_size(
            text, profile, font, size, settings=settings, hyphenator=hyphenator
        )
        if fitted is None:
            lines, hyphenated, lh = _force_place(
                text, profile, font, size, settings=settings
            )
            overflow = True
        else:
            lines, hyphenated, lh = fitted
        m = font.metrics(size)
        return BubbleLayout(
            bubble_id=bubble_id,
            status="ok",
            font_family=font.family,
            font_px=float(size),
            font_ratio=float(size) / float(page_h) if page_h else None,
            leading_px=float(lh),
            font_postscript_name=font.postscript_name,
            font_file=str(font.path),
            align=settings.align,
            lines=lines,
            hyphenated=hyphenated,
            frame=layout_frame(lines, ascent=m.ascent, descent=m.descent),
            overflow=overflow,
            override_source=override_source,
        )

    sizes = sorted(float(s) for s in ladder)
    if not sizes:
        return BubbleLayout(bubble_id=bubble_id, status="editor", reason="no_fit")

    best: Optional[Tuple[float, List[TypesetLine], bool, float]] = None
    lo, hi = 0, len(sizes) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        size = sizes[mid]
        fitted = try_fit_size(
            text, profile, font, size, settings=settings, hyphenator=hyphenator
        )
        if fitted is None:
            hi = mid - 1
        else:
            lines, hyphenated, lh = fitted
            best = (size, lines, hyphenated, lh)
            lo = mid + 1

    if best is None:
        return BubbleLayout(bubble_id=bubble_id, status="editor", reason="no_fit")

    size, lines, hyphenated, lh = best

    if font_scale is not None and float(font_scale) != 1.0:
        override_source = "font_scale"
        scaled = max(1.0, float(size) * float(font_scale))
        fitted = try_fit_size(
            text, profile, font, scaled, settings=settings, hyphenator=hyphenator
        )
        if fitted is None:
            lines, hyphenated, lh = _force_place(
                text, profile, font, scaled, settings=settings
            )
            overflow = True
            size = scaled
        else:
            lines, hyphenated, lh = fitted
            size = scaled

    m = font.metrics(size)
    return BubbleLayout(
        bubble_id=bubble_id,
        status="ok",
        font_family=font.family,
        font_px=float(size),
        font_ratio=float(size) / float(page_h) if page_h else None,
        leading_px=float(lh),
        font_postscript_name=font.postscript_name,
        font_file=str(font.path),
        align=settings.align,
        lines=lines,
        hyphenated=hyphenated,
        frame=layout_frame(lines, ascent=m.ascent, descent=m.descent),
        overflow=overflow,
        override_source=override_source,
    )
