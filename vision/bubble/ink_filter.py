"""Detect-stage ink filter: drop empty FP; force editor on textured ink."""

from __future__ import annotations

import logging
from typing import List, Sequence, Set, Tuple

import numpy as np

from vision.masks import InkStructure, ink_structure
from vision.types import Bubble

log = logging.getLogger(__name__)

# Decision labels (for measure/debug; classify returns kept + force_editor only)
DROP = "DROP"
FORCE_EDITOR = "FORCE_EDITOR"
FILL_CANDIDATE = "FILL_CANDIDATE"


class InkFilterSettings:
    """Thresholds for classify(); defaults match app.settings."""

    def __init__(
        self,
        *,
        ink_tol: float = 40.0,
        ink_min_px: int = 50,
        ink_min_cc_area: int = 12,
        conf_soft_hi: float = 0.80,
        conf_hard: float = 0.65,
        inset: int = 10,
        ring: int = 6,
        min_ring_px: int = 30,
        uniform_tol: float = 18.0,
        uniform_frac: float = 0.85,
    ) -> None:
        self.ink_tol = float(ink_tol)
        self.ink_min_px = int(ink_min_px)
        self.ink_min_cc_area = int(ink_min_cc_area)
        self.conf_soft_hi = float(conf_soft_hi)
        self.conf_hard = float(conf_hard)
        self.inset = int(inset)
        self.ring = int(ring)
        self.min_ring_px = int(min_ring_px)
        self.uniform_tol = float(uniform_tol)
        self.uniform_frac = float(uniform_frac)


def decide(structure: InkStructure) -> str:
    """Map ink structure → DROP | FORCE_EDITOR | FILL_CANDIDATE.

    Unusable ring → FORCE_EDITOR (can't measure safely).
    Empty (no solid ink component) → DROP.
    Text present + uniform ring → FILL; text + textured → editor.
    """
    if not structure.ring_ok:
        return FORCE_EDITOR
    if not structure.text_present:
        return DROP
    if structure.ring_uniform:
        return FILL_CANDIDATE
    return FORCE_EDITOR


def _measure(
    img_rgb: np.ndarray,
    bubble: Bubble,
    settings: InkFilterSettings,
) -> InkStructure:
    return ink_structure(
        img_rgb,
        bubble.polygon,
        bubble.bbox,
        inset=settings.inset,
        ring=settings.ring,
        tol=settings.ink_tol,
        min_ring_px=settings.min_ring_px,
        min_px=settings.ink_min_px,
        min_cc_area=settings.ink_min_cc_area,
        uniform_tol=settings.uniform_tol,
        uniform_frac=settings.uniform_frac,
    )


def classify(
    img_rgb: np.ndarray,
    bubbles: Sequence[Bubble],
    settings: InkFilterSettings,
) -> Tuple[List[Bubble], Set[int]]:
    """Filter detect output.

    Returns:
      kept_bubbles — DROP removed; ids preserved from input
      force_editor_ids — kept ids that must skip fill
    """
    kept: List[Bubble] = []
    force_editor: Set[int] = set()
    dropped = 0

    for b in bubbles:
        structure = _measure(img_rgb, b, settings)
        action = decide(structure)
        if action == DROP:
            dropped += 1
            log.debug(
                "ink_filter DROP id=%s conf=%.3f dark_px=%d max_cc=%d",
                b.id,
                b.confidence,
                structure.dark_px,
                structure.max_cc_area,
            )
            continue
        kept.append(b)
        if action == FORCE_EDITOR:
            force_editor.add(int(b.id))
            log.debug(
                "ink_filter FORCE_EDITOR id=%s conf=%.3f dark_px=%d max_cc=%d",
                b.id,
                b.confidence,
                structure.dark_px,
                structure.max_cc_area,
            )

    if dropped:
        log.info("ink_filter dropped %d bubble(s)", dropped)
    return kept, force_editor


def measure_rows(
    img_rgb: np.ndarray,
    bubbles: Sequence[Bubble],
    settings: InkFilterSettings,
    *,
    page: str,
) -> List[dict]:
    """Per-bubble rows for calibration CSV (no filtering)."""
    rows: List[dict] = []
    for b in bubbles:
        structure = _measure(img_rgb, b, settings)
        action = decide(structure)
        x1, y1, x2, y2 = (int(v) for v in b.bbox)
        rows.append(
            {
                "page": page,
                "id": int(b.id),
                "conf": float(b.confidence),
                "bbox": f"{x1},{y1},{x2},{y2}",
                "dark_px": structure.dark_px,
                "dark_frac": round(structure.dark_frac, 6),
                "cc_count": structure.cc_count,
                "max_cc_area": structure.max_cc_area,
                "ring_uniform": int(structure.ring_uniform),
                "text_present": int(structure.text_present),
                "decision": action,
            }
        )
    return rows


def settings_from_app(app_settings: object) -> InkFilterSettings:
    """Build InkFilterSettings from app.settings.Settings-like object."""
    return InkFilterSettings(
        ink_tol=getattr(app_settings, "ink_tol"),
        ink_min_px=getattr(app_settings, "ink_min_px"),
        ink_min_cc_area=getattr(app_settings, "ink_min_cc_area"),
        conf_soft_hi=getattr(app_settings, "bubble_conf_soft_hi"),
        conf_hard=getattr(app_settings, "bubble_conf"),
        inset=getattr(app_settings, "clean_inset"),
        ring=getattr(app_settings, "clean_ring"),
        min_ring_px=getattr(app_settings, "clean_min_ring_px"),
        uniform_tol=getattr(app_settings, "clean_uniform_tol"),
        uniform_frac=getattr(app_settings, "clean_uniform_frac"),
    )


__all__ = [
    "DROP",
    "FORCE_EDITOR",
    "FILL_CANDIDATE",
    "InkFilterSettings",
    "classify",
    "decide",
    "measure_rows",
    "settings_from_app",
]
