"""Assign bubbles to panels via Intersection-over-Bubble (IoB) + bbox min-distance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from vision.types import BBox, Bubble, Panel


@dataclass
class AssignSettings:
    iob_contained: float = 0.70
    iob_partial: float = 0.15
    ambig_delta: float = 0.05


@dataclass
class AssignResult:
    panels: List[Panel]
    bubbles: List[Bubble]
    n_orphan: int = 0
    n_ambiguous: int = 0


def iob(bubble_bbox: BBox, panel_bbox: BBox) -> float:
    """Intersection over BUBBLE area (not IoU: small bubble in large panel → 1.0)."""
    b, p = bubble_bbox, panel_bbox
    ix = max(0, min(b[2], p[2]) - max(b[0], p[0]))
    iy = max(0, min(b[3], p[3]) - max(b[1], p[1]))
    area = (b[2] - b[0]) * (b[3] - b[1])
    return (ix * iy) / area if area > 0 else 0.0


def bbox_min_dist(b: BBox, p: BBox) -> float:
    """Edge-to-edge distance (not centroid): gutter bubbles go to physical neighbor."""
    dx = max(0, max(b[0], p[0]) - min(b[2], p[2]))
    dy = max(0, max(b[1], p[1]) - min(b[3], p[3]))
    return (dx * dx + dy * dy) ** 0.5


def synthetic_full_page(width: int, height: int) -> Panel:
    return Panel(
        id=0,
        bbox=(0, 0, max(0, width - 1), max(0, height - 1)),
        confidence=0.0,
        reading_order=1,
        is_synthetic=True,
    )


def assign_bubbles(
    bubbles: Sequence[Bubble],
    panels: Sequence[Panel],
    *,
    page_size: Tuple[int, int],
    settings: Optional[AssignSettings] = None,
) -> AssignResult:
    """Assign each bubble to a panel; mutates copies, returns new lists."""
    s = settings or AssignSettings()
    width, height = page_size

    if not panels:
        synth = synthetic_full_page(width, height)
        out_bubbles: List[Bubble] = []
        for b in bubbles:
            out_bubbles.append(
                b.model_copy(
                    update={
                        "panel_id": synth.id,
                        "assign_mode": "no_panel",
                    }
                )
            )
        return AssignResult(panels=[synth], bubbles=out_bubbles)

    panel_list = list(panels)
    out: List[Bubble] = []
    n_orphan = 0
    n_ambiguous = 0

    for b in bubbles:
        scores = [(iob(b.bbox, p.bbox), p) for p in panel_list]
        scores.sort(key=lambda t: t[0], reverse=True)
        best_iob, best_panel = scores[0]
        second_iob = scores[1][0] if len(scores) > 1 else 0.0

        mode: str
        chosen = best_panel

        if best_iob > s.iob_contained:
            mode = "contained"
        elif best_iob > s.iob_partial:
            mode = "partial"
        else:
            # orphan: nearest panel by edge distance
            chosen = min(panel_list, key=lambda p: bbox_min_dist(b.bbox, p.bbox))
            mode = "orphan"
            n_orphan += 1

        if (
            mode in ("contained", "partial")
            and len(scores) > 1
            and (best_iob - second_iob) < s.ambig_delta
        ):
            mode = "ambiguous"
            n_ambiguous += 1
            # keep higher IoB assignment

        out.append(
            b.model_copy(
                update={
                    "panel_id": chosen.id,
                    "assign_mode": mode,
                }
            )
        )

    return AssignResult(
        panels=panel_list,
        bubbles=out,
        n_orphan=n_orphan,
        n_ambiguous=n_ambiguous,
    )
