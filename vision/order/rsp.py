"""Recursive Space Partitioning (RSP) for manga reading order.

H-first (horizontal bands before columns), then V with RTL (right before left).
Candidate cuts are panel *edge* coordinates, not gap midpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from vision.types import BBox


@dataclass
class RspStats:
    """Tracks real ambiguities where no valid H/V cut exists."""

    fallbacks: List[List[int]] = field(default_factory=list)
    calls: int = 0


def _mid(box: BBox, axis: str) -> float:
    if axis == "H":
        return (box[1] + box[3]) / 2.0
    return (box[0] + box[2]) / 2.0


def _straddles(box: BBox, cut: float, axis: str, tol: float) -> bool:
    lo, hi = (box[1], box[3]) if axis == "H" else (box[0], box[2])
    return (cut - lo) > tol and (hi - cut) > tol


def _valid_cuts(
    boxes: Sequence[BBox], axis: str, tol: float
) -> List[Tuple[float, float]]:
    """Candidate cut = panel edge coord (gap midpoint DEGIL)."""
    idx = (1, 3) if axis == "H" else (0, 2)
    edges = sorted({b[idx[0]] for b in boxes} | {b[idx[1]] for b in boxes})
    out: List[Tuple[float, float]] = []
    for cut in edges:
        if any(_straddles(b, cut, axis, tol) for b in boxes):
            continue
        lo_side = [b for b in boxes if _mid(b, axis) < cut]
        hi_side = [b for b in boxes if _mid(b, axis) >= cut]
        if lo_side and hi_side:
            gap = min(
                min(abs(cut - b[idx[0]]), abs(cut - b[idx[1]])) for b in boxes
            )
            out.append((cut, gap))
    return out


def order_boxes(
    boxes: Sequence[BBox],
    ids: Sequence[int],
    tol: float,
    stats: RspStats,
) -> List[int]:
    """Return ids ordered for manga reading (H-first, then RTL vertical)."""
    stats.calls += 1
    if len(ids) <= 1:
        return list(ids)

    sub = [boxes[i] for i in ids]
    h = _valid_cuts(sub, "H", tol)
    if h:  # manga: bands before columns
        cut = max(h, key=lambda c: c[1])[0]
        top = [i for i in ids if _mid(boxes[i], "H") < cut]
        bot = [i for i in ids if _mid(boxes[i], "H") >= cut]
        return order_boxes(boxes, top, tol, stats) + order_boxes(
            boxes, bot, tol, stats
        )

    v = _valid_cuts(sub, "V", tol)
    if v:
        cut = max(v, key=lambda c: c[1])[0]
        right = [i for i in ids if _mid(boxes[i], "V") >= cut]
        left = [i for i in ids if _mid(boxes[i], "V") < cut]
        return order_boxes(boxes, right, tol, stats) + order_boxes(
            boxes, left, tol, stats
        )

    stats.fallbacks.append(list(ids))
    # Score fallback: top-right first (larger x, smaller y)
    return sorted(ids, key=lambda i: (-boxes[i][1] + boxes[i][0]), reverse=True)
