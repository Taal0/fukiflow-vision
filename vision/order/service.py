"""Reading-order orchestration: panel NMS → RSP → assign → flatten → score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from vision.order.assign import AssignSettings, assign_bubbles
from vision.order.rsp import RspStats, order_boxes
from vision.panel.detect import PanelDetector, get_detector as get_panel_detector
from vision.types import BBox, Bubble, Panel

# Module-level singleton set by app lifespan
_service: Optional["OrderService"] = None


def set_service(service: Optional["OrderService"]) -> None:
    global _service
    _service = service


def get_service() -> "OrderService":
    if _service is None:
        raise RuntimeError("OrderService not initialized")
    return _service


@dataclass
class OrderSettings:
    tol_frac: float = 0.02
    nms_iou: float = 0.60
    iob_contained: float = 0.70
    iob_partial: float = 0.15
    ambig_delta: float = 0.05
    conf_min: float = 0.50


@dataclass
class OrderResult:
    panels: List[Panel]
    bubbles: List[Bubble]
    order_confidence: float
    needs_review: bool
    n_orphan: int = 0
    n_ambiguous: int = 0
    n_fallback: int = 0
    rsp_calls: int = 0
    fallback_groups: List[List[int]] = field(default_factory=list)


def _iou(a: BBox, b: BBox) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_panels(panels: Sequence[Panel], iou_thresh: float) -> List[Panel]:
    """Greedy NMS by confidence; keeps original detect ids."""
    if not panels:
        return []
    ordered = sorted(panels, key=lambda p: p.confidence, reverse=True)
    kept: List[Panel] = []
    for p in ordered:
        if any(_iou(p.bbox, k.bbox) >= iou_thresh for k in kept):
            continue
        kept.append(p)
    # stable-ish: restore detect-id order for determinism before RSP
    kept.sort(key=lambda p: p.id)
    return kept


def _overlapping_pair_frac(panels: Sequence[Panel], nms_iou: float) -> float:
    """Fraction of pairs with 0.1 < IoU < NMS_IOU (soft overlaps)."""
    n = len(panels)
    if n < 2:
        return 0.0
    total = n * (n - 1) // 2
    soft = 0
    for i in range(n):
        for j in range(i + 1, n):
            u = _iou(panels[i].bbox, panels[j].bbox)
            if 0.1 < u < nms_iou:
                soft += 1
    return soft / total


class OrderService:
    """Panel detect + RSP reading order for surviving (ink-filtered) bubbles."""

    def __init__(
        self,
        settings: Optional[OrderSettings] = None,
        panel_detector: Optional[PanelDetector] = None,
    ) -> None:
        self.settings = settings or OrderSettings()
        self._panel_detector = panel_detector
        self.ready = False

    def load(self) -> None:
        # Detector is loaded separately; this marks the service ready.
        self.ready = True

    def _panels_from(self, img_rgb: np.ndarray) -> List[Panel]:
        det = self._panel_detector
        if det is None:
            det = get_panel_detector()
        return det.predict(img_rgb)

    def order(
        self,
        img_rgb: np.ndarray,
        bubbles: Sequence[Bubble],
        *,
        panels: Optional[Sequence[Panel]] = None,
    ) -> OrderResult:
        h, w = img_rgb.shape[:2]
        tol = min(w, h) * self.settings.tol_frac
        s = self.settings

        raw_panels = list(panels) if panels is not None else self._panels_from(img_rgb)
        deduped = nms_panels(raw_panels, s.nms_iou)

        panel_stats = RspStats()
        if deduped:
            boxes = [p.bbox for p in deduped]
            idxs = list(range(len(deduped)))
            ordered_idxs = order_boxes(boxes, idxs, tol, panel_stats)
            ordered_panels: List[Panel] = []
            for rank, i in enumerate(ordered_idxs, start=1):
                ordered_panels.append(
                    deduped[i].model_copy(update={"reading_order": rank})
                )
        else:
            ordered_panels = []

        assign_settings = AssignSettings(
            iob_contained=s.iob_contained,
            iob_partial=s.iob_partial,
            ambig_delta=s.ambig_delta,
        )
        assigned = assign_bubbles(
            bubbles,
            ordered_panels,
            page_size=(w, h),
            settings=assign_settings,
        )
        panels_out = assigned.panels
        # ensure reading_order on synthetic
        if len(panels_out) == 1 and panels_out[0].is_synthetic:
            panels_out[0] = panels_out[0].model_copy(update={"reading_order": 1})

        bubble_stats = RspStats()
        # Order bubbles within each panel (orphans included positionally), then flatten
        by_panel: dict[int, List[Bubble]] = {p.id: [] for p in panels_out}
        unassigned: List[Bubble] = []
        for b in assigned.bubbles:
            if b.panel_id is not None and b.panel_id in by_panel:
                by_panel[b.panel_id].append(b)
            else:
                unassigned.append(b)

        # Walk panels in reading_order
        panels_sorted = sorted(panels_out, key=lambda p: p.reading_order or 0)
        flat: List[Bubble] = []
        for p in panels_sorted:
            group = by_panel.get(p.id, [])
            if not group:
                continue
            gboxes = [b.bbox for b in group]
            gids = list(range(len(group)))
            ordered_g = order_boxes(gboxes, gids, tol, bubble_stats)
            for i in ordered_g:
                flat.append(group[i])

        # Any stragglers (shouldn't happen) at end
        flat.extend(unassigned)

        # Page-level ambiguity score
        n_bub = max(1, len(assigned.bubbles))
        rsp_calls = panel_stats.calls + bubble_stats.calls
        n_fallback = len(panel_stats.fallbacks) + len(bubble_stats.fallbacks)
        rsp_fallback_frac = (n_fallback / rsp_calls) if rsp_calls > 0 else 0.0
        overlap_frac = _overlapping_pair_frac(
            [p for p in panels_out if not p.is_synthetic], s.nms_iou
        )
        ambig_frac = assigned.n_ambiguous / n_bub
        orphan_frac = assigned.n_orphan / n_bub

        conf = 1.0
        conf -= 0.30 * overlap_frac
        conf -= 0.30 * rsp_fallback_frac
        conf -= 0.20 * ambig_frac
        conf -= 0.20 * orphan_frac
        conf = max(0.0, min(1.0, conf))
        needs_review = conf < s.conf_min

        out_bubbles: List[Bubble] = []
        for rank, b in enumerate(flat, start=1):
            out_bubbles.append(
                b.model_copy(
                    update={
                        "reading_order": rank,
                        "order_source": "algo",
                        "order_confidence": conf,
                    }
                )
            )

        return OrderResult(
            panels=panels_sorted,
            bubbles=out_bubbles,
            order_confidence=conf,
            needs_review=needs_review,
            n_orphan=assigned.n_orphan,
            n_ambiguous=assigned.n_ambiguous,
            n_fallback=n_fallback,
            rsp_calls=rsp_calls,
            fallback_groups=panel_stats.fallbacks + bubble_stats.fallbacks,
        )
