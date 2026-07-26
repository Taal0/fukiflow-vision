"""Debug overlay: draw YOLO-seg polygons + reading-order labels (and optional panels)."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from vision.types import Bubble, Panel

# BGR for OpenCV drawing (image is RGB; we convert for drawing consistency).
_OUTLINE_BGR = (0, 220, 0)
_FILL_BGR = (0, 220, 0)
_PANEL_BGR = (220, 140, 0)  # cyan-ish in BGR → distinct from bubble green
_TEXT_BG_BGR = (0, 0, 0)
_TEXT_FG_BGR = (255, 255, 255)
_PANEL_TEXT_FG_BGR = (255, 255, 0)


def draw_bubbles(
    img_rgb: np.ndarray,
    bubbles: Sequence[Bubble],
    *,
    panels: Optional[Sequence[Panel]] = None,
    alpha: float = 0.22,
    thickness: int = 2,
) -> np.ndarray:
    """Return RGB copy with bubble polygons + labels; optional panel bboxes."""
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {img_rgb.shape}")

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    fill_layer = bgr.copy()
    h, w = bgr.shape[:2]
    labeled: list[tuple[int, np.ndarray]] = []

    for i, b in enumerate(bubbles):
        n = b.reading_order if b.reading_order is not None else i + 1
        pts = _poly_int(b.polygon)
        if pts is None:
            x1, y1, x2, y2 = (int(v) for v in b.bbox)
            pts = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
            )
        if len(pts) < 3:
            continue
        cv2.fillPoly(fill_layer, [pts], _FILL_BGR)
        labeled.append((n, pts))

    cv2.addWeighted(fill_layer, alpha, bgr, 1.0 - alpha, 0, dst=bgr)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(1.2, min(w, h) / 900))

    if panels:
        panel_scale = scale * 1.15
        for p in panels:
            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), _PANEL_BGR, thickness + 1)
            label = str(p.reading_order if p.reading_order else p.id)
            (tw, th), baseline = cv2.getTextSize(label, font, panel_scale, 2)
            tx = max(0, min(x1 + 4, w - tw - 6))
            ty = max(th + 4, min(y1 + th + 6, h - 2))
            cv2.rectangle(
                bgr,
                (tx - 2, ty - th - 2),
                (tx + tw + 2, ty + baseline + 2),
                _PANEL_BGR,
                thickness=-1,
            )
            cv2.putText(
                bgr,
                label,
                (tx, ty),
                font,
                panel_scale,
                _PANEL_TEXT_FG_BGR,
                2,
                cv2.LINE_AA,
            )

    for n, pts in labeled:
        cv2.polylines(bgr, [pts], isClosed=True, color=_OUTLINE_BGR, thickness=thickness)
        x, y, _, _ = cv2.boundingRect(pts)
        label = str(n)
        (tw, th), baseline = cv2.getTextSize(label, font, scale, 2)
        tx = max(0, min(x, w - tw - 6))
        ty = max(th + 4, min(y + th + 4, h - 2))
        cv2.rectangle(
            bgr,
            (tx - 2, ty - th - 2),
            (tx + tw + 2, ty + baseline + 2),
            _TEXT_BG_BGR,
            thickness=-1,
        )
        cv2.putText(bgr, label, (tx, ty), font, scale, _TEXT_FG_BGR, 2, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _poly_int(polygon: Sequence[Sequence[float]]) -> np.ndarray | None:
    pts: list[list[int]] = []
    for p in polygon or []:
        if p is None or len(p) < 2:
            continue
        pts.append([int(round(float(p[0]))), int(round(float(p[1])))])
    if len(pts) < 3:
        return None
    return np.asarray(pts, dtype=np.int32)
