"""Debug renderer: paint layout JSON onto a cleaned page."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision.types import Bubble, BubbleLayout


_EDITOR_BGR = (40, 40, 220)  # red-ish in BGR
_EDITOR_FILL_BGR = (40, 40, 220)


def draw_typeset(
    img_rgb: np.ndarray,
    layouts: Sequence[BubbleLayout],
    bubbles: Sequence[Bubble],
    *,
    font_path: str,
    mark_editor: bool = True,
) -> np.ndarray:
    """Return RGB copy with typeset text; escalate bubbles outlined."""
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {img_rgb.shape}")

    out = img_rgb.copy()
    by_id = {b.id: b for b in bubbles}

    if mark_editor:
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        for lay in layouts:
            if lay.status != "editor":
                continue
            b = by_id.get(lay.bubble_id)
            if b is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.bbox)
            overlay = bgr.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), _EDITOR_FILL_BGR, thickness=-1)
            cv2.addWeighted(overlay, 0.18, bgr, 0.82, 0, dst=bgr)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), _EDITOR_BGR, thickness=2)
            label = lay.reason or "editor"
            cv2.putText(
                bgr,
                label,
                (x1 + 4, max(y1 + 16, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                _EDITOR_BGR,
                1,
                cv2.LINE_AA,
            )
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    font_cache: dict[int, ImageFont.FreeTypeFont] = {}

    for lay in layouts:
        if lay.status != "ok" or not lay.lines or lay.font_px is None:
            continue
        size = max(1, int(round(lay.font_px)))
        if size not in font_cache:
            font_cache[size] = ImageFont.truetype(font_path, size=size)
        font = font_cache[size]
        for line in lay.lines:
            # PIL anchor "ls" = left baseline (Pillow ≥8).
            draw.text(
                (line.x, line.baseline_y),
                line.text,
                font=font,
                fill=(0, 0, 0),
                anchor="ls",
            )

    return np.asarray(pil, dtype=np.uint8)
