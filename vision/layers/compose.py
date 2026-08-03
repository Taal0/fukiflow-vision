"""Compose source + enabled layers → preview RGB; single-layer RGBA for export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision.document import Layer, PageDoc
from vision.masks import as_bbox, erode_mask, polygon_mask


def render_mask(layer: Layer, shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize layer mask (inset_polygon → polygon_mask + erode)."""
    h, w = shape
    mask_spec = layer.mask
    if mask_spec is None:
        return np.zeros((h, w), dtype=np.uint8)

    if mask_spec.type == "inset_polygon" or not mask_spec.type:
        bbox = as_bbox(mask_spec.bbox or layer.bbox)
        poly = polygon_mask((h, w), mask_spec.polygon, bbox)
        return erode_mask(poly, int(mask_spec.inset))

    if mask_spec.type == "png_alpha" and mask_spec.file:
        rgba = np.asarray(Image.open(mask_spec.file).convert("RGBA"), dtype=np.uint8)
        if rgba.shape[0] != h or rgba.shape[1] != w:
            full = np.zeros((h, w), dtype=np.uint8)
            # Place at bbox origin if patch-sized.
            x1, y1, x2, y2 = as_bbox(mask_spec.bbox or layer.bbox)
            ph, pw = rgba.shape[:2]
            y2e, x2e = min(h, y1 + ph), min(w, x1 + pw)
            full[y1:y2e, x1:x2e] = (rgba[: y2e - y1, : x2e - x1, 3] > 0).astype(
                np.uint8
            )
            return full
        return (rgba[:, :, 3] > 0).astype(np.uint8)

    # rle / unknown → empty (not implemented this iteration)
    return np.zeros((h, w), dtype=np.uint8)


def _paint_text(
    img_rgb: np.ndarray,
    layer: Layer,
    *,
    font_path: Optional[str] = None,
) -> np.ndarray:
    lay = layer.layout
    if lay is None or lay.status != "ok" or not lay.lines or lay.font_px is None:
        return img_rgb
    # Prefer per-layer font_file so mixed-font pages compose correctly.
    path = lay.font_file or font_path
    if not path or not Path(path).is_file():
        return img_rgb

    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    size = max(1, int(round(lay.font_px)))
    font = ImageFont.truetype(str(path), size=size)
    for line in lay.lines:
        draw.text(
            (line.x, line.baseline_y),
            line.text,
            font=font,
            fill=(0, 0, 0),
            anchor="ls",
        )
    return np.asarray(pil, dtype=np.uint8)


def compose(
    source_rgb: np.ndarray,
    doc: PageDoc,
    *,
    font_path: Optional[str] = None,
    off: Optional[Set[str]] = None,
    only_kind: Optional[str] = None,
) -> np.ndarray:
    """preview = source + enabled clean layers + enabled text layers (z-order)."""
    if source_rgb.ndim != 3 or source_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {source_rgb.shape}")

    out = source_rgb.copy()
    h, w = out.shape[:2]
    disabled = off or set()
    layers = sorted(doc.layers, key=lambda L: (L.z, L.id))

    for layer in layers:
        if not layer.enabled or layer.id in disabled:
            continue
        if only_kind and layer.kind != only_kind:
            continue
        if layer.kind == "clean":
            if layer.fill_color is None:
                continue
            mask = render_mask(layer, (h, w))
            if mask.any():
                out[mask > 0] = layer.fill_color
        elif layer.kind == "text":
            out = _paint_text(out, layer, font_path=font_path)
    return out


def render_layer_rgba(
    layer: Layer,
    shape: Tuple[int, int],
    *,
    font_path: Optional[str] = None,
) -> np.ndarray:
    """Single layer as HxWx4 RGBA (transparent outside content)."""
    h, w = shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if layer.kind == "clean":
        if layer.fill_color is None:
            return rgba
        mask = render_mask(layer, (h, w))
        if not mask.any():
            return rgba
        r, g, b = layer.fill_color
        rgba[mask > 0] = (r, g, b, 255)
        return rgba

    if layer.kind == "text":
        # Paint onto white then recover: non-white → opaque black text.
        white = np.full((h, w, 3), 255, dtype=np.uint8)
        painted_w = _paint_text(white, layer, font_path=font_path)
        ink = np.any(painted_w < 250, axis=2)
        rgba[ink] = (0, 0, 0, 255)
        return rgba

    return rgba


def apply_layer_overrides(
    doc: PageDoc,
    *,
    off: Optional[Sequence[str]] = None,
    only_kind: Optional[str] = None,
) -> PageDoc:
    """Return a copy of doc with enabled flags adjusted (does not mutate)."""
    off_set = set(off or [])
    layers = []
    for L in doc.layers:
        enabled = L.enabled and L.id not in off_set
        if only_kind and L.kind != only_kind:
            enabled = False
        layers.append(L.model_copy(update={"enabled": enabled}))
    return doc.model_copy(update={"layers": layers})
