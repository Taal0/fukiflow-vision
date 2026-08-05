"""Compose source + enabled layers → preview RGB; single-layer RGBA for export."""

from __future__ import annotations

from typing import Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision.document import Layer, PageDoc, Stroke
from vision.masks import as_bbox, erode_mask, polygon_mask
from vision.typeset.catalog import resolve_portable_font


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


def stroke_mask(stroke: Stroke, shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize a brush stroke as a binary mask (round stamps + segments)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = stroke.points or []
    if not pts:
        return mask
    radius = max(0.5, float(stroke.radius_px))
    # Draw with PIL for stable round stamps along the polyline.
    pil = Image.fromarray(mask * 255, mode="L")
    draw = ImageDraw.Draw(pil)
    r = radius
    diam = max(1, int(round(r * 2)))
    coords = [(float(p[0]), float(p[1])) for p in pts if len(p) >= 2]
    if not coords:
        return mask
    if len(coords) == 1:
        x, y = coords[0]
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=255,
        )
    else:
        # Wide line covers gaps between stamps; end caps are round.
        draw.line(coords, fill=255, width=max(1, diam), joint="curve")
        for x, y in coords:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return (np.asarray(pil, dtype=np.uint8) > 0).astype(np.uint8)


def apply_paint_layer(
    out: np.ndarray,
    source_rgb: np.ndarray,
    layer: Layer,
) -> np.ndarray:
    """Apply paint strokes onto ``out`` (fill → solid color, restore → source)."""
    if not layer.strokes:
        return out
    for stroke in layer.strokes:
        mask = stroke_mask(stroke, out.shape[:2])
        if not mask.any():
            continue
        if stroke.mode == "restore":
            out[mask > 0] = source_rgb[mask > 0]
        elif stroke.mode == "fill":
            if stroke.color is None:
                continue
            out[mask > 0] = stroke.color
    return out


def _paint_text(
    img_rgb: np.ndarray,
    layer: Layer,
    *,
    font_path: Optional[str] = None,
) -> np.ndarray:
    lay = layer.layout
    if lay is None or lay.status != "ok" or not lay.lines or lay.font_px is None:
        return img_rgb
    # Catalog / basename first; absolute path only as fallback so Mac-seeded
    # font_file paths still compose on other machines.
    resolved = resolve_portable_font(lay.font_file, fallback=font_path)
    if resolved is None:
        return img_rgb

    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    size = max(1, int(round(lay.font_px)))
    font = ImageFont.truetype(str(resolved), size=size)
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
    """preview = source + enabled clean/paint/text layers (z-order)."""
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
        elif layer.kind == "paint":
            out = apply_paint_layer(out, source_rgb, layer)
        elif layer.kind == "text":
            out = _paint_text(out, layer, font_path=font_path)
    return out


def render_layer_rgba(
    layer: Layer,
    shape: Tuple[int, int],
    *,
    font_path: Optional[str] = None,
    source_rgb: Optional[np.ndarray] = None,
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

    if layer.kind == "paint":
        # Bake strokes onto a transparent canvas. restore needs source pixels.
        if not layer.strokes:
            return rgba
        src = source_rgb
        if src is None:
            src = np.zeros((h, w, 3), dtype=np.uint8)
        # Start from zeros; apply fill as opaque color, restore as source RGB.
        for stroke in layer.strokes:
            mask = stroke_mask(stroke, (h, w))
            if not mask.any():
                continue
            if stroke.mode == "restore":
                rgba[mask > 0, :3] = src[mask > 0]
                rgba[mask > 0, 3] = 255
            elif stroke.mode == "fill" and stroke.color is not None:
                r, g, b = stroke.color
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
