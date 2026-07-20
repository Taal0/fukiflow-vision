from __future__ import annotations

import base64
from typing import Sequence, Tuple

import cv2
import numpy as np


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes to RGB uint8 ndarray."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_image_b64(b64: str) -> np.ndarray:
    raw = b64
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return decode_image_bytes(base64.b64decode(raw))


def encode_png_bytes(img_rgb: np.ndarray) -> bytes:
    if img_rgb.ndim == 2:
        to_encode = img_rgb
    else:
        to_encode = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", to_encode)
    if not ok:
        raise ValueError("Could not encode PNG")
    return buf.tobytes()


def encode_png_b64(img: np.ndarray) -> str:
    return base64.b64encode(encode_png_bytes(img)).decode("ascii")


def decode_mask_b64(b64: str) -> np.ndarray:
    raw = b64
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Could not decode mask PNG")
    return mask


def _as_quads(polygon) -> list:
    """Normalize Box.polygon into a list of quads (each Nx2)."""
    if not polygon:
        return []
    # Single point list: [[x,y], ...]
    first = polygon[0]
    if first and not isinstance(first[0], (list, tuple)):
        return [polygon]
    # List of quads
    return list(polygon)


def polygons_to_mask(
    shape_hw: Tuple[int, int],
    polygons: Sequence,
) -> np.ndarray:
    """Rasterize polygons into a binary mask (255 = inpaint)."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        for quad in _as_quads(poly):
            pts = np.array(quad, dtype=np.int32)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], 255)
    return mask


def boxes_to_mask(shape_hw: Tuple[int, int], boxes) -> np.ndarray:
    polys = []
    for b in boxes:
        poly = b.polygon if hasattr(b, "polygon") else b["polygon"]
        polys.append(poly)
    return polygons_to_mask(shape_hw, polys)
