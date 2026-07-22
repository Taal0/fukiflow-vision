from __future__ import annotations

import base64

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
