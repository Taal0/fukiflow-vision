from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def detect_device() -> str:
    override = os.environ.get("DEVICE")
    if override:
        return override
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Settings:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parent.parent
        models = Path(os.environ.get("MODELS_DIR", root / "models")).resolve()
        self.models_dir = models
        self.device = detect_device()

        bubble_weight = os.environ.get(
            "BUBBLE_WEIGHT", "models/bubble/yolov8n-seg-bubble-v2.pt"
        )
        bp = Path(bubble_weight)
        self.bubble_weight = bp if bp.is_absolute() else (root / bp).resolve()
        self.bubble_conf = float(os.environ.get("BUBBLE_CONF", "0.65"))
        self.bubble_conf_soft_hi = float(os.environ.get("BUBBLE_CONF_SOFT_HI", "0.80"))
        self.bubble_imgsz = int(os.environ.get("BUBBLE_IMGSZ", "640"))

        # Detect-stage ink filter (structure: glyph/punctuation CC presence).
        # Locked from fixtures/out/ink_measure.csv on test manga (v2 @ conf=0.65).
        # Empty FP: dark_px=0; ellipsis-only: max_cc≈16 / dark_px≈87 → floors below.
        self.ink_tol = float(os.environ.get("INK_TOL", "40"))
        self.ink_min_px = int(os.environ.get("INK_MIN_PX", "50"))
        self.ink_min_cc_area = int(os.environ.get("INK_MIN_CC_AREA", "12"))

        # 3px left outline ink; 8 cleared it (v11); +2 keeps fill off art that clips bubbles.
        self.clean_inset = int(os.environ.get("CLEAN_INSET", "10"))
        self.clean_ring = int(os.environ.get("CLEAN_RING", "6"))
        self.clean_uniform_tol = float(os.environ.get("CLEAN_UNIFORM_TOL", "18"))
        self.clean_uniform_frac = float(os.environ.get("CLEAN_UNIFORM_FRAC", "0.85"))
        self.clean_min_ring_px = int(os.environ.get("CLEAN_MIN_RING_PX", "30"))

        self.verbose = os.environ.get("VERBOSE", "").lower() in ("1", "true", "yes")
        os.environ.setdefault("MODELS_DIR", str(self.models_dir))


@lru_cache
def get_settings() -> Settings:
    return Settings()
