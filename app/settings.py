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

        bubble_weight = os.environ.get("BUBBLE_WEIGHT", "models/bubble/yolov8n-seg-bubble-v1.pt")
        bp = Path(bubble_weight)
        self.bubble_weight = bp if bp.is_absolute() else (root / bp).resolve()
        self.bubble_conf = float(os.environ.get("BUBBLE_CONF", "0.5"))
        self.bubble_imgsz = int(os.environ.get("BUBBLE_IMGSZ", "640"))

        # Local dir if present; otherwise treat as Hugging Face hub id.
        ocr_raw = os.environ.get("OCR_MODEL", "models/ocr/manga-ocr-base")
        ocr_candidate = Path(ocr_raw) if Path(ocr_raw).is_absolute() else (root / ocr_raw)
        self.ocr_model = str(ocr_candidate.resolve()) if ocr_candidate.exists() else ocr_raw
        self.ocr_crop_pad = int(os.environ.get("OCR_CROP_PAD", "4"))

        self.verbose = os.environ.get("VERBOSE", "").lower() in ("1", "true", "yes")
        os.environ.setdefault("MODELS_DIR", str(self.models_dir))


@lru_cache
def get_settings() -> Settings:
    return Settings()
