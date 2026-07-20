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
        self.detection_size = int(os.environ.get("DETECTION_SIZE", "2048"))
        self.text_threshold = float(os.environ.get("TEXT_THRESHOLD", "0.5"))
        self.box_threshold = float(os.environ.get("BOX_THRESHOLD", "0.7"))
        self.unclip_ratio = float(os.environ.get("UNCLIP_RATIO", "2.3"))
        self.inpainting_size = int(os.environ.get("INPAINTING_SIZE", "2048"))
        # lama_large | solidfill | none
        self.inpainter = os.environ.get("INPAINTER", "lama_large").strip().lower()
        self.solidfill_ring = int(os.environ.get("SOLIDFILL_RING", "6"))
        self.solidfill_uniform_tol = float(os.environ.get("SOLIDFILL_UNIFORM_TOL", "18"))
        self.solidfill_uniform_frac = float(os.environ.get("SOLIDFILL_UNIFORM_FRAC", "0.85"))
        self.solidfill_fill_dilate = int(os.environ.get("SOLIDFILL_FILL_DILATE", "2"))
        self.solidfill_min_ring_px = int(os.environ.get("SOLIDFILL_MIN_RING_PX", "30"))
        self.mask_dilation_offset = int(os.environ.get("MASK_DILATION_OFFSET", "20"))
        self.kernel_size = int(os.environ.get("KERNEL_SIZE", "3"))
        self.verbose = os.environ.get("VERBOSE", "").lower() in ("1", "true", "yes")
        # Inpaint post-process (small stain cleanup after LaMa); off until CTD trial.
        self.inpaint_postprocess = os.environ.get("INPAINT_POSTPROCESS", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        # Narrow scan ROI; expand fill on accepted stains (not the search window).
        self.postprocess_mask_dilate = int(os.environ.get("POSTPROCESS_MASK_DILATE", "2"))
        self.postprocess_gray_max = int(os.environ.get("POSTPROCESS_GRAY_MAX", "220"))
        self.postprocess_min_area = int(os.environ.get("POSTPROCESS_MIN_AREA", "3"))
        self.postprocess_max_area = int(os.environ.get("POSTPROCESS_MAX_AREA", "500"))
        self.postprocess_neighbor_median_min = int(
            os.environ.get("POSTPROCESS_NEIGHBOR_MEDIAN_MIN", "230")
        )
        self.postprocess_neighbor_ring = int(os.environ.get("POSTPROCESS_NEIGHBOR_RING", "4"))
        self.postprocess_paint_dilate = int(os.environ.get("POSTPROCESS_PAINT_DILATE", "8"))
        os.environ.setdefault("MODELS_DIR", str(self.models_dir))


@lru_cache
def get_settings() -> Settings:
    return Settings()
