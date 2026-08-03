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

        panel_weight = os.environ.get(
            "PANEL_WEIGHT", "models/panel/yolov8n-panel-v3.pt"
        )
        pp = Path(panel_weight)
        self.panel_weight = pp if pp.is_absolute() else (root / pp).resolve()
        self.panel_conf = float(os.environ.get("PANEL_CONF", "0.40"))
        self.panel_imgsz = int(os.environ.get("PANEL_IMGSZ", "640"))
        self.panel_nms_iou = float(os.environ.get("PANEL_NMS_IOU", "0.60"))

        self.order_tol_frac = float(os.environ.get("ORDER_TOL_FRAC", "0.02"))
        self.order_iob_contained = float(os.environ.get("ORDER_IOB_CONTAINED", "0.70"))
        self.order_iob_partial = float(os.environ.get("ORDER_IOB_PARTIAL", "0.15"))
        self.order_ambig_delta = float(os.environ.get("ORDER_AMBIG_DELTA", "0.05"))
        self.order_conf_min = float(os.environ.get("ORDER_CONF_MIN", "0.50"))

        # Detect-stage ink filter (structure: glyph/punctuation CC presence).
        # Locked from fixtures/out/ink_measure.csv on test manga (v2 @ conf=0.65).
        # Empty FP: dark_px=0; ellipsis-only: max_cc≈16 / dark_px≈87 → floors below.
        self.ink_tol = float(os.environ.get("INK_TOL", "40"))
        self.ink_min_px = int(os.environ.get("INK_MIN_PX", "50"))
        self.ink_min_cc_area = int(os.environ.get("INK_MIN_CC_AREA", "12"))

        # 3px left outline ink; 8 cleared it (v11); +2 keeps fill off art that clips bubbles.
        # Authoritative default is 10 (not the older "3 px" note).
        self.clean_inset = int(os.environ.get("CLEAN_INSET", "10"))
        self.clean_ring = int(os.environ.get("CLEAN_RING", "6"))
        self.clean_uniform_tol = float(os.environ.get("CLEAN_UNIFORM_TOL", "18"))
        self.clean_uniform_frac = float(os.environ.get("CLEAN_UNIFORM_FRAC", "0.85"))
        self.clean_min_ring_px = int(os.environ.get("CLEAN_MIN_RING_PX", "30"))

        # Typeset (layout JSON; page-height-relative ladder; Bubble Sans OFL)
        font_default = root / "assets" / "fonts" / "BubbleSans-Regular.otf"
        font_env = os.environ.get("TYPESET_FONT", str(font_default))
        fp = Path(font_env)
        self.typeset_font = fp if fp.is_absolute() else (root / fp).resolve()
        self.typeset_margin_ratio = float(os.environ.get("TYPESET_MARGIN_RATIO", "0.006"))
        # Per-balloon margin cap: margin_px ≤ min(balloon_w, balloon_h) × ratio.
        # Prevents over-erosion of narrow bubbles where page-height margin eats
        # into the available area. 0.20 keeps 60% of a 63px-wide balloon intact.
        self.typeset_margin_cap_ratio = float(
            os.environ.get("TYPESET_MARGIN_CAP_RATIO", "0.20")
        )
        # Floor low enough for narrow manga bubbles (~10px on 1686h pages).
        # Binary search still picks the largest size that fits.
        self.typeset_font_min_ratio = float(
            os.environ.get("TYPESET_FONT_MIN_RATIO", "0.006")
        )
        self.typeset_font_max_ratio = float(
            os.environ.get("TYPESET_FONT_MAX_RATIO", "0.042")
        )
        self.typeset_font_steps = int(os.environ.get("TYPESET_FONT_STEPS", "16"))
        self.typeset_leading = float(os.environ.get("TYPESET_LEADING", "1.15"))
        self.typeset_width_safety = float(
            os.environ.get("TYPESET_WIDTH_SAFETY", "0.98")
        )
        # Secondary CC must be ≥ this fraction of the primary to escalate as
        # multi_lobe. Speech-tail residue after erode is typically well below.
        self.typeset_multi_lobe_min_area_ratio = float(
            os.environ.get("TYPESET_MULTI_LOBE_MIN_AREA_RATIO", "0.20")
        )
        self.typeset_hyphenate = os.environ.get("TYPESET_HYPHENATE", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        self.typeset_hyphen_lang = os.environ.get("TYPESET_HYPHEN_LANG", "tr_TR")
        hyphen_default = root / "assets" / "hyphen" / "hyph_tr_TR.dic"
        hyphen_env = os.environ.get("TYPESET_HYPHEN_DIC", str(hyphen_default))
        hp = Path(hyphen_env)
        self.typeset_hyphen_dic = hp if hp.is_absolute() else (root / hp).resolve()

        # Page DPI for pt conversion on layout export fields (PSD/IDML).
        self.page_dpi = float(os.environ.get("PAGE_DPI", "350"))

        self.verbose = os.environ.get("VERBOSE", "").lower() in ("1", "true", "yes")
        # Skip YOLO bubble/panel load — typeset + compose only (cheap CPU demos).
        self.vision_light = os.environ.get("VISION_LIGHT", "").lower() in (
            "1",
            "true",
            "yes",
        )
        os.environ.setdefault("MODELS_DIR", str(self.models_dir))


@lru_cache
def get_settings() -> Settings:
    return Settings()
