"""Typeset orchestration: width profile → font-fit → layout JSON + escalation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from vision.masks import as_bbox
from vision.types import Bubble, BubbleLayout, BubbleText, EditorRegion
from vision.typeset.font import FontFace
from vision.typeset.layout import FitSettings, fit_bubble, font_ladder
from vision.typeset.profile import WidthProfile, build_width_profile
from vision.typeset.wrap import Hyphenator, pyphen_hyphenator

_service: Optional["TypesetService"] = None


def set_service(service: Optional["TypesetService"]) -> None:
    global _service
    _service = service


def get_service() -> "TypesetService":
    if _service is None:
        raise RuntimeError("TypesetService not initialized")
    return _service


@dataclass
class TypesetSettings:
    font_path: Path = Path("assets/fonts/NotoSans-Regular.ttf")
    hyphen_dic: Path = Path("assets/hyphen/hyph_tr_TR.dic")
    clean_inset: int = 10
    margin_ratio: float = 0.006  # typographic margin ≈ page_h * ratio
    # Per-balloon cap: margin_px ≤ min(bw, bh) × margin_cap_ratio.
    # Prevents over-erosion of narrow bubbles.
    margin_cap_ratio: float = 0.20
    font_min_ratio: float = 0.006
    font_max_ratio: float = 0.042
    font_steps: int = 16
    leading: float = 1.15
    width_safety: float = 0.98
    # Secondary CC ≥ this × primary → multi_lobe escalate.
    multi_lobe_min_area_ratio: float = 0.20
    hyphenate: bool = True
    hyphen_lang: str = "tr_TR"
    align: str = "center"


@dataclass
class TypesetResult:
    layouts: List[BubbleLayout]
    status: str  # ok | partial
    typeset_regions: List[EditorRegion] = field(default_factory=list)
    profiles: Dict[int, WidthProfile] = field(default_factory=dict)


def _text_map(texts: Sequence[BubbleText]) -> Dict[int, str]:
    return {int(t.bubble_id): (t.text or "") for t in texts}


class TypesetService:
    """Place Turkish text into cleaned bubble polygons; escalate failures."""

    def __init__(self, settings: Optional[TypesetSettings] = None) -> None:
        self.settings = settings or TypesetSettings()
        self._font: Optional[FontFace] = None
        self._hyphenator: Optional[Hyphenator] = None
        self.ready = False

    def load(self) -> None:
        self._font = FontFace(self.settings.font_path)
        if self.settings.hyphenate:
            try:
                dic = self.settings.hyphen_dic
                if dic and Path(dic).is_file():
                    self._hyphenator = pyphen_hyphenator(
                        self.settings.hyphen_lang, dic_path=dic
                    )
                else:
                    self._hyphenator = pyphen_hyphenator(self.settings.hyphen_lang)
            except Exception:
                # Missing dict / import → degrade to word-boundary-only wrap.
                self._hyphenator = None
        else:
            self._hyphenator = None
        self.ready = True

    @property
    def font(self) -> FontFace:
        if self._font is None:
            raise RuntimeError("TypesetService not loaded")
        return self._font

    def typeset(
        self,
        img_rgb: np.ndarray,
        bubbles: Sequence[Bubble],
        texts: Sequence[BubbleText],
    ) -> TypesetResult:
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB, got shape {img_rgb.shape}")

        h, w = img_rgb.shape[:2]
        s = self.settings
        page_margin_px = max(0, int(round(h * float(s.margin_ratio))))
        ladder = font_ladder(
            h,
            min_ratio=s.font_min_ratio,
            max_ratio=s.font_max_ratio,
            steps=s.font_steps,
        )
        fit = FitSettings(
            leading=s.leading,
            width_safety=s.width_safety,
            align=s.align,
        )
        by_text = _text_map(texts)

        layouts: List[BubbleLayout] = []
        regions: List[EditorRegion] = []
        profiles: Dict[int, WidthProfile] = {}

        for b in bubbles:
            # Cap margin so narrow bubbles don't get over-eroded.
            bbox = as_bbox(b.bbox)
            bw = max(1, bbox[2] - bbox[0])
            bh = max(1, bbox[3] - bbox[1])
            cap_px = max(0, int(min(bw, bh) * float(s.margin_cap_ratio)))
            margin_px = min(page_margin_px, cap_px)
            profile = build_width_profile(
                (h, w),
                b,
                clean_inset=s.clean_inset,
                margin_px=margin_px,
                multi_lobe_min_area_ratio=s.multi_lobe_min_area_ratio,
            )
            profiles[int(b.id)] = profile

            if (b.clean_status or "unknown") == "editor":
                lay = BubbleLayout(
                    bubble_id=int(b.id), status="editor", reason="clean_editor"
                )
                layouts.append(lay)
                regions.append(as_bbox(b.bbox))
                continue

            text = by_text.get(int(b.id))
            if text is None or not str(text).strip():
                lay = BubbleLayout(
                    bubble_id=int(b.id), status="editor", reason="no_text"
                )
                layouts.append(lay)
                regions.append(as_bbox(b.bbox))
                continue

            lay = fit_bubble(
                str(text),
                profile,
                self.font,
                ladder,
                h,
                bubble_id=int(b.id),
                settings=fit,
                hyphenator=self._hyphenator,
            )
            layouts.append(lay)
            if lay.status == "editor":
                regions.append(as_bbox(b.bbox))

        status = "partial" if regions else "ok"
        return TypesetResult(
            layouts=layouts,
            status=status,
            typeset_regions=regions,
            profiles=profiles,
        )
