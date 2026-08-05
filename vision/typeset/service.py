"""Typeset orchestration: width profile → font-fit → layout JSON + escalation."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from vision.masks import as_bbox
from vision.types import Bubble, BubbleLayout, BubbleText, EditorRegion
from vision.typeset.catalog import resolve_font_path
from vision.typeset.font import FontFace
from vision.typeset.layout import (
    FitSettings,
    enrich_layout_export,
    fit_bubble,
    font_ladder,
)
from vision.typeset.profile import (
    WidthProfile,
    build_width_profile,
    rect_width_profile,
)
from vision.typeset.wrap import Hyphenator, pyphen_hyphenator

_service: Optional["TypesetService"] = None

# Per-bubble override: {font_px?, font_scale?, font?, frame?, leading?, align?}
BubbleOverride = Dict[str, Any]


def set_service(service: Optional["TypesetService"]) -> None:
    global _service
    _service = service


def get_service() -> "TypesetService":
    if _service is None:
        raise RuntimeError("TypesetService not initialized")
    return _service


@dataclass
class TypesetSettings:
    font_path: Path = Path("assets/fonts/BubbleSans-Regular.otf")
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
    dpi: float = 350.0  # PAGE_DPI; used for font_pt / leading_pt export fields


@dataclass
class TypesetResult:
    layouts: List[BubbleLayout]
    status: str  # ok | partial
    typeset_regions: List[EditorRegion] = field(default_factory=list)
    profiles: Dict[int, WidthProfile] = field(default_factory=dict)


def _text_map(texts: Sequence[BubbleText]) -> Dict[int, str]:
    return {int(t.bubble_id): (t.text or "") for t in texts}


def _normalize_overrides(
    overrides: Optional[Dict[Union[int, str], BubbleOverride]],
) -> Dict[int, BubbleOverride]:
    if not overrides:
        return {}
    out: Dict[int, BubbleOverride] = {}
    for k, v in overrides.items():
        if v is None:
            continue
        out[int(k)] = dict(v)
    return out


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

    def _font_for(self, font_key: Optional[str]) -> FontFace:
        if not font_key:
            return self.font
        path = resolve_font_path(font_key, default=self.settings.font_path)
        if path.resolve() == Path(self.settings.font_path).resolve():
            return self.font
        return _cached_font(str(path))

    def typeset(
        self,
        img_rgb: np.ndarray,
        bubbles: Sequence[Bubble],
        texts: Sequence[BubbleText],
        *,
        overrides: Optional[Dict[Union[int, str], BubbleOverride]] = None,
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
        by_override = _normalize_overrides(overrides)

        layouts: List[BubbleLayout] = []
        regions: List[EditorRegion] = []
        profiles: Dict[int, WidthProfile] = {}

        for b in bubbles:
            ov = by_override.get(int(b.id), {})
            frame_ov = ov.get("frame")

            # Cap margin so narrow bubbles don't get over-eroded.
            # Frame override uses the letterer's box; margin from that box.
            if (
                isinstance(frame_ov, (list, tuple))
                and len(frame_ov) >= 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in frame_ov[:4])
            ):
                fx0, fy0, fx1, fy1 = (float(v) for v in frame_ov[:4])
                bw = max(1, int(fx1 - fx0))
                bh = max(1, int(fy1 - fy0))
            else:
                bbox = as_bbox(b.bbox)
                bw = max(1, bbox[2] - bbox[0])
                bh = max(1, bbox[3] - bbox[1])
                frame_ov = None

            cap_px = max(0, int(min(bw, bh) * float(s.margin_cap_ratio)))
            margin_px = min(page_margin_px, cap_px)

            if frame_ov is not None:
                # Editor text box — fill the dashed rect; only a tiny safety inset.
                # (Polygon path still uses clean_inset + typographic margin.)
                profile = rect_width_profile(
                    (h, w),
                    frame_ov[:4],
                    margin_px=min(2, margin_px),
                )
            else:
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

            face = self._font_for(ov.get("font") if isinstance(ov.get("font"), str) else None)
            force_px = ov.get("font_px")
            scale = ov.get("font_scale")
            leading = ov.get("leading")
            align = ov.get("align")
            bubble_fit = FitSettings(
                leading=float(leading) if leading is not None else fit.leading,
                width_safety=fit.width_safety,
                max_iters=fit.max_iters,
                align=str(align) if align else fit.align,
            )
            lay = fit_bubble(
                str(text),
                profile,
                face,
                ladder,
                h,
                bubble_id=int(b.id),
                settings=bubble_fit,
                hyphenator=self._hyphenator,
                force_size_px=float(force_px) if force_px is not None else None,
                font_scale=float(scale) if scale is not None else None,
            )
            if lay.status == "ok":
                lay = enrich_layout_export(lay, face, dpi=float(s.dpi))
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


@lru_cache(maxsize=16)
def _cached_font(path: str) -> FontFace:
    return FontFace(path)
