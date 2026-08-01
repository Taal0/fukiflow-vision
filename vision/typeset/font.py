"""Font loading + measurement helpers. No metrics are hard-coded in the engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from PIL import ImageFont

FontPath = Union[str, Path]


@dataclass(frozen=True)
class FontMetrics:
    """Ascender/descender for one rendered size (from the font file, not baked constants)."""

    ascent: float
    descent: float

    @property
    def em_height(self) -> float:
        return self.ascent + self.descent


class FontFace:
    """TrueType face loaded from a path; size is chosen per call."""

    def __init__(self, path: FontPath) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"font not found: {self.path}")
        self.family = self.path.stem
        # Probe load so missing/corrupt files fail at init, not mid-typeset.
        ImageFont.truetype(str(self.path), size=12)

    def _font(self, size_px: float) -> ImageFont.FreeTypeFont:
        # PIL sizes are integer; keep subpixel intent via max(1, round).
        return ImageFont.truetype(str(self.path), size=max(1, int(round(size_px))))

    def measure(self, text: str, size_px: float) -> float:
        if not text:
            return 0.0
        return float(self._font(size_px).getlength(text))

    def metrics(self, size_px: float) -> FontMetrics:
        ascent, descent = self._font(size_px).getmetrics()
        return FontMetrics(ascent=float(ascent), descent=float(descent))

    def line_height(self, size_px: float, leading: float) -> float:
        m = self.metrics(size_px)
        return float(leading) * m.em_height
