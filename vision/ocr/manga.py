"""manga-ocr over bubble crops."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from vision.types import Bubble

_ocr: Optional["MangaOcrService"] = None


def set_ocr(service: Optional["MangaOcrService"]) -> None:
    global _ocr
    _ocr = service


def get_ocr() -> "MangaOcrService":
    if _ocr is None:
        raise RuntimeError("MangaOcrService not initialized")
    return _ocr


class MangaOcrService:
    """Singleton wrapper around kha-white/manga-ocr."""

    def __init__(
        self,
        model: str = "kha-white/manga-ocr-base",
        device: str = "cpu",
        crop_pad: int = 4,
    ) -> None:
        self.model_name = model
        self.device = device
        self.crop_pad = crop_pad
        self._mocr = None
        self.ready = False

    def load(self) -> None:
        from manga_ocr import MangaOcr

        # MangaOcr auto-picks cuda/mps unless force_cpu; map our device setting.
        force_cpu = self.device == "cpu"
        self._mocr = MangaOcr(
            pretrained_model_name_or_path=self.model_name,
            force_cpu=force_cpu,
        )
        self.ready = True

    def recognize(
        self,
        img_rgb: np.ndarray,
        bubbles: Sequence[Bubble],
        crop_pad: Optional[int] = None,
    ) -> List[Bubble]:
        if self._mocr is None:
            raise RuntimeError("MangaOcrService.load() not called")

        pad = self.crop_pad if crop_pad is None else crop_pad
        h, w = img_rgb.shape[:2]
        out: List[Bubble] = []
        for b in bubbles:
            crop = _crop_bbox(img_rgb, b.bbox, pad=pad, width=w, height=h)
            text = ""
            if crop.size > 0 and crop.shape[0] > 1 and crop.shape[1] > 1:
                pil = Image.fromarray(crop)
                text = self._mocr(pil) or ""
            out.append(
                b.model_copy(
                    update={
                        "text_ja": text,
                        "ocr_confidence": None,  # generation model; no score
                    }
                )
            )
        return out


def _crop_bbox(
    img_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    pad: int,
    width: int,
    height: int,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return img_rgb[y1:y2, x1:x2]
