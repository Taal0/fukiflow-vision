"""Pruned config for detection / OCR / inpainting only."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class InpaintPrecision(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"

    def __str__(self):
        return self.name


class Detector(str, Enum):
    default = "default"
    none = "none"


class Inpainter(str, Enum):
    default = "default"
    lama_large = "lama_large"
    none = "none"
    original = "original"


class Ocr(str, Enum):
    ocr48px = "48px"


class DetectorConfig(BaseModel):
    detector: Detector = Detector.default
    detection_size: int = 2048
    text_threshold: float = 0.5
    det_rotate: bool = False
    det_auto_rotate: bool = False
    det_invert: bool = False
    det_gamma_correct: bool = False
    box_threshold: float = 0.7
    unclip_ratio: float = 2.3


class InpainterConfig(BaseModel):
    inpainter: Inpainter = Inpainter.lama_large
    inpainting_size: int = 2048
    inpainting_precision: InpaintPrecision = InpaintPrecision.bf16


class OcrConfig(BaseModel):
    use_mocr_merge: bool = False
    ocr: Ocr = Ocr.ocr48px
    min_text_length: int = 0
    ignore_bubble: int = 0
    prob: float | None = None


class TranslatorConfig(BaseModel):
    """Stub kept for ModelWrapper / InfererModule compatibility."""
    pass


class Config(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    inpainter: InpainterConfig = InpainterConfig()
    ocr: OcrConfig = OcrConfig()
    kernel_size: int = 3
    mask_dilation_offset: int = 20
