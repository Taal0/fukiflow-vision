from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from vision_engine.config import (
    Detector,
    DetectorConfig,
    InpaintPrecision,
    Inpainter,
    InpainterConfig,
    Ocr,
    OcrConfig,
)
from vision_engine.detection import dispatch as dispatch_detection
from vision_engine.detection import prepare as prepare_detection
from vision_engine.inpainting import dispatch as dispatch_inpainting
from vision_engine.inpainting import prepare as prepare_inpainting
from vision_engine.mask_refinement import dispatch as dispatch_mask_refinement
from vision_engine.ocr import dispatch as dispatch_ocr
from vision_engine.ocr import prepare as prepare_ocr
from vision_engine.textline_merge import dispatch as dispatch_textline_merge

from .postprocess import remove_inpaint_stains
from .schemas import Box
from .settings import Settings, get_settings
from .solidfill import solid_fill_clean


@dataclass
class DetectOcrResult:
    page_id: str
    boxes: List[Box]
    mask: np.ndarray
    width: int
    height: int


class VisionEngine:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def prepare(self) -> None:
        device = self.settings.device
        await prepare_detection(Detector.ctd)
        await prepare_ocr(Ocr.ocr48px, device)
        # solidfill needs no model; none is lightweight; only LaMa downloads weights.
        if self.settings.inpainter == "lama_large":
            await prepare_inpainting(Inpainter.lama_large, device)
        elif self.settings.inpainter == "none":
            await prepare_inpainting(Inpainter.none, device)
        elif self.settings.inpainter != "solidfill":
            raise ValueError(
                f"Unknown INPAINTER={self.settings.inpainter!r}; "
                "use lama_large | solidfill | none"
            )
        self._ready = True

    def _detector_cfg(self) -> DetectorConfig:
        s = self.settings
        return DetectorConfig(
            detector=Detector.ctd,
            detection_size=s.detection_size,
            text_threshold=s.text_threshold,
            box_threshold=s.box_threshold,
            unclip_ratio=s.unclip_ratio,
        )

    def _ocr_cfg(self) -> OcrConfig:
        return OcrConfig(ocr=Ocr.ocr48px)

    def _inpainter_cfg(self) -> InpainterConfig:
        # bf16 is unreliable on MPS; prefer fp32 there / on CPU.
        precision = InpaintPrecision.bf16
        if self.settings.device in ("cpu", "mps"):
            precision = InpaintPrecision.fp32
        key = (
            Inpainter.none
            if self.settings.inpainter == "none"
            else Inpainter.lama_large
        )
        return InpainterConfig(
            inpainter=key,
            inpainting_size=self.settings.inpainting_size,
            inpainting_precision=precision,
        )

    async def detect_ocr(self, img_rgb: np.ndarray, page_id: Optional[str] = None) -> DetectOcrResult:
        det = self._detector_cfg()
        device = self.settings.device
        verbose = self.settings.verbose

        textlines, mask_raw, mask = await dispatch_detection(
            Detector.ctd,
            img_rgb,
            det.detection_size,
            det.text_threshold,
            det.box_threshold,
            det.unclip_ratio,
            det.det_invert,
            det.det_gamma_correct,
            det.det_rotate,
            det.det_auto_rotate,
            device,
            verbose,
        )

        if textlines:
            textlines = await dispatch_ocr(
                Ocr.ocr48px,
                img_rgb,
                textlines,
                self._ocr_cfg(),
                device,
                verbose,
            )
            text_regions = await dispatch_textline_merge(
                textlines,
                img_rgb.shape[1],
                img_rgb.shape[0],
                verbose=verbose,
            )
        else:
            text_regions = []

        boxes: List[Box] = []
        for region in text_regions:
            lines = np.asarray(region.lines)
            if lines.ndim == 3 and len(lines) == 1:
                polygon = lines[0].tolist()
            elif lines.ndim == 3:
                polygon = [quad.tolist() for quad in lines]
            else:
                polygon = lines.reshape(-1, 2).tolist()

            x1, y1, x2, y2 = region.xyxy
            boxes.append(
                Box(
                    polygon=polygon,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    text_ja=region.text or "",
                    confidence=float(region.prob or 0.0),
                )
            )

        # Prefer a detector-provided refined mask (CTD). DBNet leaves this None and
        # needs fit_text on the soft raw mask — do not run that path on CTD or the
        # DBNet-tuned dilation_offset balloons masks outside bubbles.
        if mask is None and text_regions and mask_raw is not None:
            mask = await dispatch_mask_refinement(
                text_regions,
                img_rgb,
                mask_raw,
                "fit_text",
                self.settings.mask_dilation_offset,
                0,  # ignore_bubble off
                verbose,
                self.settings.kernel_size,
            )
        if mask is None:
            mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        elif mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        return DetectOcrResult(
            page_id=page_id or str(uuid.uuid4()),
            boxes=boxes,
            mask=mask,
            width=int(img_rgb.shape[1]),
            height=int(img_rgb.shape[0]),
        )

    async def inpaint(
        self,
        img_rgb: np.ndarray,
        mask: np.ndarray,
        boxes: Optional[List[Box]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        if mask.shape[:2] != img_rgb.shape[:2]:
            raise ValueError(
                f"mask shape {mask.shape[:2]} does not match image {img_rgb.shape[:2]}"
            )

        s = self.settings
        if s.inpainter == "solidfill":
            cleaned, editor_regions = solid_fill_clean(
                img_rgb,
                mask,
                boxes,
                ring=s.solidfill_ring,
                uniform_tol=s.solidfill_uniform_tol,
                uniform_frac=s.solidfill_uniform_frac,
                fill_dilate=s.solidfill_fill_dilate,
                min_ring_px=s.solidfill_min_ring_px,
            )
            return cleaned.astype(np.uint8), editor_regions

        cfg = self._inpainter_cfg()
        key = Inpainter.none if s.inpainter == "none" else Inpainter.lama_large
        cleaned = await dispatch_inpainting(
            key,
            img_rgb,
            mask,
            cfg,
            s.inpainting_size,
            s.device,
            s.verbose,
        )
        cleaned = cleaned.astype(np.uint8)
        if s.inpainter == "lama_large" and s.inpaint_postprocess:
            cleaned = remove_inpaint_stains(
                cleaned,
                mask,
                mask_dilate=s.postprocess_mask_dilate,
                gray_max=s.postprocess_gray_max,
                min_area=s.postprocess_min_area,
                max_area=s.postprocess_max_area,
                neighbor_median_min=s.postprocess_neighbor_median_min,
                neighbor_ring=s.postprocess_neighbor_ring,
                paint_dilate=s.postprocess_paint_dilate,
            )
        return cleaned, []


# Module-level singleton used by FastAPI lifespan
engine: Optional[VisionEngine] = None


def get_engine() -> VisionEngine:
    if engine is None:
        raise RuntimeError("VisionEngine not initialized")
    return engine
