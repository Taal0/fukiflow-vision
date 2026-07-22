"""YOLO-seg speech-bubble detector (ultralytics)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from vision.types import Bubble

# Module-level singleton set by app lifespan
_detector: Optional["BubbleDetector"] = None


def set_detector(detector: Optional["BubbleDetector"]) -> None:
    global _detector
    _detector = detector


def get_detector() -> "BubbleDetector":
    if _detector is None:
        raise RuntimeError("BubbleDetector not initialized")
    return _detector


class BubbleDetector:
    """Load ultralytics YOLO-seg once; predict → list[Bubble]."""

    def __init__(
        self,
        weight: Union[str, Path],
        conf: float = 0.5,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        self.weight = Path(weight)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._model = None
        self.ready = False

    def load(self) -> None:
        if not self.weight.is_file():
            raise FileNotFoundError(f"Bubble weight not found: {self.weight}")
        from ultralytics import YOLO

        self._model = YOLO(str(self.weight))
        self.ready = True

    def predict(self, img_rgb: np.ndarray) -> List[Bubble]:
        if self._model is None:
            raise RuntimeError("BubbleDetector.load() not called")

        # Ultralytics / YOLO train+eval load via OpenCV (BGR). Match that.
        import cv2

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        results = self._model.predict(
            source=img_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        masks = result.masks
        bubbles: List[Bubble] = []
        n = len(boxes)
        for i in range(n):
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
            conf = float(boxes.conf[i].cpu().item()) if boxes.conf is not None else 0.0
            cls = int(boxes.cls[i].cpu().item()) if boxes.cls is not None else 0

            polygon: list[list[float]]
            if masks is not None and masks.xy is not None and i < len(masks.xy):
                pts = masks.xy[i]
                if pts is not None and len(pts) >= 3:
                    polygon = [[float(x), float(y)] for x, y in pts]
                else:
                    polygon = _bbox_polygon(x1, y1, x2, y2)
            else:
                polygon = _bbox_polygon(x1, y1, x2, y2)

            bubbles.append(
                Bubble(
                    id=i,
                    polygon=polygon,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    cls=cls,
                )
            )
        return bubbles


def _bbox_polygon(x1: int, y1: int, x2: int, y2: int) -> list[list[float]]:
    return [
        [float(x1), float(y1)],
        [float(x2), float(y1)],
        [float(x2), float(y2)],
        [float(x1), float(y2)],
    ]
