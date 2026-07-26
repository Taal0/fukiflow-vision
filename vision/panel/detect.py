"""YOLO panel detector (ultralytics, bbox-only)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from vision.types import Panel

# Module-level singleton set by app lifespan
_detector: Optional["PanelDetector"] = None


def set_detector(detector: Optional["PanelDetector"]) -> None:
    global _detector
    _detector = detector


def get_detector() -> "PanelDetector":
    if _detector is None:
        raise RuntimeError("PanelDetector not initialized")
    return _detector


class PanelDetector:
    """Load ultralytics YOLO once; predict → list[Panel] (bbox only)."""

    def __init__(
        self,
        weight: Union[str, Path],
        conf: float = 0.40,
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
            raise FileNotFoundError(f"Panel weight not found: {self.weight}")
        from ultralytics import YOLO

        self._model = YOLO(str(self.weight))
        self.ready = True

    def predict(self, img_rgb: np.ndarray) -> List[Panel]:
        if self._model is None:
            raise RuntimeError("PanelDetector.load() not called")

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

        panels: List[Panel] = []
        n = len(boxes)
        for i in range(n):
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
            conf = float(boxes.conf[i].cpu().item()) if boxes.conf is not None else 0.0
            panels.append(
                Panel(
                    id=i,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                )
            )
        return panels
