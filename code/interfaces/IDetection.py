from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .IFrameInput import RgbdFrame


@dataclass
class DetectedObject:
    label: str
    score: float
    bbox_2d: np.ndarray
    mask: np.ndarray | None = None


@dataclass
class DetectionResult:
    detections: list[DetectedObject]


class ObjectDetector(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def detect(
        self,
        frame: RgbdFrame,
        prompt: str,
        context_labels: list[str] | None = None,
    ) -> DetectionResult:
        pass
