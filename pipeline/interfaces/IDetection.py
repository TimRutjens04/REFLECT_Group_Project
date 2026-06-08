from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from interfaces.IFrameInput import RgbdFrame


@dataclass
class DetectedObject:
    label: str
    score: float
    bbox_2d: np.ndarray
    mask: np.ndarray | None = None
    prompt: str | None = None
    alternatives: list["DetectedObject"] | None = None


@dataclass
class DetectionResult:
    detections: list[DetectedObject]
    success: bool = True
    failure_reason: str | None = None
    prompt_used: str | None = None


class ObjectDetector(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def detect(self, frame: RgbdFrame, prompt: str) -> DetectionResult: ...
