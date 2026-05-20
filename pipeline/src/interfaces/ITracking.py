from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from interfaces.IDetection import DetectionResult
from interfaces.IFrameInput import RgbdFrame


@dataclass
class TrackedObject:
    track_id: int
    label: str
    bbox_2d: np.ndarray
    confidence: float


@dataclass
class TrackingResult:
    tracked_objects: list[TrackedObject]


class ObjectTracker(ABC):
    @abstractmethod
    def initialize(
        self,
        frame: RgbdFrame,
        detections: DetectionResult,
    ) -> None:
        pass

    @abstractmethod
    def track(
        self,
        frame: RgbdFrame,
    ) -> TrackingResult:
        pass
