from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .IFrameInput import RgbdFrame
from .ITracking import TrackingResult


@dataclass
class ValidationResult:
    is_valid: bool
    confidence: float
    reason: str | None = None
    area_ratio: float | None = None       # current_area / init_area
    drift_px: float | None = None         # centroid Euclidean distance from prev frame
    depth_delta: float | None = None      # |current_depth - prev_depth| in metres
    frames_since_init: int | None = None  # frames elapsed since last tracker reset


class TrackingValidator(ABC):
    @abstractmethod
    def validate(
        self,
        frame: RgbdFrame,
        tracking_result: TrackingResult,
    ) -> ValidationResult:
        pass
