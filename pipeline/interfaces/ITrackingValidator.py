from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from interfaces.IFrameInput import RgbdFrame
from interfaces.ITracking import TrackingResult


@dataclass
class ValidationResult:
    track_id: int | None = None
    is_valid: bool
    confidence: float
    reason: str | None = None
    area_ratio: float | None = None
    drift_px: float | None = None
    depth_delta: float | None = None
    frames_since_init: int | None = None


class TrackingValidator(ABC):
    @abstractmethod
    def validate(
        self,
        frame: RgbdFrame,
        tracking_result: TrackingResult,
    ) -> ValidationResult:
        pass
