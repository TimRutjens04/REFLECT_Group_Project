from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from interfaces.IFrameInput import RgbdFrame
from interfaces.ITracking import TrackingResult


@dataclass
class ValidationResult:
    is_valid: bool
    confidence: float
    reason: str | None = None


class TrackingValidator(ABC):
    @abstractmethod
    def validate(
        self,
        frame: RgbdFrame,
        tracking_result: TrackingResult,
    ) -> ValidationResult:
        pass
