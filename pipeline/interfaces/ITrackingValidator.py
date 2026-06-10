from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from interfaces.IFrameInput import RgbdFrame
from interfaces.ITracking import TrackingResult


@dataclass
class ObjectValidation:
    """Per-tracked-object validation outcome.

    ``flags`` holds the raw check names that fired for this object, drawn from
    ``{"area_change", "drift", "depth_jump", "timeout"}``.
    """

    track_id: int
    label: str | None = None
    is_valid: bool = True
    flags: list[str] = field(default_factory=list)
    area_ratio: float | None = None       # current_area / init_area
    drift_px: float | None = None         # centroid Euclidean distance from prev frame
    depth_delta: float | None = None      # |current_depth - prev_depth| in metres
    frames_since_init: int | None = None  # frames elapsed since last tracker reset


@dataclass
class ValidationResult:
    """Frame-level aggregate plus per-object breakdown.

    ``is_valid`` / ``reason`` summarise the whole frame (used to decide whether a
    re-detect should fire). ``objects`` carries the individual outcome for every
    tracked object so flags can be assigned per object rather than per frame.
    """

    is_valid: bool
    confidence: float
    reason: str | None = None
    objects: list[ObjectValidation] = field(default_factory=list)
    # Aggregate metrics (last object inspected) kept for overlay / back-compat.
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
