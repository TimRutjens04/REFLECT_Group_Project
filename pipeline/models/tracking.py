from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .base import FrameBase


class TrackerStatus(str, Enum):
    OK = "ok"
    DRIFTING = "drifting"
    LOST = "lost"
    RECOVERED = "recovered"
    OCCLUDED = "occluded"


@dataclass
class TrackedObject:
    object_id: str
    bbox_xyxy: list[float]
    bbox_area_px: float
    bbox_area_ratio_to_init: float
    center_xy: list[float]
    displacement_px: Optional[float]
    tracker_confidence: float
    tracker_status: TrackerStatus
    frames_since_redetect: int


@dataclass
class TrackingFlags:
    bbox_size_change_flag: bool
    drift_flag: bool
    any_recovery_trigger: bool


@dataclass
class TrackingFrame(FrameBase):
    """
    One tracking JSONL row per frame.
    """

    tracked_objects: list[TrackedObject]
    flags: TrackingFlags
