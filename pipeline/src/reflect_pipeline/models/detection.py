from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .base import FrameBase


class TriggerReason(str, Enum):
    INIT = "init"
    TRACKER_LOW_CONFIDENCE = "tracker_low_confidence"
    DEPTH_JUMP = "depth_jump"
    BBOX_SIZE_CHANGE = "bbox_size_change"
    DRIFT = "drift"
    FRAME_COUNTER_K = "frame_counter_K"
    MANUAL = "manual"
    NONE = "none"


class DetectionFailureMode(str, Enum):
    NO_OBJECT = "no_object"
    LOW_CONFIDENCE = "low_confidence"
    MULTIPLE_AMBIGUOUS = "multiple_ambiguous"
    WRONG_CATEGORY = "wrong_category"


@dataclass
class Detection:
    object_id: str
    label: str
    bbox_xyxy: list[float]
    confidence: float
    is_selected: bool


@dataclass
class DetectionFrame(FrameBase):
    """
    One detection JSONL row per frame.
    """

    detector_ran: bool
    trigger_reason: TriggerReason
    prompts_used: list[str]
    detections: list[Detection]
    detection_success: bool
    failure_mode: Optional[DetectionFailureMode] = None
    runtime_ms: Optional[float] = None
