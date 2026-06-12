"""Model dataclasses copied from the REFLECT pipeline (models/{base,tracking,detection,scene_graph}.py).

Merged into one module so the scene-graph builder is self-contained; the
original ``reflect_pipeline`` package is not a dependency of this project.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
@dataclass
class FrameBase:
    """
    Shared keys for every JSONL row.

    Every pipeline module should inherit from this so rows can be merged on:
    sequence_id, frame_id, timestamp.
    """

    sequence_id: str
    frame_id: int
    timestamp: float


def to_jsonable(value: Any) -> Any:
    """
    Convert dataclasses and enums into plain JSON-compatible values.
    """

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]

    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}

    return value


class JsonlWriter:
    """
    Append one dataclass row as one JSON object per line.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: FrameBase) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")

    def write_many(self, rows: list[FrameBase]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# tracking
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# detection (only what the builder signature needs; no detector runs here)
# --------------------------------------------------------------------------- #
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
    trigger_reason: str
    prompts_used: list[str]
    detections: list[Detection]
    detection_success: bool
    failure_mode: Optional[DetectionFailureMode] = None
    runtime_ms: Optional[float] = None


# --------------------------------------------------------------------------- #
# scene graph
# --------------------------------------------------------------------------- #
class NodeStatus(str, Enum):
    """Per-object reliability state, derived from tracking + depth signals."""

    OK = "ok"
    DRIFTING = "drifting"
    OCCLUDED = "occluded"
    LOST = "lost"
    RECOVERED = "recovered"
    UNCERTAIN = "uncertain"  # tracked but depth unreliable


class LocalizationFailureType(str, Enum):
    """Failure taxonomy surfaced to the REFLECT LLM."""

    WRONG_OBJECT = "Wrong_object"
    SLIP = "Slip"
    NO_GRASP = "No_Grasp"
    MISSING = "Missing"


@dataclass
class Position3D:
    x: float
    y: float
    z: float


@dataclass
class SceneGraphNode:
    object_id: str
    label: str
    pixel_center: list[float]
    depth_used_m: float
    position_3d: Position3D
    # --- enrichments carried from detection / tracking / depth ---
    bbox_xyxy: list[float] = field(default_factory=list)
    status: NodeStatus = NodeStatus.OK
    confidence: float = 1.0
    depth_validity_flag: bool = True
    depth_coherence_flag: bool = True
    depth_jump_flag: bool = False
    any_depth_trigger: bool = False


@dataclass
class SceneGraphEdge:
    from_object_id: str
    to_object_id: str
    relation: str
    distance_3d_m: float
    # --- enrichments: where the relation came from + how sure we are ---
    source: str = "3d_distance"
    confidence: float = 1.0

    def to_output_dict(self) -> dict:
        """
        Optional helper if you want the exact JSON keys 'from' and 'to'.
        Normal dataclass output uses from_object_id and to_object_id because
        'from' is a reserved Python keyword.
        """
        return {
            "from": self.from_object_id,
            "to": self.to_object_id,
            "relation": self.relation,
            "distance_3d_m": self.distance_3d_m,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass
class LocalizationFlag:
    """
    Explicit failure signal for the LLM, instead of leaving it to infer one
    from an ambiguous scene description.
    """

    failure_detected: bool
    type: LocalizationFailureType | None = None
    affected_object_id: str | None = None


@dataclass
class SceneGraphFrame(FrameBase):
    """
    One scene graph JSONL row per frame.
    """

    near_distance_threshold_m: float
    nodes: list[SceneGraphNode]
    edges: list[SceneGraphEdge]
    localization_flag: LocalizationFlag = field(
        default_factory=lambda: LocalizationFlag(failure_detected=False)
    )
