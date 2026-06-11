from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .base import FrameBase


class NodeStatus(str, Enum):
    """Per-object reliability state, derived from tracking + depth signals."""

    OK = "ok"
    DRIFTING = "drifting"
    OCCLUDED = "occluded"
    LOST = "lost"
    RECOVERED = "recovered"
    UNCERTAIN = "uncertain"  # tracked but depth unreliable


class LocalizationFailureType(str, Enum):
    """Failure taxonomy surfaced to the REFLECT LLM (see CLAUDE.md)."""

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
