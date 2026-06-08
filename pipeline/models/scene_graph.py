from __future__ import annotations

from dataclasses import dataclass

from .base import FrameBase


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


@dataclass
class SceneGraphEdge:
    from_object_id: str
    to_object_id: str
    relation: str
    distance_3d_m: float

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
        }


@dataclass
class SceneGraphFrame(FrameBase):
    """
    One scene graph JSONL row per frame.
    """

    near_distance_threshold_m: float
    nodes: list[SceneGraphNode]
    edges: list[SceneGraphEdge]
