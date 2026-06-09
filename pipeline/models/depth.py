from __future__ import annotations

from dataclasses import dataclass

from .base import FrameBase


@dataclass
class ObjectDepth:
    object_id: str
    depth_median_m: float
    depth_iqr_m: float
    valid_depth_pixel_ratio: float
    depth_jump_flag: bool
    depth_coherence_flag: bool
    depth_validity_flag: bool
    any_depth_trigger: bool


@dataclass
class DepthFrame(FrameBase):
    """
    One depth JSONL row per frame.
    """

    per_object_depth: list[ObjectDepth]
