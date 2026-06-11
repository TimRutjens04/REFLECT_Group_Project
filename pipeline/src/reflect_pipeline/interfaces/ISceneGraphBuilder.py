from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reflect_pipeline.interfaces.IDepthExtraction import DepthResult
from reflect_pipeline.interfaces.IDetection import DetectionResult
from reflect_pipeline.interfaces.IFrameInput import RgbdFrame
from reflect_pipeline.interfaces.ITracking import TrackingResult


@dataclass
class SceneGraphBuildInput:
    frame: RgbdFrame
    detections: DetectionResult
    depth_result: DepthResult
    tracking_result: TrackingResult | None = None
    task_info: dict[str, Any] | None = None


@dataclass
class SceneGraphBuildState:
    total_points_dict: dict[str, np.ndarray] = field(default_factory=dict)
    bbox3d_dict: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGraphBuildResult:
    scene_graph: Any
    bbox2d_dict: dict[str, np.ndarray]
    bbox3d_dict: dict[str, Any]


class SceneGraphBuilder(ABC):
    @abstractmethod
    def build(
        self,
        build_input: SceneGraphBuildInput,
        state: SceneGraphBuildState,
    ) -> SceneGraphBuildResult:
        pass
