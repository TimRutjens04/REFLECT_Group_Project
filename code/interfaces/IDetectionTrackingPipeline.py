from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .IDepthExtraction import DepthResult
from .IDetection import DetectionResult
from .IFrameInput import RgbdFrame
from .ISceneGraphBuilder import SceneGraphBuildResult
from .ISceneGraphSummarizer import SceneSummary
from .ITracking import TrackingResult
from .ITrackingValidator import ValidationResult


@dataclass
class PipelineResult:
    detection_result: DetectionResult
    depth_result: DepthResult
    tracking_result: TrackingResult | None
    validation_result: ValidationResult | None
    scene_graph_result: SceneGraphBuildResult
    scene_summary: SceneSummary


class DetectionTrackingPipeline(ABC):
    @abstractmethod
    def process_frame(
        self,
        frame: RgbdFrame,
        prompt: str,
    ) -> PipelineResult:
        pass
