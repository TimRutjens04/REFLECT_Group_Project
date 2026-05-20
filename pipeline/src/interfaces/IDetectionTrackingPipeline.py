from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from interfaces.IDepthExtraction import DepthResult
from interfaces.IDetection import DetectionResult
from interfaces.IFrameInput import RgbdFrame
from interfaces.ISceneGraphBuilder import SceneGraphBuildResult
from interfaces.ISceneGraphSummarizer import SceneSummary
from interfaces.ITracking import TrackingResult
from interfaces.ITrackingValidator import ValidationResult


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
