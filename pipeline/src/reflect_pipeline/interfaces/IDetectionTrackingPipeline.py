from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from reflect_pipeline.interfaces.IDepthExtraction import DepthResult
from reflect_pipeline.interfaces.IDetection import DetectionResult
from reflect_pipeline.interfaces.IFrameInput import RgbdFrame
from reflect_pipeline.interfaces.ISceneGraphBuilder import SceneGraphBuildResult
from reflect_pipeline.interfaces.ISceneGraphSummarizer import SceneSummary
from reflect_pipeline.interfaces.ITracking import TrackingResult
from reflect_pipeline.interfaces.ITrackingValidator import ValidationResult


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
