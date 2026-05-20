from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RgbdFrame:
    rgb: np.ndarray
    depth: np.ndarray
    step_idx: int
    metadata: dict[str, Any] | None = None


class RgbdFrameProvider(ABC):
    @abstractmethod
    def get_frame(self, step_idx: int) -> RgbdFrame:
        pass


@dataclass
class DetectedObject:
    label: str
    score: float
    bbox_2d: np.ndarray
    mask: np.ndarray | None = None


@dataclass
class DetectionResult:
    detections: list[DetectedObject]


class ObjectDetector(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def detect(
        self,
        frame: RgbdFrame,
        prompt: str,
    ) -> DetectionResult:
        pass


@dataclass
class DepthObjectResult:
    label: str
    point_cloud: np.ndarray
    bbox_3d: Any
    center_3d: np.ndarray


@dataclass
class DepthResult:
    objects: list[DepthObjectResult]
    flags: list[str]
    trigger: bool
    depth_estimate: float


class DepthExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        frame: RgbdFrame,
        detections: DetectionResult,
    ) -> DepthResult:
        pass


@dataclass
class TrackedObject:
    track_id: int
    label: str
    bbox_2d: np.ndarray
    confidence: float


@dataclass
class TrackingResult:
    tracked_objects: list[TrackedObject]


class ObjectTracker(ABC):
    @abstractmethod
    def initialize(
        self,
        frame: RgbdFrame,
        detections: DetectionResult,
    ) -> None:
        pass

    @abstractmethod
    def track(
        self,
        frame: RgbdFrame,
    ) -> TrackingResult:
        pass


@dataclass
class ValidationResult:
    is_valid: bool
    confidence: float
    reason: str | None = None


class TrackingValidator(ABC):
    @abstractmethod
    def validate(
        self,
        frame: RgbdFrame,
        tracking_result: TrackingResult,
    ) -> ValidationResult:
        pass


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


@dataclass
class SceneGraphInput:
    nodes: list[str]
    edges: list[tuple[str, str, str]]
    metadata: dict[str, Any] | None = None


@dataclass
class SceneSummary:
    l1_summary: str
    l2_summary: str


class SceneGraphSummarizer(ABC):
    @abstractmethod
    def summarize(
        self,
        scene_graph: SceneGraphInput,
    ) -> SceneSummary:
        pass


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
