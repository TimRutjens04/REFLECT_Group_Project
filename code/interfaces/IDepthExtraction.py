from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .IDetection import DetectionResult
from .IFrameInput import RgbdFrame


@dataclass
class DepthObjectResult:
    label: str
    point_cloud: np.ndarray
    bbox_3d: Any
    center_3d: np.ndarray


@dataclass
class DepthResult:
    objects: list[DepthObjectResult]


class DepthExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        frame: RgbdFrame,
        detections: DetectionResult,
    ) -> DepthResult:
        pass
