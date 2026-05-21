from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BenchmarkTracker(ABC):
    """
    Common interface for all benchmark trackers.

    The harness controls the GDINO detection cadence. At every call to step()
    it either passes a fresh GDINO bbox (at detection interval frames) or None
    (between detections). The tracker decides how to incorporate each.

    Returns:
        predicted bbox [x1,y1,x2,y2] or None if the object is lost
        current track_id (increments on each reinitialization)
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear all state. Called before each benchmark run."""

    @abstractmethod
    def step(
        self,
        frame_bgr: np.ndarray,
        gdino_bbox: np.ndarray | None,
        gdino_score: float | None,
    ) -> tuple[np.ndarray | None, int]:
        """Process one frame and return (bbox_xyxy | None, track_id)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short display name for results table."""

    @property
    @abstractmethod
    def id_switches(self) -> int:
        """Total number of track identity changes since last reset()."""
