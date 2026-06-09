from __future__ import annotations

import cv2
import numpy as np

from .base import BenchmarkTracker


class CsrtTracker(BenchmarkTracker):
    """
    Baseline: OpenCV CSRT with update-freeze on failure.

    - Seeds from the first GDINO bbox.
    - At each detection interval: if GDINO finds the object, reinitialize.
    - Between detections: CSRT tracks freely.
    - On CSRT failure: freeze the model (stop calling update) and hold the
      last valid bbox until the next GDINO detection recovers the object.
    """

    def __init__(self) -> None:
        self._tracker: cv2.legacy.TrackerCSRT | None = None
        self._track_id = 0
        self._id_switches = 0
        self._frozen = False
        self._last_bbox: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "CSRT"

    @property
    def id_switches(self) -> int:
        return self._id_switches

    def reset(self) -> None:
        self._tracker = None
        self._track_id = 0
        self._id_switches = 0
        self._frozen = False
        self._last_bbox = None

    def _init(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> None:
        x1, y1, x2, y2 = bbox
        w = max(float(x2 - x1), 1.0)
        h = max(float(y2 - y1), 1.0)
        self._tracker = cv2.legacy.TrackerCSRT_create()
        self._tracker.init(frame_bgr, (float(x1), float(y1), w, h))
        self._last_bbox = bbox.copy()
        self._frozen = False

    def step(
        self,
        frame_bgr: np.ndarray,
        gdino_bbox: np.ndarray | None,
        gdino_score: float | None,
    ) -> tuple[np.ndarray | None, int]:
        if gdino_bbox is not None:
            if self._tracker is not None:
                self._id_switches += 1
            self._track_id += 1
            self._init(frame_bgr, gdino_bbox)
            return self._last_bbox, self._track_id

        if self._tracker is None:
            return None, self._track_id

        if self._frozen:
            return self._last_bbox, self._track_id

        ok, (rx, ry, rw, rh) = self._tracker.update(frame_bgr)
        if not ok or rw <= 0 or rh <= 0:
            self._frozen = True
            return self._last_bbox, self._track_id

        bbox = np.array([rx, ry, rx + rw, ry + rh], dtype=np.float32)
        self._last_bbox = bbox
        return bbox, self._track_id
