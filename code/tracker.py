from __future__ import annotations

import cv2
import numpy as np
from interfaces import (
    DetectionResult,
    ObjectTracker,
    RgbdFrame,
    TrackedObject,
    TrackingResult,
)


class CsrtObjectTracker(ObjectTracker):
    """
    Single-object CSRT tracker.
    initialize() seeds from the highest-scoring DetectedObject.
    track_id increments on each initialize() call so the validator can
    detect reinitialization and reset its per-track state.
    """

    def __init__(self) -> None:
        self._tracker: cv2.legacy.TrackerCSRT | None = None
        self._track_id: int = 0
        self._label: str = ""

    def initialize(self, frame: RgbdFrame, detections: DetectionResult) -> None:
        if not detections.detections:
            return
        best = detections.detections[0]  # sorted by score descending
        frame_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        x1, y1, x2, y2 = best.bbox_2d
        w = max(float(x2 - x1), 1.0)
        h = max(float(y2 - y1), 1.0)
        self._tracker = cv2.legacy.TrackerCSRT_create()
        self._tracker.init(frame_bgr, (float(x1), float(y1), w, h))
        self._track_id += 1
        self._label = best.label

    def track(self, frame: RgbdFrame) -> TrackingResult:
        if self._tracker is None:
            return TrackingResult(tracked_objects=[])
        frame_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        ok, (rx, ry, rw, rh) = self._tracker.update(frame_bgr)
        if not ok:
            return TrackingResult(tracked_objects=[])
        rw = max(rw, 1.0)
        rh = max(rh, 1.0)
        return TrackingResult(tracked_objects=[
            TrackedObject(
                track_id=self._track_id,
                label=self._label,
                bbox_2d=np.array([rx, ry, rx + rw, ry + rh], dtype=np.float32),
                confidence=1.0,  # raw CSRT result; validator adjusts this
            )
        ])
