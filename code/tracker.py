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

    def reinitialize(self, frame: RgbdFrame, detections: DetectionResult) -> None:
        """Reseed CSRT without bumping track_id — same object re-acquired after occlusion."""
        if not detections.detections:
            return
        best = detections.detections[0]
        frame_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        x1, y1, x2, y2 = best.bbox_2d
        w = max(float(x2 - x1), 1.0)
        h = max(float(y2 - y1), 1.0)
        self._tracker = cv2.legacy.TrackerCSRT_create()
        self._tracker.init(frame_bgr, (float(x1), float(y1), w, h))

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


class Sam2ObjectTracker(ObjectTracker):
    """
    SAM 2 offline tracker implementing ObjectTracker.

    Because SAM 2 requires all frames upfront, this tracker buffers frames
    during track() calls and processes the full episode lazily on finalize().

    Typical usage (offline episode):
        tracker.initialize(frame0, detections)
        for frame in frames[1:]:
            tracker.track(frame)          # buffers frame, returns last known result
        results = tracker.finalize()      # runs SAM 2, returns list[TrackingResult]
    """

    def __init__(self) -> None:
        from trackers.sam2_tracker import Sam2Tracker
        self._sam2 = Sam2Tracker()
        self._seed_bbox: np.ndarray | None = None
        self._label: str = ""
        self._frames: list[np.ndarray] = []
        self._cache: list[TrackingResult] = []
        self._cache_idx: int = 0

    def initialize(self, frame: RgbdFrame, detections: DetectionResult) -> None:
        if not detections.detections:
            return
        best = detections.detections[0]
        self._seed_bbox = best.bbox_2d.copy()
        self._label = best.label
        frame_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        self._frames = [frame_bgr]
        self._cache = []
        self._cache_idx = 0

    def track(self, frame: RgbdFrame) -> TrackingResult:
        frame_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        self._frames.append(frame_bgr)
        # Return from cache if finalize() was already called.
        if self._cache and self._cache_idx < len(self._cache):
            result = self._cache[self._cache_idx]
            self._cache_idx += 1
            return result
        # Not yet finalized — return last known (empty before first finalize).
        return self._cache[-1] if self._cache else TrackingResult(tracked_objects=[])

    def finalize(self) -> list[TrackingResult]:
        """Run SAM 2 on all buffered frames. Call after the last track() for the episode."""
        if self._seed_bbox is None or not self._frames:
            return []
        preds_per_obj, _ = self._sam2.run_video(self._frames, [self._seed_bbox])
        preds = preds_per_obj[0]  # single object
        self._cache = []
        for bbox in preds:
            if bbox is None:
                self._cache.append(TrackingResult(tracked_objects=[]))
            else:
                self._cache.append(TrackingResult(tracked_objects=[
                    TrackedObject(
                        track_id=1,
                        label=self._label,
                        bbox_2d=bbox,
                        confidence=1.0,
                    )
                ]))
        self._cache_idx = 0
        return self._cache


def create_tracker(cfg: dict) -> ObjectTracker:
    """Factory — reads cfg['tracker']['backend'] (default: 'csrt')."""
    backend = cfg.get("tracker", {}).get("backend", "csrt")
    if backend == "sam2":
        return Sam2ObjectTracker()
    return CsrtObjectTracker()
