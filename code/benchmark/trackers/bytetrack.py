from __future__ import annotations

import warnings

import numpy as np
import supervision as sv
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from supervision.tracker.byte_tracker.core import ByteTrack

from .base import BenchmarkTracker


class ByteTrackTracker(BenchmarkTracker):
    """
    ByteTrack via the supervision library.

    ByteTrack uses Kalman + IoU association internally. We feed it GDINO
    detections at the configured detection interval; between detections the
    Kalman filter propagates the track. Since we track a single object, we
    always return the track with the highest IoU to the last known position.
    """

    def __init__(self, cfg: dict) -> None:
        bt = cfg["bytetrack"]
        self._bt_cfg = bt
        self._tracker = self._make_tracker()
        self._track_id = 0
        self._id_switches = 0
        self._target_sv_id: int | None = None
        self._last_bbox: np.ndarray | None = None

    def _make_tracker(self) -> ByteTrack:
        bt = self._bt_cfg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return ByteTrack(
                track_activation_threshold=bt["track_activation_threshold"],
                lost_track_buffer=bt["lost_track_buffer"],
                minimum_matching_threshold=bt["minimum_matching_threshold"],
            )

    @property
    def name(self) -> str:
        return "ByteTrack"

    @property
    def id_switches(self) -> int:
        return self._id_switches

    def reset(self) -> None:
        self._tracker = self._make_tracker()
        self._track_id = 0
        self._id_switches = 0
        self._target_sv_id = None
        self._last_bbox = None

    def step(
        self,
        frame_bgr: np.ndarray,
        gdino_bbox: np.ndarray | None,
        gdino_score: float | None,
    ) -> tuple[np.ndarray | None, int]:
        if gdino_bbox is not None:
            boxes = gdino_bbox[np.newaxis]           # (1, 4)
            scores = np.array([gdino_score or 1.0])
            class_ids = np.zeros(1, dtype=int)
            sv_det = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)
        else:
            sv_det = sv.Detections.empty()

        tracked = self._tracker.update_with_detections(sv_det)

        if len(tracked) == 0 or tracked.tracker_id is None:
            # Track went to lost state — ByteTrack drops it from output but its
            # Kalman filter still predicts. Read predicted position from lost_tracks.
            if self._target_sv_id is not None:
                for strack in self._tracker.lost_tracks:
                    if strack.external_track_id == self._target_sv_id:
                        self._last_bbox = np.array(strack.tlbr, dtype=np.float32)
                        break
            return self._last_bbox, self._track_id

        chosen_idx = self._pick(tracked)
        if chosen_idx is None:
            return self._last_bbox, self._track_id

        sv_id = int(tracked.tracker_id[chosen_idx])
        if sv_id != self._target_sv_id:
            if self._target_sv_id is not None:
                self._id_switches += 1
            self._target_sv_id = sv_id
            self._track_id += 1

        bbox = tracked.xyxy[chosen_idx].astype(np.float32)
        self._last_bbox = bbox
        return bbox, self._track_id

    def _pick(self, tracked: sv.Detections) -> int | None:
        if tracked.tracker_id is None or len(tracked) == 0:
            return None
        if self._target_sv_id is not None:
            ids = list(tracked.tracker_id)
            if self._target_sv_id in ids:
                return ids.index(self._target_sv_id)
        if self._last_bbox is not None:
            ious = np.array([_iou(self._last_bbox, b) for b in tracked.xyxy])
            best = int(np.argmax(ious))
            return best if ious[best] > 0.0 else 0
        return 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / (ua + 1e-6))
