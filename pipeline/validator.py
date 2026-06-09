from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.ITracking import TrackingResult
from interfaces.ITrackingValidator import TrackingValidator, ValidationResult
from interfaces.IFrameInput import RgbdFrame


@dataclass
class _TrackState:
    init_area: float
    prev_center: tuple[float, float]
    prev_depth_mean: float | None
    frames_since_init: int = 0
    init_depth_norm_area: float | None = None  # pixel_area * depth² at seed (metres)


class CompositeTrackingValidator(TrackingValidator):
    """
    Stateful validator implementing four failure checks per tracked object:
      - area_change : bbox area deviates > area_change_thresh from initialisation
      - drift       : centroid moves > drift_thresh_px in a single step
      - depth_jump  : mean depth inside bbox jumps > depth_jump_thresh metres
      - timeout     : redetect_interval frames elapsed since last reinit

    Call reset(track_id) after the tracker reinitialises to clear stale state.
    """

    def __init__(
        self,
        area_change_thresh: float = 0.50,
        drift_thresh_px: float = 30.0,
        depth_jump_thresh: float = 10,
        redetect_interval: int = 30,
        area_check_grace_frames: int = 5,
    ) -> None:
        self._area_thresh = area_change_thresh
        self._drift_thresh = drift_thresh_px
        self._depth_thresh = depth_jump_thresh
        self._redet_int = redetect_interval
        self._area_grace = area_check_grace_frames
        self._state: dict[int, _TrackState] = {}

    def reset(self, track_id: int) -> None:
        """Clear stored state for track_id. Call after tracker.initialize()."""
        self._state.pop(track_id, None)

    def validate(
        self,
        frame: RgbdFrame,
        tracking_result: TrackingResult,
    ) -> ValidationResult:
        if not tracking_result.tracked_objects:
            return ValidationResult(
                is_valid=False, confidence=0.0, reason="no tracked objects"
            )

        reasons: list[str] = []
        min_conf = 1.0

        last_area_ratio: float | None = None
        last_drift_px: float | None = None
        last_depth_delta: float | None = None
        last_age: int | None = None

        for obj in tracking_result.tracked_objects:
            x1, y1, x2, y2 = obj.bbox_2d
            area = max((x2 - x1) * (y2 - y1), 0.0)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            depth = self._mean_depth_in_box(obj.bbox_2d, frame.depth)

            state = self._state.get(obj.track_id)
            if state is None:
                init_depth_norm = area * depth**2 if (depth and depth > 0) else None
                self._state[obj.track_id] = _TrackState(
                    init_area=area,
                    prev_center=(cx, cy),
                    prev_depth_mean=depth,
                    init_depth_norm_area=init_depth_norm,
                )
                continue

            state.frames_since_init += 1
            flags: list[str] = []

            # Area check — depth-normalised when metric depth is available.
            # Skipped for the first area_check_grace_frames after (re-)init so
            # CSRT has time to settle before we compare against the seed bbox.
            if state.frames_since_init > self._area_grace:
                if state.init_depth_norm_area is not None and depth and depth > 0:
                    cur_norm = area * depth**2
                    rel_change = abs(cur_norm - state.init_depth_norm_area) / (
                        state.init_depth_norm_area + 1e-6
                    )
                    last_area_ratio = cur_norm / (state.init_depth_norm_area + 1e-6)
                else:
                    rel_change = abs(area - state.init_area) / (state.init_area + 1e-6)
                    last_area_ratio = area / (state.init_area + 1e-6)

                if rel_change > self._area_thresh:
                    flags.append("area_change")
            else:
                # Inside grace period — report ratio for overlay but don't flag
                if state.init_depth_norm_area is not None and depth and depth > 0:
                    last_area_ratio = (area * depth**2) / (
                        state.init_depth_norm_area + 1e-6
                    )
                else:
                    last_area_ratio = area / (state.init_area + 1e-6)

            drift = (
                (cx - state.prev_center[0]) ** 2 + (cy - state.prev_center[1]) ** 2
            ) ** 0.5
            last_drift_px = float(drift)
            if drift > self._drift_thresh:
                flags.append("drift")

            if depth is not None and state.prev_depth_mean is not None:
                delta = abs(depth - state.prev_depth_mean)
                last_depth_delta = delta
                if delta > self._depth_thresh:
                    flags.append("depth_jump")

            if state.frames_since_init >= self._redet_int:
                flags.append("timeout")
                state.frames_since_init = 0

            last_age = state.frames_since_init

            state.prev_center = (cx, cy)
            state.prev_depth_mean = depth

            if flags:
                reasons.extend(f"{obj.label}:{f}" for f in flags)
                min_conf = min(min_conf, max(0.0, 1.0 - 0.25 * len(flags)))

        return ValidationResult(
            is_valid=len(reasons) == 0,
            confidence=min_conf,
            reason="; ".join(reasons) if reasons else None,
            area_ratio=last_area_ratio,
            drift_px=last_drift_px,
            depth_delta=last_depth_delta,
            frames_since_init=last_age,
        )

    @staticmethod
    def _mean_depth_in_box(bbox: np.ndarray, depth: np.ndarray) -> float | None:
        if depth is None:
            return None
        x1, y1, x2, y2 = bbox
        h, w = depth.shape[:2]
        xi1 = int(max(x1, 0))
        yi1 = int(max(y1, 0))
        xi2 = int(min(x2, w))
        yi2 = int(min(y2, h))
        patch = depth[yi1:yi2, xi1:xi2]
        if patch.size == 0:
            return None
        valid = patch[patch > 0]
        if valid.size == 0:
            return float(patch.mean())
        # 25th percentile = foreground depth, ignores background pixels in the box
        return float(np.percentile(valid, 25))
