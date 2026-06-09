"""
Unit tests for CompositeTrackingValidator — the composite recovery trigger.

Covers each of the four individually-testable failure checks:
  - area_change  : bbox area deviates too much from initialisation size
  - drift        : centroid moves too far in a single step
  - depth_jump   : mean depth inside bbox jumps more than threshold
  - timeout      : redetect_interval frames elapsed since last reinit

Also tests combinations, logging of triggering condition in reason string,
reset behaviour, and the no-objects edge case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from interfaces import RgbdFrame, TrackingResult, TrackedObject
from validator import CompositeTrackingValidator


# ── helpers ───────────────────────────────────────────────────────────────────

def _frame(depth: np.ndarray | None = None, hw: tuple[int, int] = (480, 640)) -> RgbdFrame:
    h, w = hw
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if depth is None:
        depth = np.zeros((h, w), dtype=np.float32)
    return RgbdFrame(rgb=rgb, depth=depth, step_idx=0, metadata={})


def _result(bbox: np.ndarray, track_id: int = 1, label: str = "obj") -> TrackingResult:
    return TrackingResult(tracked_objects=[
        TrackedObject(track_id=track_id, label=label,
                      bbox_2d=bbox.astype(np.float32), confidence=1.0)
    ])


def _bbox(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ── initialisation (first call always valid) ──────────────────────────────────

class TestInitialisation:
    def test_first_call_always_valid(self):
        v = CompositeTrackingValidator()
        r = v.validate(_frame(), _result(_bbox(10, 10, 110, 110)))
        assert r.is_valid
        assert r.reason is None

    def test_no_tracked_objects_is_invalid(self):
        v = CompositeTrackingValidator()
        r = v.validate(_frame(), TrackingResult(tracked_objects=[]))
        assert not r.is_valid
        assert r.reason == "no tracked objects"


# ── area_change check ─────────────────────────────────────────────────────────

class TestAreaChange:
    def test_area_increase_beyond_thresh_triggers(self):
        v = CompositeTrackingValidator(area_change_thresh=0.5)
        frame = _frame()
        v.validate(frame, _result(_bbox(0, 0, 100, 100)))   # init: area=10000
        # area=30000 → 200% change > 50% threshold
        r = v.validate(frame, _result(_bbox(0, 0, 100, 300)))
        assert not r.is_valid
        assert "area_change" in r.reason

    def test_area_decrease_beyond_thresh_triggers(self):
        v = CompositeTrackingValidator(area_change_thresh=0.5)
        frame = _frame()
        v.validate(frame, _result(_bbox(0, 0, 100, 100)))   # init: area=10000
        # area=1000 → 90% decrease > 50% threshold
        r = v.validate(frame, _result(_bbox(0, 0, 10, 100)))
        assert not r.is_valid
        assert "area_change" in r.reason

    def test_area_within_thresh_is_valid(self):
        v = CompositeTrackingValidator(area_change_thresh=0.5, drift_thresh_px=9999,
                                       redetect_interval=9999)
        frame = _frame()
        v.validate(frame, _result(_bbox(0, 0, 100, 100)))   # init: area=10000
        # area=12000 → 20% change < 50% threshold
        r = v.validate(frame, _result(_bbox(0, 0, 120, 100)))
        assert r.is_valid

    def test_reason_encodes_label(self):
        v = CompositeTrackingValidator(area_change_thresh=0.1)
        frame = _frame()
        v.validate(frame, _result(_bbox(0, 0, 100, 100), label="mug"))
        r = v.validate(frame, _result(_bbox(0, 0, 200, 200), label="mug"))
        assert "mug:area_change" in r.reason


# ── drift check ───────────────────────────────────────────────────────────────

class TestDrift:
    def test_large_centroid_jump_triggers(self):
        v = CompositeTrackingValidator(drift_thresh_px=30.0, area_change_thresh=9999,
                                       redetect_interval=9999)
        frame = _frame()
        v.validate(frame, _result(_bbox(0, 0, 100, 100)))     # centroid=(50,50)
        # centroid=(150,50) → drift=100px > 30px threshold
        r = v.validate(frame, _result(_bbox(100, 0, 200, 100)))
        assert not r.is_valid
        assert "drift" in r.reason

    def test_small_centroid_move_is_valid(self):
        v = CompositeTrackingValidator(drift_thresh_px=30.0, area_change_thresh=9999,
                                       redetect_interval=9999)
        frame = _frame()
        v.validate(frame, _result(_bbox(0, 0, 100, 100)))     # centroid=(50,50)
        # centroid=(60,50) → drift=10px < 30px threshold
        r = v.validate(frame, _result(_bbox(10, 0, 110, 100)))
        assert r.is_valid

    def test_diagonal_drift_uses_euclidean_distance(self):
        v = CompositeTrackingValidator(drift_thresh_px=30.0, area_change_thresh=9999,
                                       redetect_interval=9999)
        frame = _frame()
        v.validate(frame, _result(_bbox(0, 0, 100, 100)))   # centroid=(50,50)
        # centroid=(71,71) → euclidean ≈ 29.7px < 30px — should be valid
        r = v.validate(frame, _result(_bbox(21, 21, 121, 121)))
        assert r.is_valid

    def test_reason_encodes_label(self):
        v = CompositeTrackingValidator(drift_thresh_px=10.0)
        frame = _frame()
        v.validate(frame, _result(_bbox(0, 0, 50, 50), label="pot"))
        r = v.validate(frame, _result(_bbox(50, 50, 100, 100), label="pot"))
        assert "pot:drift" in r.reason


# ── depth_jump check ──────────────────────────────────────────────────────────

class TestDepthJump:
    def _depth_frame(self, value: float) -> RgbdFrame:
        depth = np.full((480, 640), value, dtype=np.float32)
        return _frame(depth=depth)

    def test_depth_jump_beyond_thresh_triggers(self):
        v = CompositeTrackingValidator(depth_jump_thresh=0.3, area_change_thresh=9999,
                                       drift_thresh_px=9999, redetect_interval=9999)
        bbox = _bbox(0, 0, 100, 100)
        v.validate(self._depth_frame(1.0), _result(bbox))   # init depth=1.0m
        r = v.validate(self._depth_frame(1.5), _result(bbox))  # jump=0.5m > 0.3m
        assert not r.is_valid
        assert "depth_jump" in r.reason

    def test_depth_within_thresh_is_valid(self):
        v = CompositeTrackingValidator(depth_jump_thresh=0.3, area_change_thresh=9999,
                                       drift_thresh_px=9999, redetect_interval=9999)
        bbox = _bbox(0, 0, 100, 100)
        v.validate(self._depth_frame(1.0), _result(bbox))
        r = v.validate(self._depth_frame(1.1), _result(bbox))  # jump=0.1m < 0.3m
        assert r.is_valid

    def test_zero_depth_array_skips_check(self):
        v = CompositeTrackingValidator(depth_jump_thresh=0.3, area_change_thresh=9999,
                                       drift_thresh_px=9999, redetect_interval=9999)
        bbox = _bbox(0, 0, 100, 100)
        v.validate(_frame(), _result(bbox))
        r = v.validate(_frame(), _result(bbox))
        assert r.is_valid


# ── timeout check ─────────────────────────────────────────────────────────────

class TestTimeout:
    def test_timeout_triggers_after_interval(self):
        v = CompositeTrackingValidator(redetect_interval=3, area_change_thresh=9999,
                                       drift_thresh_px=9999, depth_jump_thresh=9999)
        frame = _frame()
        bbox = _bbox(0, 0, 100, 100)
        v.validate(frame, _result(bbox))   # init (frame 0)
        v.validate(frame, _result(bbox))   # frame 1
        v.validate(frame, _result(bbox))   # frame 2
        r = v.validate(frame, _result(bbox))  # frame 3 → frames_since_init==3 → timeout
        assert not r.is_valid
        assert "timeout" in r.reason

    def test_timeout_resets_counter(self):
        v = CompositeTrackingValidator(redetect_interval=2, area_change_thresh=9999,
                                       drift_thresh_px=9999, depth_jump_thresh=9999)
        frame = _frame()
        bbox = _bbox(0, 0, 100, 100)
        v.validate(frame, _result(bbox))   # init
        v.validate(frame, _result(bbox))   # frame 1
        v.validate(frame, _result(bbox))   # frame 2 → timeout fires, counter resets
        r = v.validate(frame, _result(bbox))  # frame 1 again after reset — no timeout
        assert r.is_valid

    def test_no_timeout_before_interval(self):
        v = CompositeTrackingValidator(redetect_interval=5, area_change_thresh=9999,
                                       drift_thresh_px=9999, depth_jump_thresh=9999)
        frame = _frame()
        bbox = _bbox(0, 0, 100, 100)
        v.validate(frame, _result(bbox))
        for _ in range(3):
            r = v.validate(frame, _result(bbox))
        assert r.is_valid


# ── reset behaviour ───────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_state_so_next_call_is_init(self):
        v = CompositeTrackingValidator(area_change_thresh=0.1)
        frame = _frame()
        v.validate(frame, _result(_bbox(0, 0, 100, 100)))
        v.reset(1)
        # After reset the next call re-initialises — large bbox must not fire
        r = v.validate(frame, _result(_bbox(0, 0, 500, 500)))
        assert r.is_valid

    def test_reset_unknown_track_id_is_safe(self):
        v = CompositeTrackingValidator()
        v.reset(999)  # must not raise


# ── multiple flags ─────────────────────────────────────────────────────────────

class TestMultipleFlags:
    def test_area_and_drift_both_appear_in_reason(self):
        v = CompositeTrackingValidator(area_change_thresh=0.1, drift_thresh_px=5.0,
                                       redetect_interval=9999, depth_jump_thresh=9999)
        frame = _frame()
        v.validate(frame, _result(_bbox(0, 0, 100, 100)))
        r = v.validate(frame, _result(_bbox(200, 200, 500, 500)))
        assert not r.is_valid
        assert "area_change" in r.reason
        assert "drift" in r.reason

    def test_confidence_decreases_with_more_flags(self):
        v = CompositeTrackingValidator(area_change_thresh=0.1, drift_thresh_px=5.0,
                                       redetect_interval=9999, depth_jump_thresh=9999)
        frame = _frame()
        v.validate(frame, _result(_bbox(0, 0, 100, 100)))
        r_multi = v.validate(frame, _result(_bbox(200, 200, 500, 500)))

        v2 = CompositeTrackingValidator(area_change_thresh=0.1, drift_thresh_px=9999,
                                        redetect_interval=9999, depth_jump_thresh=9999)
        frame2 = _frame()
        v2.validate(frame2, _result(_bbox(0, 0, 100, 100)))
        r_single = v2.validate(frame2, _result(_bbox(0, 0, 500, 500)))

        assert r_multi.confidence < r_single.confidence
