"""Smoke test: YOLOE loads, accepts text prompts, and satisfies the
TrackedObject contract. Asserts the contract and id format, not specific
detections. Skips (not fails) if weights cannot be downloaded.
"""

from __future__ import annotations

import re

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("ultralytics")

from livescene.config import AppConfig
from livescene.device import pick_device
from livescene.scene_graph.models import TrackedObject


@pytest.fixture(scope="module")
def tracker():
    from livescene.detector import YoloeTracker

    try:
        t = YoloeTracker(AppConfig(), pick_device())
        t.set_prompts(["person", "bus"])
    except Exception as e:  # weight download / env failure -> skip, don't fake
        pytest.skip(f"YOLOE unavailable in this environment: {e}")
    return t


def _test_image():
    from ultralytics.utils import ASSETS

    path = ASSETS / "bus.jpg"
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip(f"ultralytics asset image not found: {path}")
    return img


def test_track_returns_tracked_objects(tracker):
    img = _test_image()
    objs = tracker.track(img, frame_idx=0)
    assert isinstance(objs, list)
    for obj in objs:
        assert isinstance(obj, TrackedObject)
        assert re.fullmatch(r"(person|bus)_\d+", obj.object_id), obj.object_id
        assert len(obj.bbox_xyxy) == 4
        x1, y1, x2, y2 = obj.bbox_xyxy
        assert x2 > x1 and y2 > y1
        assert 0.0 <= obj.tracker_confidence <= 1.0
        assert obj.bbox_area_ratio_to_init == pytest.approx(1.0)  # first frame


def test_second_frame_persists_and_fills_displacement(tracker):
    img = _test_image()
    objs = tracker.track(img, frame_idx=1)
    # Same image again: any object tracked across both frames has a
    # displacement (possibly ~0) instead of None.
    for obj in objs:
        if obj.frames_since_redetect > 0:
            assert obj.displacement_px is not None


def test_prompt_change_resets_track_state(tracker):
    tracker.set_prompts(["person"])
    assert tracker._reset_next
    assert tracker._init_area == {}
    objs = tracker.track(_test_image(), frame_idx=2)
    for obj in objs:
        assert obj.object_id.startswith("person_")


def test_tracking_frame_wrapper(tracker):
    frame = tracker.tracking_frame([], frame_idx=5, timestamp=0.5)
    assert frame.sequence_id == "live"
    assert frame.frame_id == 5
    assert frame.tracked_objects == []
