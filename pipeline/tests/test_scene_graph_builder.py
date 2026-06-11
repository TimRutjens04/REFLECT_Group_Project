"""
Deterministic tests for SceneGraphBuilder on synthetic tracking rows + depth.

Run from the pipeline/ directory (imports resolve `models` on sys.path):
    python3 tests/test_scene_graph_builder.py
or:
    python3 -m pytest tests/test_scene_graph_builder.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import JsonlWriter
from models.detection import (
    Detection,
    DetectionFailureMode,
    DetectionFrame,
    TriggerReason,
)
from models.scene_graph import LocalizationFailureType, NodeStatus
from models.tracking import TrackedObject, TrackerStatus, TrackingFlags, TrackingFrame
from scene_graph.scene_graph_builder import SceneGraphBuilder, SceneGraphConfig
from scene_graph.build_scene_graphs import assemble


def _obj(object_id, bbox, conf=0.9, ratio=1.0, disp=0.0):
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return TrackedObject(
        object_id=object_id,
        bbox_xyxy=list(bbox),
        bbox_area_px=area,
        bbox_area_ratio_to_init=ratio,
        center_xy=[cx, cy],
        displacement_px=disp,
        tracker_confidence=conf,
        tracker_status=TrackerStatus.OK,
        frames_since_redetect=0,
    )


def _tracking_frame(objs, frame_id=0):
    return TrackingFrame(
        sequence_id="seq",
        frame_id=frame_id,
        timestamp=float(frame_id),
        tracked_objects=objs,
        flags=TrackingFlags(False, False, False),
    )


def _depth_with(boxes_depths, shape=(300, 300)):
    """Background = 0 (invalid); each (bbox, depth_mm) filled solid."""
    depth = np.zeros(shape, dtype=np.float64)
    for bbox, mm in boxes_depths:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        depth[y1:y2, x1:x2] = mm
    return depth


A = [50, 50, 100, 100]
B = [110, 50, 160, 100]


def test_nodes_enriched_and_near_edge():
    builder = SceneGraphBuilder("seq")
    depth = _depth_with([(A, 1000.0), (B, 1000.0)])
    frame = builder.build(_tracking_frame([_obj("apple_1", A), _obj("bowl_2", B)]), depth)

    assert len(frame.nodes) == 2
    node = frame.nodes[0]
    assert node.bbox_xyxy == A
    assert node.confidence == 0.9
    assert node.depth_validity_flag is True
    assert node.status == NodeStatus.OK
    assert abs(node.depth_used_m - 1.0) < 1e-6  # 1000 mm auto-scaled to metres

    near = [e for e in frame.edges if e.relation == "near"]
    assert near, "expected a near edge between the two close objects"
    assert near[0].source == "3d_distance"
    assert near[0].confidence == 0.9


def test_depth_jump_marks_occluded():
    builder = SceneGraphBuilder("seq")
    builder.build(_tracking_frame([_obj("apple_1", A)], 0), _depth_with([(A, 1000.0)]))
    frame = builder.build(_tracking_frame([_obj("apple_1", A)], 1), _depth_with([(A, 1500.0)]))
    node = frame.nodes[0]
    assert node.depth_jump_flag is True
    assert node.any_depth_trigger is True
    assert node.status == NodeStatus.OCCLUDED


def test_invalid_depth_is_uncertain_and_no_edges():
    builder = SceneGraphBuilder("seq")
    # apple over background only -> no valid depth pixels.
    depth = _depth_with([(B, 1000.0)])  # only B has depth; A region is background
    frame = builder.build(_tracking_frame([_obj("apple_1", A), _obj("bowl_2", B)]), depth)
    apple = next(n for n in frame.nodes if n.object_id == "apple_1")
    assert apple.depth_validity_flag is False
    assert apple.status == NodeStatus.UNCERTAIN
    # an uncertain node must not silently produce 3D edges
    assert all("apple_1" not in (e.from_object_id, e.to_object_id) for e in frame.edges)


def test_drifting_status_from_displacement():
    builder = SceneGraphBuilder("seq")
    depth = _depth_with([(A, 1000.0)])
    frame = builder.build(_tracking_frame([_obj("apple_1", A, disp=80.0)]), depth)
    assert frame.nodes[0].status == NodeStatus.DRIFTING


def test_localization_wrong_object_from_detection():
    builder = SceneGraphBuilder("seq")
    depth = _depth_with([(A, 1000.0)])
    det = DetectionFrame(
        sequence_id="seq",
        frame_id=0,
        timestamp=0.0,
        detector_ran=True,
        trigger_reason=TriggerReason.INIT,
        prompts_used=["apple"],
        detections=[Detection("pear_0", "pear", A, 0.7, True)],
        detection_success=True,
        failure_mode=DetectionFailureMode.WRONG_CATEGORY,
    )
    frame = builder.build(_tracking_frame([_obj("apple_1", A)]), depth, det)
    assert frame.localization_flag.failure_detected is True
    assert frame.localization_flag.type == LocalizationFailureType.WRONG_OBJECT
    assert frame.localization_flag.affected_object_id == "pear_0"


def test_localization_missing_on_disappearance():
    builder = SceneGraphBuilder("seq", config=SceneGraphConfig(occlusion_buffer_frames=0))
    depth = _depth_with([(A, 1000.0), (B, 1000.0)])
    builder.build(_tracking_frame([_obj("apple_1", A), _obj("bowl_2", B)], 0), depth)
    frame = builder.build(_tracking_frame([_obj("apple_1", A)], 1), _depth_with([(A, 1000.0)]))
    assert frame.localization_flag.failure_detected is True
    assert frame.localization_flag.type == LocalizationFailureType.MISSING
    assert frame.localization_flag.affected_object_id == "bowl_2"


def test_assembler_roundtrip(tmp_path=None):
    tmp = tmp_path or tempfile.mkdtemp()
    tracking_path = os.path.join(tmp, "tracking.jsonl")
    out_path = os.path.join(tmp, "scene_graph.jsonl")

    # Write input tracking rows through the real serializer (validates the parser).
    writer = JsonlWriter(tracking_path)
    writer.write(_tracking_frame([_obj("apple_1", A), _obj("bowl_2", B)], 0))
    writer.write(_tracking_frame([_obj("apple_1", A)], 1))  # bowl_2 disappears

    depths = {
        0: _depth_with([(A, 1000.0), (B, 1000.0)]),
        1: _depth_with([(A, 1000.0)]),
    }
    written = assemble(tracking_path, lambda fid: depths[fid], out_path, config=SceneGraphConfig(occlusion_buffer_frames=0))
    assert written == 2

    rows = [json.loads(line) for line in open(out_path, encoding="utf-8") if line.strip()]
    assert len(rows) == 2
    # Frame 0: enriched nodes + a near edge.
    assert rows[0]["nodes"][0]["status"] == "ok"
    assert "bbox_xyxy" in rows[0]["nodes"][0]
    assert any(e["relation"] == "near" for e in rows[0]["edges"])
    # Frame 1: bowl_2 missing -> localization flag.
    assert rows[1]["localization_flag"]["type"] == "Missing"
    assert rows[1]["localization_flag"]["affected_object_id"] == "bowl_2"


def test_gripper_node_and_held_edge():
    """Gripper closes after object moves → gripper node added + held_by_gripper edge."""
    builder = SceneGraphBuilder("seq", config=SceneGraphConfig(grip_approach_window=3))
    # apple_1 moves across frames (centroid displacement); bowl_2 is stationary.
    # Attribution must pick apple_1 (highest displacement), not bowl_2.
    moving_bboxes = [[50, 50, 100, 100], [60, 60, 110, 110], [70, 70, 120, 120]]
    for fi, bbox in enumerate(moving_bboxes):
        builder.build(
            _tracking_frame([_obj("apple_1", bbox), _obj("bowl_2", B)], fi),
            _depth_with([(bbox, 1000.0), (B, 1000.0)]),
        )
    frame = builder.build(
        _tracking_frame([_obj("apple_1", [80, 80, 130, 130]), _obj("bowl_2", B)], 3),
        _depth_with([([80, 80, 130, 130], 1000.0), (B, 1000.0)]),
        gripper_closed=True,
    )
    assert any(n.object_id == "gripper" for n in frame.nodes), "gripper node missing"
    held = [e for e in frame.edges if e.relation == "held_by_gripper"]
    assert held, "held_by_gripper edge missing"
    assert held[0].from_object_id == "apple_1", f"expected apple_1 held, got {held[0].from_object_id}"
    assert held[0].to_object_id == "gripper"
    assert held[0].source == "gripper_state_displacement"


def test_slip_detection():
    """Object slips: tracker drops it while held, then gripper opens → SLIP on open."""
    builder = SceneGraphBuilder("seq", config=SceneGraphConfig(grip_approach_window=2))
    # apple_1 moves between frames so displacement attribution fires.
    builder.build(_tracking_frame([_obj("apple_1", A)], 0), _depth_with([(A, 1000.0)]))
    builder.build(
        _tracking_frame([_obj("apple_1", [60, 60, 110, 110])], 1),
        _depth_with([([60, 60, 110, 110], 1000.0)]),
        gripper_closed=True,
    )
    assert builder._held_object_id == "apple_1"
    # Tracker drops object while gripper is still closed (arm occlusion) — NOT a slip yet.
    frame_closed = builder.build(
        _tracking_frame([], 2),
        np.zeros((300, 300), dtype=np.float64),
        gripper_closed=True,
    )
    assert frame_closed.localization_flag.failure_detected is False, "should not fire SLIP while gripper closed"
    # Ghost node should still be present (gripper state confirms hold).
    assert any(n.object_id == "apple_1" for n in frame_closed.nodes)
    # Gripper opens; object still not visible to tracker → SLIP.
    frame_open = builder.build(
        _tracking_frame([], 3),
        np.zeros((300, 300), dtype=np.float64),
        gripper_closed=False,
    )
    assert frame_open.localization_flag.failure_detected is True
    assert frame_open.localization_flag.type == LocalizationFailureType.SLIP
    assert frame_open.localization_flag.affected_object_id == "apple_1"


def test_no_grasp_when_no_depth():
    """Gripper closes but all objects have invalid depth → NO_GRASP."""
    builder = SceneGraphBuilder("seq")
    # Object with invalid depth (bbox over zero background).
    frame = builder.build(
        _tracking_frame([_obj("apple_1", A)], 0),
        np.zeros((300, 300), dtype=np.float64),
        gripper_closed=True,
    )
    assert frame.localization_flag.failure_detected is True
    assert frame.localization_flag.type == LocalizationFailureType.NO_GRASP


def test_occlusion_buffer_ghost_node():
    """Object absent from tracker is re-emitted as OCCLUDED ghost; MISSING fires after buffer expires."""
    builder = SceneGraphBuilder("seq", config=SceneGraphConfig(occlusion_buffer_frames=2))
    depth = _depth_with([(A, 1000.0), (B, 1000.0)])
    builder.build(_tracking_frame([_obj("apple_1", A), _obj("bowl_2", B)], 0), depth)

    # Frame 1: bowl_2 absent — should appear as ghost, no MISSING yet.
    frame1 = builder.build(_tracking_frame([_obj("apple_1", A)], 1), _depth_with([(A, 1000.0)]))
    ghost_ids = [n.object_id for n in frame1.nodes]
    assert "bowl_2" in ghost_ids, "ghost node missing on first absence"
    ghost = next(n for n in frame1.nodes if n.object_id == "bowl_2")
    assert ghost.status == NodeStatus.OCCLUDED
    assert frame1.localization_flag.failure_detected is False

    # Frame 2: still absent — still ghost.
    frame2 = builder.build(_tracking_frame([_obj("apple_1", A)], 2), _depth_with([(A, 1000.0)]))
    assert any(n.object_id == "bowl_2" for n in frame2.nodes)
    assert frame2.localization_flag.failure_detected is False

    # Frame 3: buffer expired (occlusion_buffer_frames=2) → evicted → MISSING fires.
    frame3 = builder.build(_tracking_frame([_obj("apple_1", A)], 3), _depth_with([(A, 1000.0)]))
    assert not any(n.object_id == "bowl_2" for n in frame3.nodes)
    assert frame3.localization_flag.failure_detected is True
    assert frame3.localization_flag.type == LocalizationFailureType.MISSING
    assert frame3.localization_flag.affected_object_id == "bowl_2"

    # Frame 4: object reappears — buffer clears, back to OK.
    frame4 = builder.build(
        _tracking_frame([_obj("apple_1", A), _obj("bowl_2", B)], 4),
        _depth_with([(A, 1000.0), (B, 1000.0)]),
    )
    reappeared = next(n for n in frame4.nodes if n.object_id == "bowl_2")
    assert reappeared.status == NodeStatus.OK


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
