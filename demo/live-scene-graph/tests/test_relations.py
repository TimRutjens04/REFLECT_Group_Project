"""Deterministic scene-graph correctness tests: no camera, no models.

Hand-built TrackedObjects + synthetic depth arrays drive the copied
SceneGraphBuilder; these tests are the source of truth for graph logic.

All tests pin an explicit SceneGraphConfig (the REFLECT defaults) so later
webcam-scale retuning of AppConfig cannot silently change expectations.
Intrinsics: 640x480 @ 60 deg hfov -> fx = fy ~ 554.26, so at z = 1 m a
100 px pixel offset is ~0.18 m in 3D.
"""

from __future__ import annotations

import pytest

from fixtures import make_frame, make_tracked, uniform_depth
from livescene.intrinsics import intrinsics_from_fov
from livescene.scene_graph.builder import SceneGraphBuilder, SceneGraphConfig
from livescene.scene_graph.models import LocalizationFailureType, NodeStatus


def make_builder(**cfg_overrides) -> SceneGraphBuilder:
    return SceneGraphBuilder(
        sequence_id="test",
        config=SceneGraphConfig(**cfg_overrides),
        intrinsics=intrinsics_from_fov(640, 480, 60.0),
        depth_scale_to_m=1.0,
    )


def edges_by_relation(frame, relation):
    return [e for e in frame.edges if e.relation == relation]


def test_inside_when_contained_and_similar_depth():
    builder = make_builder()
    apple = make_tracked("apple", 1, [300, 220, 340, 260])
    bowl = make_tracked("bowl", 2, [280, 200, 360, 280])
    frame = builder.build(make_frame([apple, bowl]), uniform_depth(1.0))

    assert len(frame.edges) == 1
    edge = frame.edges[0]
    assert edge.relation == "inside"
    assert edge.from_object_id == "apple_1"
    assert edge.to_object_id == "bowl_2"


def test_on_top_of_when_overlapping_equal_depth_a_higher():
    builder = make_builder()
    # mug overlaps the top edge of the book: 25% of the mug box overlaps,
    # below the inside threshold but above the on-top overlap threshold.
    mug = make_tracked("mug", 1, [300, 150, 380, 230])
    book = make_tracked("book", 2, [280, 210, 400, 330])
    frame = builder.build(make_frame([mug, book]), uniform_depth(1.0))

    assert len(frame.edges) == 1
    edge = frame.edges[0]
    assert edge.relation == "on_top_of"
    assert edge.from_object_id == "mug_1"  # higher in the image = supported
    assert edge.to_object_id == "book_2"


def test_near_when_close_in_3d():
    builder = make_builder()
    # 120 px apart at z = 1 m -> ~0.22 m in 3D, below the 0.40 m threshold.
    a = make_tracked("apple", 1, [200, 200, 260, 260])
    b = make_tracked("mug", 2, [320, 200, 380, 260])
    frame = builder.build(make_frame([a, b]), uniform_depth(1.0))

    assert len(frame.edges) == 1
    edge = frame.edges[0]
    assert edge.relation == "near"
    assert edge.distance_3d_m == pytest.approx(120.0 / 554.256, rel=1e-3)


def test_left_of_when_far_apart_horizontally():
    builder = make_builder()
    # 400 px apart at z = 1 m -> ~0.72 m: too far for near, x-dominant.
    a = make_tracked("apple", 1, [100, 200, 160, 260])
    b = make_tracked("mug", 2, [500, 200, 560, 260])
    frame = builder.build(make_frame([a, b]), uniform_depth(1.0))

    assert len(frame.edges) == 1
    edge = frame.edges[0]
    assert edge.relation == "left_of"
    assert edge.from_object_id == "apple_1"
    assert edge.to_object_id == "mug_2"


def test_above_when_far_apart_vertically():
    builder = make_builder()
    a = make_tracked("shelf", 1, [300, 80, 360, 140])
    b = make_tracked("mug", 2, [300, 420, 360, 480])
    frame = builder.build(make_frame([a, b]), uniform_depth(1.0))

    assert len(frame.edges) == 1
    edge = frame.edges[0]
    assert edge.relation == "above"
    assert edge.from_object_id == "shelf_1"
    assert edge.to_object_id == "mug_2"


def test_missing_flag_when_object_disappears():
    # occlusion_buffer_frames=0 evicts immediately, so MISSING fires on the
    # first absent frame (the default buffer ghosts the node for 10 frames).
    builder = make_builder(occlusion_buffer_frames=0)
    apple = make_tracked("apple", 1, [200, 200, 260, 260])
    mug = make_tracked("mug", 2, [400, 200, 460, 260])
    builder.build(make_frame([apple, mug], frame_id=0), uniform_depth(1.0))
    frame = builder.build(make_frame([mug], frame_id=1), uniform_depth(1.0))

    flag = frame.localization_flag
    assert flag.failure_detected
    assert flag.type == LocalizationFailureType.MISSING
    assert flag.affected_object_id == "apple_1"


def test_occlusion_buffer_ghosts_absent_object():
    # With the default buffer, a just-vanished object stays as an OCCLUDED
    # ghost node with decayed confidence instead of triggering MISSING.
    builder = make_builder()
    apple = make_tracked("apple", 1, [200, 200, 260, 260], conf=0.9)
    mug = make_tracked("mug", 2, [400, 200, 460, 260])
    builder.build(make_frame([apple, mug], frame_id=0), uniform_depth(1.0))
    frame = builder.build(make_frame([mug], frame_id=1), uniform_depth(1.0))

    ghosts = [n for n in frame.nodes if n.object_id == "apple_1"]
    assert len(ghosts) == 1
    assert ghosts[0].status == NodeStatus.OCCLUDED
    assert ghosts[0].confidence < 0.9
    assert not frame.localization_flag.failure_detected


def test_missing_fires_after_occlusion_buffer_expires():
    builder = make_builder(occlusion_buffer_frames=3)
    apple = make_tracked("apple", 1, [200, 200, 260, 260])
    mug = make_tracked("mug", 2, [400, 200, 460, 260])
    builder.build(make_frame([apple, mug], frame_id=0), uniform_depth(1.0))

    missing_at = None
    for i in range(1, 10):
        frame = builder.build(make_frame([mug], frame_id=i), uniform_depth(1.0))
        flag = frame.localization_flag
        if flag.failure_detected and flag.type == LocalizationFailureType.MISSING:
            missing_at = i
            break
    assert missing_at is not None, "MISSING never fired after buffer expiry"
    assert missing_at > 1  # ghosted for a while first
    assert frame.localization_flag.affected_object_id == "apple_1"


def test_depth_jump_marks_node_occluded():
    builder = make_builder()
    mug = make_tracked("mug", 1, [200, 200, 280, 280])
    builder.build(make_frame([mug], frame_id=0), uniform_depth(1.0))
    # Same stable bbox, but depth inside it jumps 1.0 -> 2.0 m (> 0.30 m).
    frame = builder.build(make_frame([mug], frame_id=1), uniform_depth(2.0))

    node = next(n for n in frame.nodes if n.object_id == "mug_1")
    assert node.depth_jump_flag
    assert node.any_depth_trigger
    assert node.status == NodeStatus.OCCLUDED


def test_no_edges_for_single_object():
    builder = make_builder()
    frame = builder.build(make_frame([make_tracked("mug", 1, [200, 200, 280, 280])]), uniform_depth(1.0))
    assert frame.edges == []
    assert not frame.localization_flag.failure_detected


def test_nodes_carry_metric_positions():
    builder = make_builder()
    # Centre pixel at z = 1.5 m -> position on the optical axis.
    obj = make_tracked("mug", 1, [290, 210, 350, 270])  # centre = (320, 240)
    frame = builder.build(make_frame([obj]), uniform_depth(1.5))
    node = frame.nodes[0]
    assert node.depth_used_m == pytest.approx(1.5)
    assert node.position_3d.x == pytest.approx(0.0, abs=1e-9)
    assert node.position_3d.y == pytest.approx(0.0, abs=1e-9)
    assert node.position_3d.z == pytest.approx(1.5)
    assert node.label == "mug"
