"""Render unit tests: compose one split view from a synthetic SceneGraphFrame
(no models, no camera) and assert the output geometry.
"""

from __future__ import annotations

import numpy as np

from fixtures import H, W, make_frame, make_tracked, uniform_depth
from livescene.intrinsics import intrinsics_from_fov
from livescene.render import compose, draw_bbox_panel, draw_graph_panel
from livescene.scene_graph.builder import SceneGraphBuilder, SceneGraphConfig


def _scene_graph_frame():
    builder = SceneGraphBuilder(
        "render-test",
        config=SceneGraphConfig(),
        intrinsics=intrinsics_from_fov(W, H, 60.0),
        depth_scale_to_m=1.0,
    )
    objs = [
        make_tracked("apple", 1, [300, 220, 340, 260]),
        make_tracked("bowl", 2, [280, 200, 360, 280]),
        make_tracked("mug", 3, [500, 100, 560, 160]),
    ]
    return builder.build(make_frame(objs), uniform_depth(1.0))


def test_split_view_geometry():
    sg = _scene_graph_frame()
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    left = draw_bbox_panel(frame, sg)
    right = draw_graph_panel(sg, W, H)
    view = compose(left, right, ["apple", "bowl", "mug"], fps=12.3)
    assert view.shape == (H, 2 * W, 3)
    assert view.dtype == np.uint8


def test_bbox_panel_draws_on_copy():
    sg = _scene_graph_frame()
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    left = draw_bbox_panel(frame, sg)
    assert frame.sum() == 0  # original untouched
    assert left.sum() > 0  # boxes drawn


def test_graph_panel_draws_nodes_and_edges():
    sg = _scene_graph_frame()
    assert sg.edges, "fixture scene should produce edges"
    right = draw_graph_panel(sg, W, H)
    # More than the flat background: circles, lines, and labels were drawn.
    background = np.full((H, W, 3), right[0, 0], dtype=np.uint8)
    assert (right != background).any()


def test_compose_without_fps():
    sg = _scene_graph_frame()
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    view = compose(draw_bbox_panel(frame, sg), draw_graph_panel(sg, W, H), ["apple"])
    assert view.shape == (H, 2 * W, 3)
