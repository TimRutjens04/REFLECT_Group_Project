"""Synthetic-frame helpers shared by the headless tests.

No camera, no model weights: hand-built ``TrackedObject``s and numpy depth
arrays are enough to exercise the scene-graph builder deterministically.
"""

from __future__ import annotations

import numpy as np

from livescene.scene_graph.models import (
    TrackedObject,
    TrackerStatus,
    TrackingFlags,
    TrackingFrame,
)

# Default synthetic frame size used across tests.
W, H = 640, 480


def make_tracked(
    label: str,
    track_id: int,
    bbox: list[float],
    conf: float = 0.9,
    area_ratio: float = 1.0,
    displacement: float | None = 0.0,
) -> TrackedObject:
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return TrackedObject(
        object_id=f"{label}_{track_id}",
        bbox_xyxy=list(bbox),
        bbox_area_px=area,
        bbox_area_ratio_to_init=area_ratio,
        center_xy=[(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        displacement_px=displacement,
        tracker_confidence=conf,
        tracker_status=TrackerStatus.OK,
        frames_since_redetect=0,
    )


def make_frame(objs: list[TrackedObject], frame_id: int = 0) -> TrackingFrame:
    return TrackingFrame(
        sequence_id="test",
        frame_id=frame_id,
        timestamp=float(frame_id) / 30.0,
        tracked_objects=objs,
        flags=TrackingFlags(False, False, False),
    )


def uniform_depth(z: float, w: int = W, h: int = H) -> np.ndarray:
    return np.full((h, w), z, dtype=np.float32)


def region_depth(base_z: float, regions: list[tuple[list[float], float]], w: int = W, h: int = H) -> np.ndarray:
    """Depth map of base_z with rectangular patches set to other depths.

    regions: list of ([x1, y1, x2, y2], z) in pixel coordinates.
    """
    depth = np.full((h, w), base_z, dtype=np.float32)
    for bbox, z in regions:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        depth[y1:y2, x1:x2] = z
    return depth


def make_rgb_frame(w: int = W, h: int = H, seed: int = 0) -> np.ndarray:
    """A synthetic RGB frame with a few flat coloured rectangles on grey."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 96, dtype=np.uint8)
    for color in ((200, 60, 60), (60, 200, 60), (60, 60, 200)):
        x1, y1 = int(rng.integers(0, w - 120)), int(rng.integers(0, h - 120))
        img[y1 : y1 + 100, x1 : x1 + 100] = color
    return img
