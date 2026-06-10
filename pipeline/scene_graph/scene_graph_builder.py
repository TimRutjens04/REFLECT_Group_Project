"""
Scene graph builder for the modular pipeline.

Consumes the rich per-frame JSONL rows the pipeline already produces — tracking
(the per-object spine) plus the frame's depth map and the optional detection row
— and emits one enriched ``SceneGraphFrame`` per frame.

Why build from the model rows rather than the ``ISceneGraphBuilder`` interface:
the interface ``SceneGraphBuildInput`` carries the thin ``interfaces`` dataclasses
(no tracker_status, no validation) and the detection/tracking object_id schemes
differ (``label_index`` vs ``label_trackid``), so tracking is used as the spine
and detection signals are joined at frame level.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.detection import DetectionFrame, DetectionFailureMode
from models.tracking import TrackingFrame, TrackedObject
from models.scene_graph import (
    LocalizationFailureType,
    LocalizationFlag,
    NodeStatus,
    Position3D,
    SceneGraphEdge,
    SceneGraphFrame,
    SceneGraphNode,
)

# RealSense intrinsics used across the pipeline (see depth_scene_graph baseline).
DEFAULT_INTRINSICS = np.array(
    [
        [914.27246, 0.0, 647.0733],
        [0.0, 913.2658, 356.32526],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass
class SceneGraphConfig:
    near_threshold_m: float = 0.40
    inside_overlap_threshold: float = 0.70
    on_top_overlap_threshold: float = 0.20
    support_depth_threshold_m: float = 0.15
    direction_pixel_threshold: float = 35.0
    valid_ratio_threshold: float = 0.30
    min_valid_pixels: int = 50
    iqr_threshold_m: float = 0.20
    jump_threshold_m: float = 0.30
    # Status thresholds reuse the tracker/validator conventions.
    area_ratio_low: float = 0.60
    area_ratio_high: float = 1.67
    drift_px: float = 50.0
    # Gripper: frames of depth history for held-object attribution (mirrors code/track.py)
    grip_approach_window: int = 10


def _bbox_area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _bbox_intersection(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _containment_ratio(inner: list[float], outer: list[float]) -> float:
    area = _bbox_area(inner)
    return _bbox_intersection(inner, outer) / area if area > 0 else 0.0


class SceneGraphBuilder:
    """Stateful across frames so depth jumps and disappearances can be detected."""

    def __init__(
        self,
        sequence_id: str,
        config: SceneGraphConfig | None = None,
        intrinsics: np.ndarray | None = None,
        depth_scale_to_m: float | None = None,
        T_cam_robot: np.ndarray | None = None,
    ) -> None:
        self.sequence_id = sequence_id
        self.cfg = config or SceneGraphConfig()
        self.K = intrinsics if intrinsics is not None else DEFAULT_INTRINSICS
        self._depth_scale = depth_scale_to_m
        # 4×4 SE3: maps robot base frame → camera frame. None = fallback to displacement.
        self._T_cam_robot = T_cam_robot
        self._prev_depth: dict[str, float] = {}
        self._prev_ids: set[str] = set()
        # Gripper state
        self._gripper_was_closed: bool = False
        self._held_object_id: str | None = None
        self._held_object_source: str = "gripper_state_displacement"
        self._depth_history: dict[str, deque] = {}
        # Centroid history for displacement fallback (no T_cam_robot)
        self._centroid_history: dict[str, deque] = {}

    # ----- depth -------------------------------------------------------------
    def _resolve_depth_scale(self, depth: np.ndarray) -> float:
        if self._depth_scale is not None:
            return self._depth_scale
        valid = depth[np.isfinite(depth) & (depth > 0)]
        median = float(np.median(valid)) if valid.size else 0.0
        # Heuristic shared with the depth baseline: large values => millimetres.
        self._depth_scale = 0.001 if median > 20 else 1.0
        return self._depth_scale

    def _depth_stats(self, depth_m: np.ndarray, bbox: list[float]) -> dict:
        h, w = depth_m.shape[:2]
        x1 = int(np.clip(np.floor(bbox[0]), 0, w))
        x2 = int(np.clip(np.ceil(bbox[2]), 0, w))
        y1 = int(np.clip(np.floor(bbox[1]), 0, h))
        y2 = int(np.clip(np.ceil(bbox[3]), 0, h))
        crop = depth_m[y1:y2, x1:x2].reshape(-1)
        total = int(crop.size)
        valid = crop[np.isfinite(crop) & (crop > 0) & (crop < 20)]
        stats = {
            "valid_ratio": float(valid.size / max(1, total)),
            "valid_count": int(valid.size),
            "median_m": None,
            "iqr_m": None,
        }
        if valid.size:
            p25, p75 = np.percentile(valid, [25, 75])
            # 25th percentile = foreground depth (matches CompositeTrackingValidator).
            stats["median_m"] = float(p25)
            stats["iqr_m"] = float(p75 - p25)
        return stats

    def _position_3d(self, u: float, v: float, z: float) -> Position3D:
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        return Position3D(x=(u - cx) * z / fx, y=(v - cy) * z / fy, z=z)

    # ----- nodes -------------------------------------------------------------
    def _make_node(self, obj: TrackedObject, depth_m: np.ndarray) -> SceneGraphNode:
        bbox = list(obj.bbox_xyxy)
        u, v = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        stats = self._depth_stats(depth_m, bbox)
        median = stats["median_m"]

        validity = (
            stats["valid_ratio"] >= self.cfg.valid_ratio_threshold
            and stats["valid_count"] >= self.cfg.min_valid_pixels
        )
        coherence = stats["iqr_m"] is not None and stats["iqr_m"] <= self.cfg.iqr_threshold_m
        prev = self._prev_depth.get(obj.object_id)
        jump = median is not None and prev is not None and abs(median - prev) > self.cfg.jump_threshold_m
        if median is not None:
            self._prev_depth[obj.object_id] = median
        any_trigger = (not validity) or (not coherence) or jump

        z = median if median is not None else 0.0
        position = self._position_3d(u, v, z)
        status = self._derive_status(obj, validity, jump)

        return SceneGraphNode(
            object_id=obj.object_id,
            label=obj.object_id.rsplit("_", 1)[0],
            pixel_center=[u, v],
            depth_used_m=z,
            position_3d=position,
            bbox_xyxy=bbox,
            status=status,
            confidence=float(obj.tracker_confidence),
            depth_validity_flag=bool(validity),
            depth_coherence_flag=bool(coherence),
            depth_jump_flag=bool(jump),
            any_depth_trigger=bool(any_trigger),
        )

    def _derive_status(self, obj: TrackedObject, validity: bool, jump: bool) -> NodeStatus:
        if not validity:
            return NodeStatus.UNCERTAIN
        if jump:
            # A sudden depth change inside a stable box usually means an occluder.
            return NodeStatus.OCCLUDED
        ratio = obj.bbox_area_ratio_to_init
        drifting = (
            (obj.displacement_px is not None and obj.displacement_px > self.cfg.drift_px)
            or ratio < self.cfg.area_ratio_low
            or ratio > self.cfg.area_ratio_high
        )
        return NodeStatus.DRIFTING if drifting else NodeStatus.OK

    # ----- gripper -----------------------------------------------------------
    def _eef_to_pixel(self, eef_pos_robot: np.ndarray) -> tuple[float, float] | None:
        """Project EEF position (robot base XYZ, metres) into image pixel (u, v)."""
        R = self._T_cam_robot[:3, :3]
        t = self._T_cam_robot[:3, 3]
        P_cam = R @ np.asarray(eef_pos_robot, dtype=np.float64) + t
        if P_cam[2] <= 0.0:
            return None
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        return float(fx * P_cam[0] / P_cam[2] + cx), float(fy * P_cam[1] / P_cam[2] + cy)

    def _update_depth_history(self, object_id: str, depth_m: float) -> None:
        if object_id not in self._depth_history:
            self._depth_history[object_id] = deque(maxlen=self.cfg.grip_approach_window)
        self._depth_history[object_id].append(depth_m)

    def _update_centroid_history(self, object_id: str, cx: float, cy: float) -> None:
        if object_id not in self._centroid_history:
            self._centroid_history[object_id] = deque(maxlen=self.cfg.grip_approach_window)
        self._centroid_history[object_id].append((cx, cy))

    def _attribute_held_object(
        self, nodes: list[SceneGraphNode], eef_pos: np.ndarray | None = None
    ) -> tuple[str | None, str]:
        """Return (object_id, source) for the object the gripper is holding.

        Primary: project EEF into image via T_cam_robot, pick closest bbox center.
        Fallback (no T_cam_robot): pick object whose pixel centroid moved most.
        """
        if eef_pos is not None and self._T_cam_robot is not None:
            px = self._eef_to_pixel(eef_pos)
            if px is not None:
                u_eef, v_eef = px
                candidates = [n for n in nodes if n.object_id != "gripper"]
                if candidates:
                    best = min(
                        candidates,
                        key=lambda n: (n.pixel_center[0] - u_eef) ** 2
                        + (n.pixel_center[1] - v_eef) ** 2,
                    )
                    return best.object_id, "gripper_state_eef_proximity"

        # Displacement fallback.
        best_id: str | None = None
        best_disp = -1.0
        best_conf = -1.0
        any_history = False
        for node in nodes:
            hist = list(self._centroid_history.get(node.object_id, []))
            if len(hist) < 2:
                continue
            any_history = True
            dx = hist[-1][0] - hist[0][0]
            dy = hist[-1][1] - hist[0][1]
            disp = float(np.sqrt(dx * dx + dy * dy))
            if disp > best_disp or (disp == best_disp and node.confidence > best_conf):
                best_disp = disp
                best_conf = node.confidence
                best_id = node.object_id
        return (best_id if any_history else None), "gripper_state_displacement"

    def _gripper_node(self, eef_pos: np.ndarray | None = None) -> SceneGraphNode:
        pixel: list[float] = [0.0, 0.0]
        position = Position3D(0.0, 0.0, 0.0)
        depth_m = 0.0
        depth_valid = False
        if eef_pos is not None and self._T_cam_robot is not None:
            R = self._T_cam_robot[:3, :3]
            t = self._T_cam_robot[:3, 3]
            P_cam = R @ np.asarray(eef_pos, dtype=np.float64) + t
            if P_cam[2] > 0.0:
                fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
                cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
                pixel = [float(fx * P_cam[0] / P_cam[2] + cx), float(fy * P_cam[1] / P_cam[2] + cy)]
                position = Position3D(float(P_cam[0]), float(P_cam[1]), float(P_cam[2]))
                depth_m = float(P_cam[2])
                depth_valid = True
        return SceneGraphNode(
            object_id="gripper",
            label="gripper",
            pixel_center=pixel,
            depth_used_m=depth_m,
            position_3d=position,
            bbox_xyxy=[],
            status=NodeStatus.OK,
            confidence=1.0,
            depth_validity_flag=depth_valid,
            depth_coherence_flag=True,
            depth_jump_flag=False,
            any_depth_trigger=False,
        )

    # ----- edges -------------------------------------------------------------
    def _edges(self, nodes: list[SceneGraphNode]) -> list[SceneGraphEdge]:
        edges: list[SceneGraphEdge] = []
        usable = [n for n in nodes if n.depth_validity_flag and n.object_id != "gripper"]
        for a, b in combinations(sorted(usable, key=lambda n: n.object_id), 2):
            edge = self._primary_relation(a, b)
            if edge is not None:
                edges.append(edge)
        return edges

    def _primary_relation(self, a: SceneGraphNode, b: SceneGraphNode) -> SceneGraphEdge | None:
        pa = np.array([a.position_3d.x, a.position_3d.y, a.position_3d.z])
        pb = np.array([b.position_3d.x, b.position_3d.y, b.position_3d.z])
        distance = float(np.linalg.norm(pa - pb))
        ax, ay = a.pixel_center
        bx, by = b.pixel_center
        conf = min(a.confidence, b.confidence)

        ab = _containment_ratio(a.bbox_xyxy, b.bbox_xyxy)
        ba = _containment_ratio(b.bbox_xyxy, a.bbox_xyxy)
        depth_diff = abs(a.depth_used_m - b.depth_used_m)

        if ab >= self.cfg.inside_overlap_threshold:
            return SceneGraphEdge(a.object_id, b.object_id, "inside", distance, "bbox_containment_depth", conf)
        if ba >= self.cfg.inside_overlap_threshold:
            return SceneGraphEdge(b.object_id, a.object_id, "inside", distance, "bbox_containment_depth", conf)

        min_area = max(1.0, min(_bbox_area(a.bbox_xyxy), _bbox_area(b.bbox_xyxy)))
        overlap = _bbox_intersection(a.bbox_xyxy, b.bbox_xyxy) / min_area
        if overlap >= self.cfg.on_top_overlap_threshold and depth_diff <= self.cfg.support_depth_threshold_m:
            top, bottom = (a, b) if ay < by else (b, a)
            return SceneGraphEdge(top.object_id, bottom.object_id, "on_top_of", distance, "image_vertical_overlap", conf)

        if distance < self.cfg.near_threshold_m:
            return SceneGraphEdge(a.object_id, b.object_id, "near", distance, "3d_distance", conf)

        dx, dy = abs(ax - bx), abs(ay - by)
        if max(dx, dy) < self.cfg.direction_pixel_threshold:
            return None
        if dx >= dy:
            left, right = (a, b) if ax < bx else (b, a)
            return SceneGraphEdge(left.object_id, right.object_id, "left_of", distance, "dominant_image_axis", conf)
        above, below = (a, b) if ay < by else (b, a)
        return SceneGraphEdge(above.object_id, below.object_id, "above", distance, "dominant_image_axis", conf)

    # ----- localization flag -------------------------------------------------
    def _localization_flag(
        self,
        nodes: list[SceneGraphNode],
        detection: DetectionFrame | None,
        gripper_closed: bool,
        just_closed: bool,
    ) -> LocalizationFlag:
        # Wrong_object: detector reports the wrong / no category this frame.
        if detection is not None and detection.failure_mode in (
            DetectionFailureMode.WRONG_CATEGORY,
            DetectionFailureMode.MULTIPLE_AMBIGUOUS,
        ):
            affected = detection.detections[0].object_id if detection.detections else None
            return LocalizationFlag(True, LocalizationFailureType.WRONG_OBJECT, affected)

        # Slip: held object vanished while gripper is still closed.
        if self._held_object_id and gripper_closed:
            current_ids = {n.object_id for n in nodes}
            if self._held_object_id not in current_ids:
                return LocalizationFlag(True, LocalizationFailureType.SLIP, self._held_object_id)

        # No_Grasp: gripper just closed but depth approach gave no candidate.
        if just_closed and self._held_object_id is None:
            return LocalizationFlag(True, LocalizationFailureType.NO_GRASP, None)

        # Missing: an object tracked last frame is absent now (non-gripper objects only).
        current_ids = {n.object_id for n in nodes if n.object_id != "gripper"}
        disappeared = sorted(self._prev_ids - current_ids)
        if disappeared:
            return LocalizationFlag(True, LocalizationFailureType.MISSING, disappeared[0])

        return LocalizationFlag(failure_detected=False)

    # ----- public API --------------------------------------------------------
    def build(
        self,
        tracking_frame: TrackingFrame,
        depth: np.ndarray,
        detection_frame: DetectionFrame | None = None,
        gripper_closed: bool = False,
        eef_pos: np.ndarray | None = None,
    ) -> SceneGraphFrame:
        depth_m = np.asarray(depth, dtype=np.float64) * self._resolve_depth_scale(np.asarray(depth))
        nodes = [self._make_node(obj, depth_m) for obj in tracking_frame.tracked_objects]

        # Update histories before attribution so this frame is included.
        for node in nodes:
            if node.depth_validity_flag:
                self._update_depth_history(node.object_id, node.depth_used_m)
            self._update_centroid_history(node.object_id, node.pixel_center[0], node.pixel_center[1])

        # Gripper transitions.
        just_closed = gripper_closed and not self._gripper_was_closed
        if just_closed:
            self._held_object_id, self._held_object_source = self._attribute_held_object(nodes, eef_pos)
        elif not gripper_closed:
            self._held_object_id = None

        # Localization flag reads _prev_ids (prev frame) and _held_object_id (just set).
        flag = self._localization_flag(nodes, detection_frame, gripper_closed, just_closed)

        # Gripper agent node + held_by_gripper edge.
        if gripper_closed:
            nodes.append(self._gripper_node(eef_pos))
        edges = self._edges(nodes)
        if gripper_closed and self._held_object_id:
            held_node = next((n for n in nodes if n.object_id == self._held_object_id), None)
            if held_node:
                edges.append(SceneGraphEdge(
                    self._held_object_id, "gripper", "held_by_gripper", 0.0,
                    self._held_object_source, held_node.confidence,
                ))

        # Exclude gripper from _prev_ids so it never triggers Missing.
        self._prev_ids = {n.object_id for n in nodes if n.object_id != "gripper"}
        self._gripper_was_closed = gripper_closed
        return SceneGraphFrame(
            sequence_id=tracking_frame.sequence_id,
            frame_id=tracking_frame.frame_id,
            timestamp=tracking_frame.timestamp,
            near_distance_threshold_m=self.cfg.near_threshold_m,
            nodes=nodes,
            edges=edges,
            localization_flag=flag,
        )
