"""
This file is intentionally separate from the tracker implementation. It does
not run detection or tracking. It expects an upstream tracker to provide one
JSONL file with stable tracked objects per sampled frame, then it adds depth
statistics, 3D object nodes, primary scene-graph relations, plots, keyframes,
and overlays.

Expected input JSONL row shape:

{
  "sequence_id": "putAppleBowl1",
  "frame_id": 12,
  "timestamp": 12.0,
  "tracked_objects": [
    {
      "object_id": "apple_0",
      "label": "red apple",
      "bbox_xyxy": [x1, y1, x2, y2],
      "tracker_confidence": 0.91,
      "tracker_status": "ok",
      "flags": {
        "bbox_size_change_flag": false,
        "drift_flag": false,
        "recovery_trigger": false
      },
      "last_detection_frame": 0
    }
  ]
}
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


DEFAULT_INTRINSICS = np.array(
    [
        [914.27246, 0.0, 647.0733],
        [0.0, 913.2658, 356.32526],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

NEAR_THRESHOLDS_TO_TEST = [0.15, 0.20, 0.25, 0.30, 0.40]
DEPTH_IQR_THRESHOLDS_TO_TEST = [0.10, 0.15, 0.20, 0.30]
VALID_RATIO_THRESHOLDS_TO_TEST = [0.20, 0.30, 0.40, 0.50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run depth reliability and scene graph analysis from tracker JSONL.",
    )
    parser.add_argument("--tracking-jsonl", required=True, help="Tracker handoff JSONL file.")
    parser.add_argument("--raw-episode-dir", help="Raw RGB-D episode directory. Defaults to Data/Data/Real Data/<sequence_id>.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to project outputs/pipeline_depth/<sequence_id>.")
    parser.add_argument("--raw-fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--near-threshold", type=float, default=0.20)
    parser.add_argument("--inside-overlap-threshold", type=float, default=0.70)
    parser.add_argument("--on-top-overlap-threshold", type=float, default=0.20)
    parser.add_argument("--support-depth-threshold", type=float, default=0.15)
    parser.add_argument("--direction-pixel-threshold", type=float, default=35.0)
    parser.add_argument("--valid-ratio-threshold", type=float, default=0.30)
    parser.add_argument("--min-valid-pixels", type=int, default=50)
    parser.add_argument("--iqr-threshold", type=float, default=0.20)
    parser.add_argument("--jump-threshold", type=float, default=0.30)
    parser.add_argument("--bad-frame-patience", type=int, default=2)
    parser.add_argument("--crop-mode", choices=["full_bbox", "center_60"], default="full_bbox")
    parser.add_argument("--center-crop-ratio", type=float, default=0.60)
    parser.add_argument("--keyframe-min-gap-sec", type=float, default=5.0)
    parser.add_argument("--keyframe-motion-threshold-px", type=float, default=120.0)
    parser.add_argument("--keyframe-motion-threshold-m", type=float, default=0.20)
    return parser.parse_args()


def bbox_area(bbox: list[float] | np.ndarray) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def containment_ratio(inner: list[float] | np.ndarray, outer: list[float] | np.ndarray) -> float:
    area = bbox_area(inner)
    return bbox_intersection(inner, outer) / area if area > 0 else 0.0


def compact_relation_label(relation: str) -> str:
    labels = {
        "near": "near",
        "inside": "inside",
        "on_top_of": "on top",
        "left_of": "left",
        "right_of": "right",
        "above": "above",
        "below": "below",
    }
    return labels.get(relation, relation)


def relation_display_edges(edges: list[dict[str, Any]], enabled_relations: set[str]) -> list[dict[str, Any]]:
    display_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    inverse = {
        "left_of": "right_of",
        "right_of": "left_of",
        "above": "below",
        "below": "above",
    }
    for edge in edges:
        relation = edge["relation"]
        if relation not in enabled_relations:
            continue
        if relation in inverse:
            pair = tuple(sorted((edge["from"], edge["to"])))
            group = "left/right" if relation in {"left_of", "right_of"} else "above/below"
            key = (pair[0], pair[1], group)
            if key not in display_edges:
                display_edges[key] = {
                    **edge,
                    "from": pair[0],
                    "to": pair[1],
                    "display_relation": group,
                }
            continue
        display_edges[(edge["from"], edge["to"], relation)] = {
            **edge,
            "display_relation": compact_relation_label(relation),
        }
    return list(display_edges.values())


def relation_offsets(edges: list[dict[str, Any]], positions: dict[str, tuple[float, float]], scale: float) -> dict[int, tuple[float, float]]:
    groups: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        if edge["from"] in positions and edge["to"] in positions:
            groups[tuple(sorted((edge["from"], edge["to"])))].append(index)
    offsets: dict[int, tuple[float, float]] = {}
    for indices in groups.values():
        center = (len(indices) - 1) / 2.0
        for rank, index in enumerate(indices):
            edge = edges[index]
            x1, y1 = positions[edge["from"]]
            x2, y2 = positions[edge["to"]]
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length
            amount = (rank - center) * scale
            offsets[index] = (nx * amount, ny * amount)
    return offsets


def stable_snapshot_layout(object_ids: list[str]) -> dict[str, tuple[float, float]]:
    """Place objects in a fixed mini-graph layout so timelines are easy to compare."""
    count = len(object_ids)
    if count == 1:
        return {object_ids[0]: (0.0, 0.0)}
    if count == 2:
        base_positions = [(-0.38, -0.20), (0.38, 0.20)]
    elif count == 3:
        base_positions = [(-0.38, -0.24), (0.38, -0.24), (0.0, 0.30)]
    elif count == 4:
        base_positions = [(-0.42, -0.28), (0.42, -0.28), (-0.42, 0.28), (0.42, 0.28)]
    else:
        radius_x, radius_y = 0.46, 0.30
        base_positions = [
            (
                radius_x * math.cos((2.0 * math.pi * index / count) - math.pi / 2.0),
                radius_y * math.sin((2.0 * math.pi * index / count) - math.pi / 2.0),
            )
            for index in range(count)
        ]
    return {object_id: base_positions[index] for index, object_id in enumerate(object_ids)}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")


def mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(clean)) if clean else None


class PipelineDepth:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(__file__).resolve().parent.parent
        self.workspace_root = self.project_root.parents[2]
        self.tracking_jsonl = Path(args.tracking_jsonl).expanduser().resolve()
        self.records: list[dict[str, Any]] = []
        self.sequence_id = ""
        self.raw_episode_dir: Path | None = None
        self.output_dir: Path | None = None
        self.color_zarr: Any = None
        self.depth_zarr: Any = None
        self.depth_divisor = 1.0
        self.intrinsics = DEFAULT_INTRINSICS.copy()

    def register_imagecodecs(self) -> None:
        try:
            imagecodecs = importlib.import_module("imagecodecs")
        except ImportError:
            return
        try:
            from numcodecs import registry
            imagecodecs_numcodecs = importlib.import_module("imagecodecs.numcodecs")
            for name in ("Jpeg2k", "Jpegxl", "Jpeg", "Jpegls", "Ljpeg"):
                cls = getattr(imagecodecs_numcodecs, name, None)
                if cls is not None:
                    with contextlib.suppress(Exception):
                        registry.register_codec(cls)
        except Exception:
            return

    def open_zarr(self, path: Path) -> Any:
        self.register_imagecodecs()
        import zarr
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return zarr.open(str(path), mode="r")

    def load_tracking_jsonl(self) -> None:
        if not self.tracking_jsonl.exists():
            raise FileNotFoundError(self.tracking_jsonl)
        rows = []
        for line_no, line in enumerate(self.tracking_jsonl.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            self.validate_tracking_row(row, line_no)
            rows.append(row)
        if not rows:
            raise ValueError(f"No rows found in {self.tracking_jsonl}")
        rows.sort(key=lambda row: (float(row["timestamp"]), int(row["frame_id"])))
        if self.args.max_frames is not None:
            rows = rows[: self.args.max_frames]
        self.records = rows
        self.sequence_id = str(rows[0]["sequence_id"])
        self.raw_episode_dir = (
            Path(self.args.raw_episode_dir).expanduser().resolve()
            if self.args.raw_episode_dir
            else self.workspace_root / "Data" / "Data" / "Real Data" / self.sequence_id
        )
        self.output_dir = (
            Path(self.args.output_dir).expanduser().resolve()
            if self.args.output_dir
            else self.project_root / "outputs" / "pipeline_depth" / self.sequence_id
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "graph_visualizations").mkdir(exist_ok=True)
        print("Tracking JSONL:", self.tracking_jsonl)
        print("Sequence:", self.sequence_id)
        print("Frames loaded:", len(self.records))
        print("Raw RGB-D episode:", self.raw_episode_dir)
        print("Output directory:", self.output_dir)

    @staticmethod
    def validate_tracking_row(row: dict[str, Any], line_no: int) -> None:
        required = ["sequence_id", "frame_id", "timestamp", "tracked_objects"]
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Line {line_no}: missing fields {missing}")
        if not isinstance(row["tracked_objects"], list):
            raise ValueError(f"Line {line_no}: tracked_objects must be a list")
        for obj in row["tracked_objects"]:
            obj_required = ["object_id", "bbox_xyxy"]
            obj_missing = [key for key in obj_required if key not in obj]
            if obj_missing:
                raise ValueError(f"Line {line_no}: tracked object missing {obj_missing}")
            bbox = obj["bbox_xyxy"]
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox)
                and bbox[2] > bbox[0]
                and bbox[3] > bbox[1]
            ):
                raise ValueError(f"Line {line_no}: invalid bbox for {obj.get('object_id')}: {bbox}")

    @staticmethod
    def object_label(obj: dict[str, Any]) -> str:
        if obj.get("label"):
            return str(obj["label"])
        object_id = str(obj.get("object_id", "object"))
        return object_id.rsplit("_", 1)[0]

    @staticmethod
    def object_flags(record: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
        flags = obj.get("flags")
        if isinstance(flags, dict):
            return flags
        frame_flags = record.get("flags")
        return frame_flags if isinstance(frame_flags, dict) else {}

    def open_raw_rgbd(self) -> None:
        assert self.raw_episode_dir is not None
        self.color_zarr = self.open_zarr(self.raw_episode_dir / "videos" / "color")
        self.depth_zarr = self.open_zarr(self.raw_episode_dir / "videos" / "depth")
        first = self.get_depth_by_timestamp(float(self.records[0]["timestamp"]))
        valid = first[np.isfinite(first) & (first > 0)]
        median = float(np.median(valid)) if valid.size else math.nan
        self.depth_divisor = 1000.0 if valid.size and median > 20 else 1.0
        unit = "millimeters" if self.depth_divisor == 1000.0 else "meters"
        print("Raw color:", self.color_zarr.shape, self.color_zarr.dtype)
        print("Raw depth:", self.depth_zarr.shape, self.depth_zarr.dtype)
        print(f"Depth unit selected: {unit}; first valid median={median:.3f}")

    def raw_index(self, timestamp: float) -> int:
        index = round(timestamp * self.args.raw_fps)
        return int(np.clip(index, 0, int(self.depth_zarr.shape[0]) - 1))

    def get_rgb_by_timestamp(self, timestamp: float) -> np.ndarray:
        return np.asarray(self.color_zarr[self.raw_index(timestamp)])

    def get_depth_by_timestamp(self, timestamp: float) -> np.ndarray:
        return np.asarray(self.depth_zarr[self.raw_index(timestamp)])

    def crop(self, depth: np.ndarray, bbox: list[float], mode: str) -> np.ndarray:
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        if mode == "center_60":
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            half_w = (x2 - x1) * self.args.center_crop_ratio / 2
            half_h = (y2 - y1) * self.args.center_crop_ratio / 2
            x1, x2, y1, y2 = mid_x - half_w, mid_x + half_w, mid_y - half_h, mid_y + half_h
        x1, x2 = int(np.clip(np.floor(x1), 0, w)), int(np.clip(np.ceil(x2), 0, w))
        y1, y2 = int(np.clip(np.floor(y1), 0, h)), int(np.clip(np.ceil(y2), 0, h))
        return depth[y1:y2, x1:x2]

    def depth_stats(self, depth: np.ndarray, bbox: list[float]) -> dict[str, Any]:
        crop = self.crop(depth, bbox, self.args.crop_mode)
        values = np.asarray(crop, dtype=float).reshape(-1) / self.depth_divisor
        values = values[np.isfinite(values) & (values > 0) & (values < 20)]
        total, valid = int(crop.size), int(values.size)
        stats: dict[str, Any] = {
            "valid_pixel_count": valid,
            "valid_depth_pixel_ratio": float(valid / max(1, total)),
            "depth_median_m": None,
            "depth_mean_m": None,
            "depth_std_m": None,
            "depth_iqr_m": None,
        }
        if valid:
            p25, p75 = np.percentile(values, [25, 75])
            stats.update(
                {
                    "depth_median_m": float(np.median(values)),
                    "depth_mean_m": float(np.mean(values)),
                    "depth_std_m": float(np.std(values)),
                    "depth_iqr_m": float(p75 - p25),
                }
            )
        return stats

    def node(self, obj: dict[str, Any], z: float | None) -> dict[str, Any] | None:
        if z is None or not np.isfinite(z):
            return None
        x1, y1, x2, y2 = [float(value) for value in obj["bbox_xyxy"]]
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        fx, fy = float(self.intrinsics[0, 0]), float(self.intrinsics[1, 1])
        cx, cy = float(self.intrinsics[0, 2]), float(self.intrinsics[1, 2])
        return {
            "object_id": str(obj["object_id"]),
            "label": self.object_label(obj),
            "state": str(obj.get("tracker_status", "tracked")),
            "bbox": [x1, y1, x2, y2],
            "pixel_center": [u, v],
            "depth_used_m": z,
            "position_3d": {"x": (u - cx) * z / fx, "y": (v - cy) * z / fy, "z": z},
        }

    def primary_relation(self, a: dict[str, Any], b: dict[str, Any], distance: float) -> dict[str, Any] | None:
        ax, ay = a["pixel_center"]
        bx, by = b["pixel_center"]
        ab_containment = containment_ratio(a["bbox"], b["bbox"])
        ba_containment = containment_ratio(b["bbox"], a["bbox"])
        depth_diff = abs(float(a["depth_used_m"]) - float(b["depth_used_m"]))

        if ab_containment >= self.args.inside_overlap_threshold:
            return {"from": a["object_id"], "to": b["object_id"], "relation": "inside", "distance_3d_m": distance, "source": "bbox_containment_depth"}
        if ba_containment >= self.args.inside_overlap_threshold:
            return {"from": b["object_id"], "to": a["object_id"], "relation": "inside", "distance_3d_m": distance, "source": "bbox_containment_depth"}

        overlap = bbox_intersection(a["bbox"], b["bbox"]) / max(1.0, min(bbox_area(a["bbox"]), bbox_area(b["bbox"])))
        if overlap >= self.args.on_top_overlap_threshold and depth_diff <= self.args.support_depth_threshold:
            top, bottom = (a, b) if ay < by else (b, a)
            return {"from": top["object_id"], "to": bottom["object_id"], "relation": "on_top_of", "distance_3d_m": distance, "source": "image_vertical_overlap"}

        if distance < self.args.near_threshold:
            return {"from": a["object_id"], "to": b["object_id"], "relation": "near", "distance_3d_m": distance, "source": "3d_distance"}

        dx, dy = abs(ax - bx), abs(ay - by)
        if max(dx, dy) < self.args.direction_pixel_threshold:
            return None
        if dx >= dy:
            left, right = (a, b) if ax < bx else (b, a)
            return {"from": left["object_id"], "to": right["object_id"], "relation": "left_of", "distance_3d_m": distance, "source": "dominant_image_axis"}
        above, below = (a, b) if ay < by else (b, a)
        return {"from": above["object_id"], "to": below["object_id"], "relation": "above", "distance_3d_m": distance, "source": "dominant_image_axis"}

    def edges(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for a, b in combinations(sorted(nodes, key=lambda value: value["object_id"]), 2):
            pa = np.array(list(a["position_3d"].values()), dtype=float)
            pb = np.array(list(b["position_3d"].values()), dtype=float)
            distance = float(np.linalg.norm(pa - pb))
            relation = self.primary_relation(a, b, distance)
            if relation is not None:
                result.append(relation)
        return result

    def process(self):
        depth_frames, graph_frames, depth_rows, graph_rows, edge_rows = [], [], [], [], []
        previous_depth: dict[str, float] = {}
        bad_counts: defaultdict[str, int] = defaultdict(int)
        for row_index, record in enumerate(self.records):
            frame_id, timestamp = int(record["frame_id"]), float(record["timestamp"])
            depth = self.get_depth_by_timestamp(timestamp)
            per_object, nodes = [], []
            for obj in record["tracked_objects"]:
                stats = self.depth_stats(depth, obj["bbox_xyxy"])
                object_id = str(obj["object_id"])
                median, iqr = stats["depth_median_m"], stats["depth_iqr_m"]
                validity = stats["valid_depth_pixel_ratio"] >= self.args.valid_ratio_threshold and stats["valid_pixel_count"] >= self.args.min_valid_pixels
                coherence = iqr is not None and iqr <= self.args.iqr_threshold
                jump = median is not None and object_id in previous_depth and abs(median - previous_depth[object_id]) > self.args.jump_threshold
                if median is not None:
                    previous_depth[object_id] = median
                raw_trigger = not validity or not coherence or jump
                bad_counts[object_id] = bad_counts[object_id] + 1 if raw_trigger else 0
                trigger = bad_counts[object_id] >= self.args.bad_frame_patience
                flags = self.object_flags(record, obj)
                row = {
                    "sequence_id": self.sequence_id,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "object_id": object_id,
                    "label": self.object_label(obj),
                    "tracker_confidence": obj.get("tracker_confidence"),
                    "tracker_status": obj.get("tracker_status"),
                    "tracker_flags": flags,
                    "last_detection_frame": obj.get("last_detection_frame"),
                    **stats,
                    "depth_jump_flag": bool(jump),
                    "depth_coherence_flag": bool(coherence),
                    "depth_validity_flag": bool(validity),
                    "raw_depth_trigger": bool(raw_trigger),
                    "any_depth_trigger": bool(trigger),
                }
                depth_rows.append(row)
                per_object.append({key: value for key, value in row.items() if key != "valid_pixel_count"})
                node = self.node(obj, median)
                if node is not None:
                    nodes.append(node)
            edges = self.edges(nodes)
            depth_frames.append({"sequence_id": self.sequence_id, "frame_id": frame_id, "timestamp": timestamp, "per_object_depth": per_object})
            graph_frames.append({"sequence_id": self.sequence_id, "frame_id": frame_id, "timestamp": timestamp, "near_distance_threshold_m": self.args.near_threshold, "nodes": nodes, "edges": edges})
            for edge in edges:
                edge_rows.append({
                    "sequence_id": self.sequence_id,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    **edge,
                    "near_distance_threshold_m": self.args.near_threshold,
                    "margin_to_threshold_m": self.args.near_threshold - edge["distance_3d_m"] if edge["relation"] == "near" else None,
                })
            graph_rows.append({
                "sequence_id": self.sequence_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "num_near_edges": sum(edge["relation"] == "near" for edge in edges),
                "num_raw_depth_triggers": sum(value["raw_depth_trigger"] for value in per_object),
                "num_depth_triggers": sum(value["any_depth_trigger"] for value in per_object),
                "num_tracker_recovery_triggers": sum(bool((value.get("tracker_flags") or {}).get("recovery_trigger") or (value.get("tracker_flags") or {}).get("any_recovery_trigger")) for value in per_object),
                "avg_depth_iqr_m": mean([value["depth_iqr_m"] for value in per_object]),
                "avg_valid_depth_pixel_ratio": mean([value["valid_depth_pixel_ratio"] for value in per_object]),
                "row_index": row_index,
            })
        return depth_frames, graph_frames, pd.DataFrame(depth_rows), pd.DataFrame(graph_rows), pd.DataFrame(edge_rows)

    def keyframe_indices(self, graph_frames, graph_df: pd.DataFrame) -> tuple[list[int], dict[str, list[str]]]:
        last_index = max(0, len(graph_frames) - 1)
        candidate_reasons: dict[int, set[str]] = defaultdict(set)
        candidate_reasons[0].add("start")
        candidate_reasons[last_index].add("end")

        for index, graph in enumerate(graph_frames):
            frame_id = int(graph["frame_id"])
            row = graph_df.iloc[index]
            if int(row.get("num_tracker_recovery_triggers", 0)) > 0:
                candidate_reasons[index].add("tracker_recovery")
            if int(row.get("num_depth_triggers", 0)) > 0:
                candidate_reasons[index].add("depth_trigger")

        previous_nodes: dict[str, dict[str, Any]] = {}
        for index, graph in enumerate(graph_frames):
            current_nodes = {node["object_id"]: node for node in graph["nodes"]}
            for object_id, node in current_nodes.items():
                if object_id not in previous_nodes:
                    continue
                previous = previous_nodes[object_id]
                pixel_move = float(np.linalg.norm(np.array(node["pixel_center"]) - np.array(previous["pixel_center"])))
                p3d = np.array(list(node["position_3d"].values()), dtype=float)
                q3d = np.array(list(previous["position_3d"].values()), dtype=float)
                move_3d = float(np.linalg.norm(p3d - q3d))
                if pixel_move >= self.args.keyframe_motion_threshold_px:
                    candidate_reasons[index].add("large_pixel_motion")
                if move_3d >= self.args.keyframe_motion_threshold_m:
                    candidate_reasons[index].add("large_3d_motion")
            previous_nodes = current_nodes

        kept, last_event_time = [], None
        for index in sorted(candidate_reasons):
            if index in {0, last_index}:
                kept.append(index)
                if index == 0:
                    last_event_time = float(graph_frames[index]["timestamp"])
                continue
            timestamp = float(graph_frames[index]["timestamp"])
            if last_event_time is None or timestamp - last_event_time >= self.args.keyframe_min_gap_sec:
                kept.append(index)
                last_event_time = timestamp
        if last_index not in kept:
            kept.append(last_index)
        kept = sorted(set(kept))
        reasons = {str(int(graph_frames[index]["frame_id"])): sorted(candidate_reasons[index]) for index in kept}
        return kept, reasons

    def export(self, depth_frames, graph_frames, depth_df, graph_df, edge_df) -> dict[str, int]:
        assert self.output_dir is not None
        seq = self.sequence_id
        write_jsonl(self.output_dir / f"{seq}__depth.jsonl", depth_frames)
        write_jsonl(self.output_dir / f"{seq}__scene_graph.jsonl", graph_frames)
        depth_df.to_csv(self.output_dir / f"{seq}__depth_timeseries.csv", index=False)
        graph_df.to_csv(self.output_dir / f"{seq}__scene_graph_timeseries.csv", index=False)
        edge_df.to_csv(self.output_dir / f"{seq}__edge_timeseries.csv", index=False)
        summary = {
            "num_frames": len(graph_frames),
            "num_depth_rows": len(depth_df),
            "num_edges": len(edge_df),
            "num_near_edges": int((edge_df["relation"] == "near").sum()) if not edge_df.empty else 0,
            "num_raw_depth_triggers": int(depth_df["raw_depth_trigger"].sum()) if not depth_df.empty else 0,
            "num_depth_triggers": int(depth_df["any_depth_trigger"].sum()) if not depth_df.empty else 0,
        }
        combined = {
            "metadata": {
                "sequence_id": seq,
                "tracking_jsonl": str(self.tracking_jsonl),
                "raw_episode_dir": str(self.raw_episode_dir),
                "input_contract": "tracker JSONL handoff; detector and tracker are not run here",
            },
            "thresholds": vars(self.args),
            "intrinsics": self.intrinsics.tolist(),
            "frames": graph_frames,
            "summary": summary,
        }
        (self.output_dir / f"{seq}__combined_depth_scene_graph.json").write_text(json.dumps(combined, indent=2, allow_nan=False), encoding="utf-8")
        return summary

    def save_keyframes(self, graph_frames, graph_df: pd.DataFrame) -> None:
        assert self.output_dir is not None
        keyframes, reasons = self.keyframe_indices(graph_frames, graph_df)
        records = [graph_frames[index] for index in keyframes]
        write_jsonl(self.output_dir / f"{self.sequence_id}__keyframe_scene_graph.jsonl", records)
        (self.output_dir / f"{self.sequence_id}__keyframes.json").write_text(
            json.dumps(
                {
                    "sequence_id": self.sequence_id,
                    "keyframe_indices": [int(graph_frames[index]["frame_id"]) for index in keyframes],
                    "keyframe_row_indices": keyframes,
                    "keyframe_reasons": reasons,
                    "frames": records,
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        graph_vis_dir = self.output_dir / "graph_visualizations"
        for index in keyframes:
            self.save_2d_graph_plot(graph_frames[index], graph_vis_dir / f"keyframe_{int(graph_frames[index]['frame_id']):04d}_graph.png")
        self.save_dynamic_graph_timeline(graph_frames, keyframes, graph_vis_dir / f"{self.sequence_id}__dynamic_graph_timeline.png")

    def save_2d_graph_plot(self, graph: dict[str, Any], output_path: Path) -> None:
        nodes = graph["nodes"]
        if not nodes:
            return
        pos = {node["object_id"]: tuple(node["pixel_center"]) for node in nodes}
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f"Frame {graph['frame_id']} scene graph")
        ax.invert_yaxis()
        ax.set_xlabel("image x")
        ax.set_ylabel("image y")
        colors = {"near": "#f4a261", "inside": "#2a9d8f", "on_top_of": "#e76f51", "left/right": "#8ab17d", "above/below": "#457b9d"}
        display_edges = relation_display_edges(graph["edges"], {"near", "inside", "on_top_of", "left_of", "right_of", "above", "below"})
        offsets = relation_offsets(display_edges, pos, scale=20.0)
        for edge_index, edge in enumerate(display_edges):
            if edge["from"] not in pos or edge["to"] not in pos:
                continue
            relation = edge["display_relation"]
            ox, oy = offsets.get(edge_index, (0.0, 0.0))
            x1, y1 = pos[edge["from"]]
            x2, y2 = pos[edge["to"]]
            x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=10, linewidth=1.4, color=colors.get(relation, "#555555"), alpha=0.75)
            ax.add_patch(arrow)
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, relation, fontsize=7, color=colors.get(relation, "#555555"), bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.3})
        for node in nodes:
            x, y = pos[node["object_id"]]
            ax.scatter([x], [y], s=180, color="#90be6d", edgecolor="black", zorder=3)
            ax.text(x + 5, y + 5, node["object_id"], fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=170)
        plt.close(fig)

    def save_dynamic_graph_timeline(self, graph_frames, keyframes: list[int], output_path: Path) -> None:
        object_ids = sorted({node["object_id"] for index in keyframes for node in graph_frames[index]["nodes"]})
        if not object_ids:
            return
        object_colors = {object_id: plt.cm.Set2(i % 8) for i, object_id in enumerate(object_ids)}
        relation_colors = {"near": "#f4a261", "inside": "#2a9d8f", "on_top_of": "#e76f51", "left/right": "#8ab17d", "above/below": "#457b9d"}
        base_layout = stable_snapshot_layout(object_ids)
        column_spacing = 2.25
        snapshot_width = 1.55
        snapshot_height = 1.02
        fig, ax = plt.subplots(figsize=(max(18, len(keyframes) * 4.8), 6.8))
        previous_positions: dict[str, tuple[float, float]] = {}
        for col, index in enumerate(keyframes):
            graph = graph_frames[index]
            nodes = graph["nodes"]
            if not nodes:
                previous_positions = {}
                continue
            positions = {
                node["object_id"]: (
                    col * column_spacing + base_layout[node["object_id"]][0],
                    base_layout[node["object_id"]][1],
                )
                for node in nodes
                if node["object_id"] in base_layout
            }
            ax.add_patch(plt.Rectangle((col * column_spacing - snapshot_width / 2 - 0.06, -snapshot_height / 2 - 0.06), snapshot_width + 0.12, snapshot_height + 0.12, fill=False, color="#dddddd", linewidth=1, zorder=0))
            display_edges = relation_display_edges(graph["edges"], {"near", "inside", "on_top_of", "left_of", "right_of", "above", "below"})
            offsets = relation_offsets(display_edges, positions, scale=0.045)
            for edge_index, edge in enumerate(display_edges):
                relation = edge["display_relation"]
                if edge["from"] not in positions or edge["to"] not in positions:
                    continue
                ox, oy = offsets.get(edge_index, (0.0, 0.0))
                x1, y1 = positions[edge["from"]]
                x2, y2 = positions[edge["to"]]
                x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
                ax.plot([x1, x2], [y1, y2], color=relation_colors.get(relation, "#555555"), linewidth=1.7, alpha=0.72)
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, relation, fontsize=7, color=relation_colors.get(relation, "#555555"), ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35})
            for object_id, (x, y) in positions.items():
                if object_id in previous_positions:
                    px, py = previous_positions[object_id]
                    ax.plot([px, x], [py, y], color="#888888", linestyle="--", linewidth=1.0, alpha=0.55)
                ax.scatter(x, y, s=170, color=object_colors[object_id], edgecolor="black", zorder=3)
                ax.text(x + 0.04, y + 0.04, object_id, fontsize=8)
            ax.text(col * column_spacing, -0.68, f"t{col}\nf{graph['frame_id']}\n{graph['timestamp']:.0f}s", ha="center", va="top", fontsize=8)
            previous_positions = positions
        ax.set_xticks([col * column_spacing for col in range(len(keyframes))])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xlim(-1.1, max(1.0, (len(keyframes) - 1) * column_spacing + 1.1))
        ax.set_ylim(-0.86, 0.76)
        ax.set_xlabel("time / keyframes")
        ax.set_title("Discrete-time dynamic scene graph over keyframes")
        handles = [plt.Line2D([0], [0], color=color, linewidth=2, label=label) for label, color in relation_colors.items()]
        handles.append(plt.Line2D([0], [0], color="#888888", linestyle="--", linewidth=1, label="same object over time"))
        ax.legend(handles=handles, loc="upper center", ncol=max(1, len(handles)), fontsize=8, frameon=False)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.grid(axis="x", alpha=0.12)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    def overlay(self, row_index: int, graph_frames, depth_frames, reliability_only: bool = False) -> np.ndarray:
        graph = graph_frames[row_index]
        depth = depth_frames[row_index]
        rgb = self.get_rgb_by_timestamp(float(graph["timestamp"])).copy()
        depth_objects = {value["object_id"]: value for value in depth["per_object_depth"]}
        centers = {value["object_id"]: tuple(int(x) for x in value["pixel_center"]) for value in graph["nodes"]}
        relation_colors = {"near": (40, 220, 255), "inside": (80, 220, 120), "on_top_of": (255, 120, 60), "left_of": (180, 180, 255), "above": (255, 180, 80)}
        if not reliability_only:
            for edge in graph["edges"]:
                if edge["relation"] in {"right_of", "below"}:
                    continue
                if edge["from"] in centers and edge["to"] in centers:
                    color = relation_colors.get(edge["relation"], (220, 220, 220))
                    cv2.line(rgb, centers[edge["from"]], centers[edge["to"]], color, 2)
                    mx = (centers[edge["from"]][0] + centers[edge["to"]][0]) // 2
                    my = (centers[edge["from"]][1] + centers[edge["to"]][1]) // 2
                    cv2.putText(rgb, edge["relation"], (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        for node in graph["nodes"]:
            object_id = node["object_id"]
            item = depth_objects.get(object_id)
            if not item:
                continue
            color = (0, 200, 0) if not item["raw_depth_trigger"] else ((255, 165, 0) if not item["any_depth_trigger"] else (255, 0, 0))
            x1, y1, x2, y2 = [int(value) for value in node["bbox"]]
            cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 3)
            text = f"{object_id} z={self.fmt(item['depth_median_m'])} iqr={self.fmt(item['depth_iqr_m'])} valid={item['valid_depth_pixel_ratio']:.2f}"
            if item["any_depth_trigger"]:
                text += " DEPTH"
            cv2.putText(rgb, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(rgb, f"frame={graph['frame_id']} t={graph['timestamp']:.2f}s", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return rgb

    @staticmethod
    def fmt(value: float | None) -> str:
        return "null" if value is None else f"{value:.2f}"

    def save_visuals(self, graph_frames, depth_frames, graph_df: pd.DataFrame) -> None:
        assert self.output_dir is not None
        frames = sorted(set([0, len(graph_frames) // 2, len(graph_frames) - 1, int(graph_df["num_depth_triggers"].idxmax()), int(graph_df["num_edges"].idxmax())]))
        for row_index in frames:
            image = self.overlay(row_index, graph_frames, depth_frames)
            cv2.imwrite(str(self.output_dir / "visualizations" / f"frame_{int(graph_frames[row_index]['frame_id']):04d}_scene_graph.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if self.args.skip_videos:
            print("Skipping overlay videos by request.")
            return
        timestamps = [float(graph["timestamp"]) for graph in graph_frames]
        fps = 1.0 / np.median(np.diff(timestamps)) if len(timestamps) > 1 else 1.0
        for name, reliability_only in [("scene_graph_overlay", False), ("depth_reliability_overlay", True)]:
            path = self.output_dir / f"{self.sequence_id}__{name}.mp4"
            first = self.overlay(0, graph_frames, depth_frames, reliability_only)
            h, w = first.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {path}")
            for row_index in range(len(graph_frames)):
                writer.write(cv2.cvtColor(self.overlay(row_index, graph_frames, depth_frames, reliability_only), cv2.COLOR_RGB2BGR))
            writer.release()
            print("Saved:", path)

    def save_plots(self, depth_df: pd.DataFrame, graph_df: pd.DataFrame) -> None:
        assert self.output_dir is not None
        plots = self.output_dir / "plots"
        specs = [
            (depth_df, "depth_median_m", "object_id"),
            (depth_df, "depth_iqr_m", "object_id"),
            (depth_df, "valid_depth_pixel_ratio", "object_id"),
            (graph_df, "num_nodes", None),
            (graph_df, "num_edges", None),
            (graph_df, "num_near_edges", None),
            (graph_df, "num_depth_triggers", None),
        ]
        for df, column, hue in specs:
            if df.empty or column not in df:
                continue
            fig, ax = plt.subplots(figsize=(11, 4))
            if hue:
                for name, group in df.groupby(hue):
                    ax.plot(group["timestamp"], group[column], label=name)
                ax.legend(fontsize=8)
            else:
                ax.plot(df["timestamp"], df[column])
            ax.set_title(column)
            ax.set_xlabel("timestamp (s)")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots / f"{self.sequence_id}__{column}.png", dpi=160)
            plt.close(fig)

    def save_flicker_and_sensitivity(self, graph_frames, depth_df: pd.DataFrame) -> None:
        assert self.output_dir is not None
        relation_keys = sorted({(edge["from"], edge["to"], edge["relation"]) for graph in graph_frames for edge in graph["edges"]})
        rows = []
        for source, target, relation in relation_keys:
            states, distances = [], []
            for graph in graph_frames:
                matches = [edge for edge in graph["edges"] if edge["from"] == source and edge["to"] == target and edge["relation"] == relation]
                states.append(bool(matches))
                if matches:
                    distances.append(matches[0]["distance_3d_m"])
            switches = sum(a != b for a, b in zip(states, states[1:]))
            rows.append({"sequence_id": self.sequence_id, "from": source, "to": target, "relation": relation, "num_present_frames": sum(states), "num_switches": switches, "flicker_rate": switches / max(1, len(states) - 1), "mean_distance_3d_m": mean(distances), "std_distance_3d_m": float(np.std(distances)) if distances else None})
        pd.DataFrame(rows).to_csv(self.output_dir / f"{self.sequence_id}__relation_flicker_summary.csv", index=False)
        pd.DataFrame([{"depth_iqr_threshold_m": threshold, "num_incoherent_objects": int((depth_df["depth_iqr_m"].isna() | (depth_df["depth_iqr_m"] > threshold)).sum())} for threshold in DEPTH_IQR_THRESHOLDS_TO_TEST]).to_csv(self.output_dir / f"{self.sequence_id}__depth_iqr_threshold_sensitivity.csv", index=False)
        pd.DataFrame([{"valid_ratio_threshold": threshold, "num_invalid_objects": int((depth_df["valid_depth_pixel_ratio"] < threshold).sum())} for threshold in VALID_RATIO_THRESHOLDS_TO_TEST]).to_csv(self.output_dir / f"{self.sequence_id}__valid_ratio_threshold_sensitivity.csv", index=False)

    def run(self) -> None:
        self.load_tracking_jsonl()
        self.open_raw_rgbd()
        depth_frames, graph_frames, depth_df, graph_df, edge_df = self.process()
        summary = self.export(depth_frames, graph_frames, depth_df, graph_df, edge_df)
        self.save_keyframes(graph_frames, graph_df)
        self.save_visuals(graph_frames, depth_frames, graph_df)
        self.save_plots(depth_df, graph_df)
        self.save_flicker_and_sensitivity(graph_frames, depth_df)
        print("\nSummary:", summary)
        print("Key message: detector/tracker are upstream; this file only consumes their JSONL and adds depth + scene graph outputs.")
        print("Outputs:", self.output_dir)


def main() -> None:
    PipelineDepth(parse_args()).run()


if __name__ == "__main__":
    main()
