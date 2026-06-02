"""
Depth consistency and baseline temporal scene graph integration for the tracker.

This script keeps tracking ownership in code/track.py. It prepares expected
RGB-D inputs from the shared REFLECT data folder when needed, optionally runs the
tracker, then turns tracked bounding boxes into robust depth summaries, one 3D point
per object, and per-frame scene graphs with only the "near" relation.

"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import math
import subprocess
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
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
DEFAULT_OBJECTS = ["apple", "pear", "carrot", "bowl"]
NEAR_THRESHOLDS_TO_TEST = [0.25, 0.30, 0.40, 0.50, 0.60]
DEPTH_IQR_THRESHOLDS_TO_TEST = [0.10, 0.15, 0.20, 0.30]
VALID_RATIO_THRESHOLDS_TO_TEST = [0.20, 0.30, 0.40, 0.50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build depth reliability outputs and near-only scene graphs from the tracker.",
    )
    parser.add_argument("--episode", default="putFruitsBowl1")
    parser.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    parser.add_argument("--raw-fps", type=float, default=30.0)
    parser.add_argument("--tracker-fps", type=float, default=2.0)
    parser.add_argument("--prep-max-frames", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--rebuild-inputs", action="store_true")
    parser.add_argument("--skip-tracker", action="store_true")
    parser.add_argument("--force-tracker", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-alternatives", action="store_true")
    parser.add_argument("--near-threshold", type=float, default=0.40)
    parser.add_argument("--valid-ratio-threshold", type=float, default=0.30)
    parser.add_argument("--min-valid-pixels", type=int, default=50)
    parser.add_argument("--iqr-threshold", type=float, default=0.20)
    parser.add_argument("--jump-threshold", type=float, default=0.30)
    parser.add_argument("--bad-frame-patience", type=int, default=2)
    parser.add_argument("--crop-mode", choices=["full_bbox", "center_60"], default="full_bbox")
    parser.add_argument("--center-crop-ratio", type=float, default=0.60)
    return parser.parse_args()


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(__file__).resolve().parent.parent
        self.code_dir = self.project_root / "code"
        self.track_dir = self.project_root / "track"
        self.aligned_dir = self.project_root / "aligned"
        self.depth_dir = self.project_root / "depth_state"
        self.output_dir = self.project_root / "outputs" / "depth_scene_graph"
        self.workspace_root = self.project_root.parents[2]
        self.raw_episode_dir = (
            self.workspace_root / "Data" / "Data" / "Real Data" / args.episode
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        self.depth_divisor = 1.0
        self.intrinsics = DEFAULT_INTRINSICS.copy()
        self.color_zarr: Any = None
        self.depth_zarr: Any = None
        self.tracker: Any = None
        self.tracker_source: Path | None = None
        self.tim_inputs_rebuilt = False

    @property
    def objects(self) -> list[str]:
        return [value.strip() for value in self.args.objects.split(",") if value.strip()]

    def register_imagecodecs(self) -> None:
        module = importlib.import_module("imagecodecs.numcodecs")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            module.register_codecs()

    def open_zarr(self, path: Path) -> Any:
        self.register_imagecodecs()
        return importlib.import_module("zarr").open(str(path), mode="r")

    def check_paths(self) -> None:
        required = [
            self.code_dir / "track.py",
            self.raw_episode_dir / "videos" / "color",
            self.raw_episode_dir / "videos" / "depth",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))
        print("Project root:", self.project_root)
        print("Raw RGB-D episode:", self.raw_episode_dir)
        print("Output directory:", self.output_dir)

    def prepare_tim_inputs(self) -> tuple[Path, Path]:
        aligned_path = self.aligned_dir / f"{self.args.episode}.npz"
        depth_path = self.depth_dir / f"{self.args.episode}.npz"
        if aligned_path.exists() and depth_path.exists() and not self.args.rebuild_inputs:
            print("Reusing the input artifacts:", aligned_path, depth_path, sep="\n  ")
            return aligned_path, depth_path

        color = self.open_zarr(self.raw_episode_dir / "videos" / "color")
        depth = self.open_zarr(self.raw_episode_dir / "videos" / "depth")
        safe_count = min(int(color.shape[0]), int(depth.shape[0]))
        step = self.args.raw_fps / self.args.tracker_fps
        indices = np.array(
            [int(i * step) for i in range(int(math.ceil(safe_count / step)))],
            dtype=int,
        )
        indices = indices[indices < safe_count]
        if self.args.prep_max_frames is not None:
            indices = indices[: self.args.prep_max_frames]
        print(f"Preparing {len(indices)} aligned RGB-D frames from {safe_count} native frames.")
        frames = np.stack([np.asarray(color[int(index)]) for index in indices]).astype(
            np.uint8
        )
        depths = np.stack([np.asarray(depth[int(index)]) for index in indices])
        timestamps = indices.astype(np.float64) / self.args.raw_fps
        self.aligned_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            aligned_path,
            timestamps=timestamps,
            frames=frames,
            failure_labels=np.zeros(len(indices), dtype=bool),
            fps_base=np.float64(self.args.tracker_fps),
        )
        np.savez_compressed(depth_path, depth=depths, timestamps=timestamps)
        self.tim_inputs_rebuilt = True
        print("Saved:", aligned_path)
        print("Saved:", depth_path)
        return aligned_path, depth_path

    def expected_track_path(self) -> Path:
        slug = "_".join(value.replace(" ", "_") for value in self.objects)
        return self.track_dir / f"{self.args.episode}_{slug}.npz"

    def run_tim_tracker(self) -> None:
        output = self.expected_track_path()
        if self.args.skip_tracker:
            print("Skipping the tracker execution by request.")
            return
        if output.exists() and not self.args.force_tracker and not self.tim_inputs_rebuilt:
            print("Reusing the existing tracker output.")
            return
        prompt = ". ".join(self.objects) + "."
        command = [
            sys.executable,
            str(self.code_dir / "track.py"),
            self.args.episode,
            "--object",
            prompt,
            "--force",
        ]
        print(f"Running the tracker for {len(self.objects)} configured object columns.")
        try:
            subprocess.run(command, cwd=self.project_root, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"The tracker failed with exit code {exc.returncode}. "
                "Review the tracker traceback printed above."
            ) from None

    def load_tracker(self) -> None:
        preferred = self.expected_track_path()
        exact = self.track_dir / f"{self.args.episode}.npz"
        available = sorted(self.track_dir.glob(f"{self.args.episode}*.npz"))
        selected = preferred if preferred.exists() else exact if exact.exists() else None
        if selected is None and available:
            selected = available[0]
            print("WARNING: using an available tracker output for this episode.")
        if selected is None:
            raise FileNotFoundError(
                f"No tracker output exists for {self.args.episode}. "
                "Run without --skip-tracker or generate the track NPZ with code/track.py."
            )
        tracker = np.load(selected, allow_pickle=False)
        required = [
            "boxes",
            "track_ids",
            "tracking_confidence",
            "failure_flags",
            "label_vocab",
        ]
        missing = [key for key in required if key not in tracker.files]
        if missing:
            raise KeyError(f"{selected} is missing tracker fields: {missing}")
        self.tracker, self.tracker_source = tracker, selected
        self.boxes = tracker["boxes"]
        self.track_ids = tracker["track_ids"]
        self.tracking_confidence = tracker["tracking_confidence"]
        self.failure_flags = tracker["failure_flags"]
        self.label_vocab = [str(value) for value in tracker["label_vocab"]]
        self.fps_base = float(tracker["fps_base"]) if "fps_base" in tracker else 2.0
        self.timestamps = (
            np.asarray(tracker["timestamps"], dtype=float)
            if "timestamps" in tracker
            else np.arange(len(self.boxes), dtype=float) / self.fps_base
        )
        self.recovery_frames = (
            tracker["recovery_frames"].tolist() if "recovery_frames" in tracker else []
        )
        self.id_switch_frames = (
            tracker["id_switch_frames"].tolist() if "id_switch_frames" in tracker else []
        )
        self.frame_count = min(len(self.boxes), len(self.timestamps))
        if self.args.max_frames is not None:
            self.frame_count = min(self.frame_count, self.args.max_frames)
        print("Tracker output loaded.")
        print("Tracked frames:", self.frame_count)
        print("Tracked object columns:", len(self.label_vocab))

    def open_raw_rgbd(self) -> None:
        self.color_zarr = self.open_zarr(self.raw_episode_dir / "videos" / "color")
        self.depth_zarr = self.open_zarr(self.raw_episode_dir / "videos" / "depth")
        print("Raw color:", self.color_zarr.shape, self.color_zarr.dtype)
        print("Raw depth:", self.depth_zarr.shape, self.depth_zarr.dtype)
        first = self.get_depth(0)
        valid = first[np.isfinite(first) & (first > 0)]
        median = float(np.median(valid)) if valid.size else math.nan
        self.depth_divisor = 1000.0 if valid.size and median > 20 else 1.0
        unit = "millimeters" if self.depth_divisor == 1000.0 else "meters"
        print(f"Depth unit selected: {unit}; first-frame valid median={median:.3f}")

    def raw_index(self, frame_idx: int) -> int:
        index = round(float(self.timestamps[frame_idx]) * self.args.raw_fps)
        return int(np.clip(index, 0, int(self.depth_zarr.shape[0]) - 1))

    def get_rgb(self, frame_idx: int) -> np.ndarray:
        return np.asarray(self.color_zarr[self.raw_index(frame_idx)])

    def get_depth(self, frame_idx: int) -> np.ndarray:
        return np.asarray(self.depth_zarr[self.raw_index(frame_idx)])

    def crop(self, depth: np.ndarray, bbox: np.ndarray, mode: str) -> np.ndarray:
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

    def depth_stats(self, depth: np.ndarray, bbox: np.ndarray, mode: str) -> dict[str, Any]:
        crop = self.crop(depth, bbox, mode)
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

    def node(self, object_id: str, label: str, bbox: np.ndarray, z: float | None) -> dict[str, Any] | None:
        if z is None or not np.isfinite(z):
            return None
        x1, y1, x2, y2 = [float(value) for value in bbox]
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        fx, fy = float(self.intrinsics[0, 0]), float(self.intrinsics[1, 1])
        cx, cy = float(self.intrinsics[0, 2]), float(self.intrinsics[1, 2])
        return {
            "object_id": object_id,
            "label": label,
            "pixel_center": [u, v],
            "depth_used_m": z,
            "position_3d": {"x": (u - cx) * z / fx, "y": (v - cy) * z / fy, "z": z},
        }

    def edges(self, nodes: list[dict[str, Any]], threshold: float | None = None) -> list[dict[str, Any]]:
        threshold = self.args.near_threshold if threshold is None else threshold
        result = []
        for a, b in combinations(sorted(nodes, key=lambda value: value["object_id"]), 2):
            pa = np.array(list(a["position_3d"].values()), dtype=float)
            pb = np.array(list(b["position_3d"].values()), dtype=float)
            distance = float(np.linalg.norm(pa - pb))
            if distance < threshold:
                result.append(
                    {"from": a["object_id"], "to": b["object_id"], "relation": "near", "distance_3d_m": distance}
                )
        return result

    def process(self, crop_mode: str | None = None):
        crop_mode = crop_mode or self.args.crop_mode
        depth_frames, graph_frames, depth_rows, graph_rows, edge_rows = [], [], [], [], []
        previous_valid: dict[str, float] = {}
        bad_counts: defaultdict[str, int] = defaultdict(int)
        skipped: defaultdict[str, int] = defaultdict(int)
        for frame_idx in range(self.frame_count):
            depth, timestamp = self.get_depth(frame_idx), float(self.timestamps[frame_idx])
            per_object, nodes = [], []
            for obj_idx, label in enumerate(self.label_vocab):
                bbox, track_id = self.boxes[frame_idx, obj_idx], int(self.track_ids[frame_idx, obj_idx])
                if track_id == -1 or not np.all(np.isfinite(bbox)):
                    skipped["lost_or_nan"] += 1
                    continue
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    skipped["invalid_area"] += 1
                    continue
                object_id = f"{label}_{obj_idx}"
                stats = self.depth_stats(depth, bbox, crop_mode)
                median, iqr = stats["depth_median_m"], stats["depth_iqr_m"]
                validity = stats["valid_depth_pixel_ratio"] >= self.args.valid_ratio_threshold and stats["valid_pixel_count"] >= self.args.min_valid_pixels
                coherence = iqr is not None and iqr <= self.args.iqr_threshold
                jump = median is not None and object_id in previous_valid and abs(median - previous_valid[object_id]) > self.args.jump_threshold
                if median is not None:
                    previous_valid[object_id] = median
                raw_trigger = not validity or not coherence or jump
                bad_counts[object_id] = bad_counts[object_id] + 1 if raw_trigger else 0
                trigger = bad_counts[object_id] >= self.args.bad_frame_patience
                row = {
                    "sequence_id": self.args.episode, "frame_id": frame_idx, "timestamp": timestamp,
                    "object_id": object_id, "label": label,
                    "tracker_confidence": float(self.tracking_confidence[frame_idx, obj_idx]),
                    "tracker_failure_flags": int(self.failure_flags[frame_idx, obj_idx]),
                    **stats, "depth_jump_flag": bool(jump), "depth_coherence_flag": bool(coherence),
                    "depth_validity_flag": bool(validity), "raw_depth_trigger": bool(raw_trigger),
                    "any_depth_trigger": bool(trigger),
                }
                depth_rows.append(row)
                per_object.append({key: value for key, value in row.items() if key != "valid_pixel_count"})
                node = self.node(object_id, label, bbox, median)
                if node is not None:
                    nodes.append(node)
            edges = self.edges(nodes)
            depth_frames.append({"sequence_id": self.args.episode, "frame_id": frame_idx, "timestamp": timestamp, "per_object_depth": per_object})
            graph_frames.append({"sequence_id": self.args.episode, "frame_id": frame_idx, "timestamp": timestamp, "near_distance_threshold_m": self.args.near_threshold, "nodes": nodes, "edges": edges})
            for edge in edges:
                edge_rows.append({"sequence_id": self.args.episode, "frame_id": frame_idx, "timestamp": timestamp, **edge, "near_distance_threshold_m": self.args.near_threshold, "margin_to_threshold_m": self.args.near_threshold - edge["distance_3d_m"]})
            graph_rows.append({"sequence_id": self.args.episode, "frame_id": frame_idx, "timestamp": timestamp, "num_nodes": len(nodes), "num_edges": len(edges), "near_distance_threshold_m": self.args.near_threshold, "num_raw_depth_triggers": sum(value["raw_depth_trigger"] for value in per_object), "num_depth_triggers": sum(value["any_depth_trigger"] for value in per_object), "avg_depth_iqr_m": self.mean([value["depth_iqr_m"] for value in per_object]), "avg_valid_depth_pixel_ratio": self.mean([value["valid_depth_pixel_ratio"] for value in per_object])})
        print("Skipped tracked-object slots:", dict(skipped))
        return depth_frames, graph_frames, pd.DataFrame(depth_rows), pd.DataFrame(graph_rows), pd.DataFrame(edge_rows)

    @staticmethod
    def mean(values: list[float | None]) -> float | None:
        clean = [float(value) for value in values if value is not None and np.isfinite(value)]
        return float(np.mean(clean)) if clean else None

    def write_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False) + "\n")

    def export(self, depth_frames, graph_frames, depth_df, graph_df, edge_df) -> dict[str, int]:
        episode = self.args.episode
        self.write_jsonl(self.output_dir / f"{episode}__depth.jsonl", depth_frames)
        self.write_jsonl(self.output_dir / f"{episode}__scene_graph.jsonl", graph_frames)
        depth_df.to_csv(self.output_dir / f"{episode}__depth_timeseries.csv", index=False)
        graph_df.to_csv(self.output_dir / f"{episode}__scene_graph_timeseries.csv", index=False)
        edge_df.to_csv(self.output_dir / f"{episode}__edge_timeseries.csv", index=False)
        summary = {"num_frames": self.frame_count, "num_depth_rows": len(depth_df), "num_edges": len(edge_df), "num_raw_depth_triggers": int(depth_df["raw_depth_trigger"].sum()), "num_depth_triggers": int(depth_df["any_depth_trigger"].sum())}
        combined = {"metadata": {"sequence_id": episode, "tracker_source": str(self.tracker_source), "raw_episode_dir": str(self.raw_episode_dir)}, "thresholds": vars(self.args), "intrinsics": self.intrinsics.tolist(), "frames": graph_frames, "summary": summary}
        (self.output_dir / f"{episode}__combined_depth_scene_graph.json").write_text(json.dumps(combined, indent=2, allow_nan=False), encoding="utf-8")
        return summary

    def overlay(self, frame_idx: int, graph_frames, depth_frames, reliability_only=False) -> np.ndarray:
        rgb = self.get_rgb(frame_idx).copy()
        graph, depth = graph_frames[frame_idx], depth_frames[frame_idx]
        objects = {value["object_id"]: value for value in depth["per_object_depth"]}
        centers = {value["object_id"]: tuple(int(x) for x in value["pixel_center"]) for value in graph["nodes"]}
        if not reliability_only:
            for edge in graph["edges"]:
                cv2.line(rgb, centers[edge["from"]], centers[edge["to"]], (40, 220, 255), 3)
        for obj_idx, label in enumerate(self.label_vocab):
            object_id, bbox = f"{label}_{obj_idx}", self.boxes[frame_idx, obj_idx]
            if object_id not in objects or not np.all(np.isfinite(bbox)):
                continue
            item = objects[object_id]
            color = (0, 200, 0) if not item["raw_depth_trigger"] else ((255, 165, 0) if not item["any_depth_trigger"] else (255, 0, 0))
            x1, y1, x2, y2 = [int(value) for value in bbox]
            cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 3)
            z, iqr, ratio = item["depth_median_m"], item["depth_iqr_m"], item["valid_depth_pixel_ratio"]
            text = f"{object_id} z={self.fmt(z)} iqr={self.fmt(iqr)} valid={ratio:.2f}"
            if item["any_depth_trigger"]:
                text += " TRIGGER"
            cv2.putText(rgb, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(rgb, f"frame={frame_idx} t={self.timestamps[frame_idx]:.2f}s", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return rgb

    @staticmethod
    def fmt(value: float | None) -> str:
        return "null" if value is None else f"{value:.2f}"

    def save_visuals(self, graph_frames, depth_frames, graph_df) -> None:
        episode = self.args.episode
        frames = sorted(set([0, self.frame_count // 2, self.frame_count - 1, int(graph_df["num_depth_triggers"].idxmax()), int(graph_df["num_edges"].idxmax())]))
        for frame_idx in frames:
            image = self.overlay(frame_idx, graph_frames, depth_frames)
            cv2.imwrite(str(self.output_dir / "visualizations" / f"frame_{frame_idx:04d}_scene_graph.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if self.args.skip_videos:
            print("Skipping overlay videos by request.")
            return
        for name, reliability_only in [("scene_graph_overlay", False), ("depth_reliability_overlay", True)]:
            path = self.output_dir / f"{episode}__{name}.mp4"
            first = self.overlay(0, graph_frames, depth_frames, reliability_only)
            h, w = first.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps_base, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {path}")
            for frame_idx in range(self.frame_count):
                writer.write(cv2.cvtColor(self.overlay(frame_idx, graph_frames, depth_frames, reliability_only), cv2.COLOR_RGB2BGR))
            writer.release()
            print("Saved:", path)

    def save_plots(self, depth_df: pd.DataFrame, graph_df: pd.DataFrame) -> None:
        episode, plots = self.args.episode, self.output_dir / "plots"
        specs = [(depth_df, "depth_median_m", "object_id"), (depth_df, "depth_iqr_m", "object_id"), (depth_df, "valid_depth_pixel_ratio", "object_id"), (graph_df, "num_nodes", None), (graph_df, "num_edges", None), (graph_df, "num_depth_triggers", None), (graph_df, "avg_depth_iqr_m", None), (graph_df, "avg_valid_depth_pixel_ratio", None)]
        for df, column, hue in specs:
            fig, ax = plt.subplots(figsize=(11, 4))
            if hue:
                for name, group in df.groupby(hue):
                    ax.plot(group["frame_id"], group[column], label=name)
                ax.legend(fontsize=8)
            else:
                ax.plot(df["frame_id"], df[column])
            ax.set_title(column); ax.grid(alpha=0.25)
            fig.tight_layout(); fig.savefig(plots / f"{episode}__{column}.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 4))
        for name, group in depth_df.groupby("object_id"):
            ax.plot(group["frame_id"], group["depth_median_m"], label=f"{name} median")
            ax.plot(group["frame_id"], group["depth_mean_m"], linestyle="--", alpha=0.7, label=f"{name} mean")
        ax.set_title("Mean vs median depth"); ax.grid(alpha=0.25); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(plots / f"{episode}__mean_vs_median_depth.png", dpi=160); plt.close(fig)

    def save_edge_flicker(self, graph_frames) -> pd.DataFrame:
        pairs = sorted({tuple(sorted((edge["from"], edge["to"]))) for graph in graph_frames for edge in graph["edges"]})
        rows = []
        for pair in pairs:
            states, distances = [], []
            for graph in graph_frames:
                matches = [edge for edge in graph["edges"] if tuple(sorted((edge["from"], edge["to"]))) == pair]
                states.append(bool(matches))
                if matches:
                    distances.append(matches[0]["distance_3d_m"])
            switches = sum(a != b for a, b in zip(states, states[1:]))
            rows.append({"sequence_id": self.args.episode, "object_pair": "|".join(pair), "num_present_frames": sum(states), "num_switches": switches, "flicker_rate": switches / max(1, len(states) - 1), "mean_distance_3d_m": self.mean(distances), "std_distance_3d_m": float(np.std(distances)) if distances else None, "mean_margin_to_threshold_m": self.mean([self.args.near_threshold - value for value in distances])})
        result = pd.DataFrame(rows)
        result.to_csv(self.output_dir / f"{self.args.episode}__edge_flicker_summary.csv", index=False)
        return result

    def save_sensitivity(self, graph_frames, depth_df: pd.DataFrame) -> None:
        episode = self.args.episode
        default = {(graph["frame_id"], edge["from"], edge["to"]) for graph in graph_frames for edge in graph["edges"]}
        rows = []
        for threshold in NEAR_THRESHOLDS_TO_TEST:
            decisions, counts = set(), []
            for graph in graph_frames:
                edges = self.edges(graph["nodes"], threshold)
                counts.append(len(edges))
                decisions |= {(graph["frame_id"], edge["from"], edge["to"]) for edge in edges}
            rows.append({"threshold_m": threshold, "average_num_edges": np.mean(counts), "num_decisions_different_from_default": len(default.symmetric_difference(decisions))})
        pd.DataFrame(rows).to_csv(self.output_dir / f"{episode}__near_threshold_sensitivity.csv", index=False)
        pd.DataFrame([{"depth_iqr_threshold_m": threshold, "num_incoherent_objects": int((depth_df["depth_iqr_m"].isna() | (depth_df["depth_iqr_m"] > threshold)).sum())} for threshold in DEPTH_IQR_THRESHOLDS_TO_TEST]).to_csv(self.output_dir / f"{episode}__depth_iqr_threshold_sensitivity.csv", index=False)
        pd.DataFrame([{"valid_ratio_threshold": threshold, "num_invalid_objects": int((depth_df["valid_depth_pixel_ratio"] < threshold).sum())} for threshold in VALID_RATIO_THRESHOLDS_TO_TEST]).to_csv(self.output_dir / f"{episode}__valid_ratio_threshold_sensitivity.csv", index=False)

    def save_crop_comparison(self) -> None:
        rows = []
        for mode in ["full_bbox", "center_60"]:
            _, _, depth_df, _, _ = self.process(mode)
            rows.append({"crop_mode": mode, "average_depth_iqr_m": float(depth_df["depth_iqr_m"].mean()), "num_raw_triggers": int(depth_df["raw_depth_trigger"].sum()), "num_final_triggers": int(depth_df["any_depth_trigger"].sum()), "depth_stability_std_m": float(depth_df["depth_median_m"].std())})
        pd.DataFrame(rows).to_csv(self.output_dir / f"{self.args.episode}__crop_mode_comparison.csv", index=False)

    def print_break_summary(self, depth_df: pd.DataFrame, graph_df: pd.DataFrame, flicker_df: pd.DataFrame) -> None:
        print("\nWhere does it break?")
        print("Frames with zero valid graph nodes:", graph_df.loc[graph_df["num_nodes"] == 0, "frame_id"].tolist())
        print("Tracker recovery frames:", self.recovery_frames)
        print("Tracker ID-switch frames:", self.id_switch_frames)
        print("Objects with most triggers:")
        print(depth_df.groupby("object_id")["any_depth_trigger"].sum().sort_values(ascending=False).head(10).to_string())
        if not flicker_df.empty:
            print("Object pairs with highest edge flicker:")
            print(flicker_df.sort_values("flicker_rate", ascending=False).head(10).to_string(index=False))

    def run(self) -> None:
        self.check_paths()
        print("\n[1/4] Preparing tracker inputs")
        self.prepare_tim_inputs()
        print("\n[2/4] Running tracker before depth analysis")
        self.run_tim_tracker()
        print("\n[3/4] Loading tracker output")
        self.load_tracker()
        print("\n[4/4] Runningi depth reliability and scene graph analysis")
        self.open_raw_rgbd()
        depth_frames, graph_frames, depth_df, graph_df, edge_df = self.process()
        if depth_df.empty:
            raise RuntimeError("No tracked objects were available for depth extraction.")
        summary = self.export(depth_frames, graph_frames, depth_df, graph_df, edge_df)
        self.save_visuals(graph_frames, depth_frames, graph_df)
        self.save_plots(depth_df, graph_df)
        self.save_sensitivity(graph_frames, depth_df)
        flicker_df = self.save_edge_flicker(graph_frames)
        if not self.args.skip_alternatives:
            self.save_crop_comparison()
        else:
            print("Skipping crop-mode comparison by request.")
        self.print_break_summary(depth_df, graph_df, flicker_df)
        print("\nSummary:", summary)
        print("Outputs:", self.output_dir)


def main() -> None:
    Pipeline(parse_args()).run()


if __name__ == "__main__":
    main()
