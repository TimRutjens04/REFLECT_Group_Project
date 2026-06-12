"""
Offline scene-graph assembler.

Reads the JSONL rows the pipeline already wrote (tracking is the per-object
spine; detection is joined at frame level) plus the per-frame depth map, runs
``SceneGraphBuilder``, and writes one ``scene_graph.jsonl`` row per frame.

Depth is supplied through an injectable ``depth_fn(frame_id) -> np.ndarray`` so
the assembler is testable without the video/zarr stack. The CLI wires
``depth_fn`` to ``VideoRgbdFrameProvider``.

    python3 build_scene_graphs.py \
        --tracking real_world/jsonl/tracking.jsonl \
        --detections real_world/jsonl/detections.jsonl \
        --task 1 --out real_world/jsonl/scene_graph.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reflect_pipeline.models.base import JsonlWriter
from reflect_pipeline.models.detection import (
    Detection,
    DetectionFailureMode,
    DetectionFrame,
    TriggerReason,
)
from reflect_pipeline.models.tracking import TrackedObject, TrackerStatus, TrackingFlags, TrackingFrame
from reflect_pipeline.scene_graph.scene_graph_builder import SceneGraphBuilder, SceneGraphConfig

DepthFn = Callable[[int], np.ndarray]
GripperFn = Callable[[int, float], bool]         # (frame_id, timestamp) -> gripper_closed
EefFn = Callable[[int, float], "np.ndarray | None"]  # (frame_id, timestamp) -> xyz or None


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_tracking(row: dict) -> TrackingFrame:
    objs = [
        TrackedObject(
            object_id=o["object_id"],
            bbox_xyxy=list(o["bbox_xyxy"]),
            bbox_area_px=o.get("bbox_area_px", 0.0),
            bbox_area_ratio_to_init=o.get("bbox_area_ratio_to_init", 1.0),
            center_xy=list(o.get("center_xy", [0.0, 0.0])),
            displacement_px=o.get("displacement_px"),
            tracker_confidence=o.get("tracker_confidence", 1.0),
            tracker_status=TrackerStatus(o.get("tracker_status", "ok")),
            frames_since_redetect=o.get("frames_since_redetect", 0),
        )
        for o in row.get("tracked_objects", [])
    ]
    flags_row = row.get("flags", {})
    flags = TrackingFlags(
        bbox_size_change_flag=flags_row.get("bbox_size_change_flag", False),
        drift_flag=flags_row.get("drift_flag", False),
        any_recovery_trigger=flags_row.get("any_recovery_trigger", False),
    )
    return TrackingFrame(
        sequence_id=row["sequence_id"],
        frame_id=row["frame_id"],
        timestamp=row["timestamp"],
        tracked_objects=objs,
        flags=flags,
    )


def _parse_detection(row: dict) -> DetectionFrame:
    failure = row.get("failure_mode")
    detections = [
        Detection(
            object_id=d["object_id"],
            label=d.get("label", ""),
            bbox_xyxy=list(d.get("bbox_xyxy", [])),
            confidence=d.get("confidence", 0.0),
            is_selected=d.get("is_selected", False),
        )
        for d in row.get("detections", [])
    ]
    return DetectionFrame(
        sequence_id=row["sequence_id"],
        frame_id=row["frame_id"],
        timestamp=row["timestamp"],
        detector_ran=row.get("detector_ran", False),
        trigger_reason=TriggerReason(row.get("trigger_reason", "none")),
        prompts_used=row.get("prompts_used", []),
        detections=detections,
        detection_success=row.get("detection_success", False),
        failure_mode=DetectionFailureMode(failure) if failure else None,
    )


def assemble(
    tracking_path: str | Path,
    depth_fn: DepthFn,
    out_path: str | Path,
    detection_path: str | Path | None = None,
    config: SceneGraphConfig | None = None,
    gripper_fn: GripperFn | None = None,
    eef_fn: "EefFn | None" = None,
    T_cam_robot: np.ndarray | None = None,
) -> int:
    tracking_rows = _read_jsonl(tracking_path)
    if not tracking_rows:
        raise ValueError(f"No tracking rows in {tracking_path}")

    detections_by_frame: dict[int, DetectionFrame] = {}
    if detection_path is not None and Path(detection_path).exists():
        for row in _read_jsonl(detection_path):
            detections_by_frame[int(row["frame_id"])] = _parse_detection(row)

    sequence_id = tracking_rows[0]["sequence_id"]
    builder = SceneGraphBuilder(sequence_id, config=config, T_cam_robot=T_cam_robot)
    out = Path(out_path)
    if out.exists():
        out.unlink()  # JsonlWriter appends; start clean
    writer = JsonlWriter(out)

    written = 0
    for row in tracking_rows:
        tracking_frame = _parse_tracking(row)
        depth = depth_fn(tracking_frame.frame_id)
        detection_frame = detections_by_frame.get(tracking_frame.frame_id)
        gripper_closed = (
            gripper_fn(tracking_frame.frame_id, tracking_frame.timestamp)
            if gripper_fn is not None
            else False
        )
        eef_pos = (
            eef_fn(tracking_frame.frame_id, tracking_frame.timestamp)
            if eef_fn is not None
            else None
        )
        scene_graph = builder.build(tracking_frame, depth, detection_frame, gripper_closed, eef_pos)
        writer.write(scene_graph)
        written += 1
    return written


def _provider_gripper_fn(task_index: int) -> GripperFn:
    from data_loader.rgbd_loader import VideoRgbdFrameProvider
    from data_loader.task_loader import TaskLoader

    data_dir = Path(__file__).resolve().parents[1] / "example_data"
    task = TaskLoader(data_dir).get(task_index)
    provider = VideoRgbdFrameProvider(task)
    result = provider.load_gripper_states()
    if result is None:
        return lambda fid, ts: False
    zarr_ts, gripper_closed = result
    # Zarr timestamps are Unix epoch; tracking timestamps are relative (0-based).
    # Normalize so searchsorted aligns correctly.
    zarr_ts_rel = zarr_ts - zarr_ts[0]
    def fn(frame_id: int, timestamp: float) -> bool:
        idx = int(np.searchsorted(zarr_ts_rel, timestamp).clip(0, len(zarr_ts) - 1))
        return bool(gripper_closed[idx])
    return fn


def _provider_eef_fn(task_index: int) -> "EefFn":
    import zarr
    from data_loader.task_loader import TaskLoader

    data_dir = Path(__file__).resolve().parents[1] / "example_data"
    task = TaskLoader(data_dir).get(task_index)
    zr = zarr.open_group(str(Path(task.task_root) / "replay_buffer.zarr"), mode="r")
    zarr_ts = np.array(zr["data/timestamp"][:])
    eef_poses = np.array(zr["data/robot_eef_pose"][:, :3])
    zarr_ts_rel = zarr_ts - zarr_ts[0]

    def fn(frame_id: int, timestamp: float) -> np.ndarray:
        idx = int(np.searchsorted(zarr_ts_rel, timestamp).clip(0, len(zarr_ts) - 1))
        return eef_poses[idx]

    return fn


def _provider_depth_fn(task_index: int) -> DepthFn:
    from data_loader.rgbd_loader import VideoRgbdFrameProvider
    from data_loader.task_loader import TaskLoader

    data_dir = Path(__file__).resolve().parents[1] / "example_data"
    task = TaskLoader(data_dir).get(task_index)
    provider = VideoRgbdFrameProvider(task)
    return lambda frame_id: provider.get_frame(frame_id).depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--detections", default=None)
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--out", default="real_world/jsonl/scene_graph.jsonl")
    parser.add_argument(
        "--T-cam-robot",
        default=str(Path(__file__).parent.parent / "annotations" / "T_cam_robot.npy"),
        help="Path to 4×4 SE3 transform numpy file (robot base → camera).",
    )
    args = parser.parse_args()

    T_cam_robot: np.ndarray | None = None
    t_path = Path(args.T_cam_robot)
    if t_path.exists():
        T_cam_robot = np.load(str(t_path))
        print(f"Loaded T_cam_robot from {t_path}")
    else:
        print(f"T_cam_robot not found at {t_path} — held-object attribution will use displacement fallback")

    written = assemble(
        tracking_path=args.tracking,
        depth_fn=_provider_depth_fn(args.task),
        out_path=args.out,
        detection_path=args.detections,
        gripper_fn=_provider_gripper_fn(args.task),
        eef_fn=_provider_eef_fn(args.task),
        T_cam_robot=T_cam_robot,
    )
    print(f"Wrote {written} scene graph rows to {args.out}")


if __name__ == "__main__":
    main()
