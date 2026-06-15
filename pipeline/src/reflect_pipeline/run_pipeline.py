from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import zarr

from reflect_pipeline.data_loader.rgbd_loader import VideoRgbdFrameProvider
from reflect_pipeline.data_loader.task_loader import Task
from reflect_pipeline.data_loader.workspace import setup_workspace
from reflect_pipeline.detector.GroundingDinoDetector import GroundingDinoDetector
from reflect_pipeline.detector.runner import DetectionRunner
from reflect_pipeline.detector.prompt_strategy import PromptStrategy
from reflect_pipeline.interfaces.IFrameInput import RgbdFrameProvider
from reflect_pipeline.models.base import JsonlWriter
from reflect_pipeline.models.detection import TriggerReason
from reflect_pipeline.scripts.notebook_helpers import detection_result_to_pil
from reflect_pipeline.tracker.validator import CompositeTrackingValidator
from reflect_pipeline.tracker.yoloe_tracker import track_video_with_yoloe_redetect
from reflect_pipeline.depth.pipeline_depth import run_depth_scene_graph
from reflect_pipeline.scene_graph.build_scene_graphs import assemble as assemble_scene_graph
from reflect_pipeline.scene_graph.visualize_scene_graph import render_mp4 as render_sg_mp4

GripperFn = Callable[[int, float], bool]
EefFn = Callable[[int, float], np.ndarray]

RUN_CONFIG = {
    "redetect_every_n_frames": 30,
    "redetect_on_lost": False,
    "redetect_on_invalid": True,
    "validate_with_depth": True,
    "dedupe_by_label": True,
}


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


@dataclass
class EpisodeInput:
    provider: RgbdFrameProvider
    object_list: list[str]
    sequence_id: str
    gripper_fn: GripperFn | None = None
    eef_fn: EefFn | None = None
    T_cam_robot: np.ndarray | None = None
    video_path: Path | None = None

    @classmethod
    def from_task(cls, task: Task) -> "EpisodeInput":
        provider = VideoRgbdFrameProvider(task)
        gripper_fn, eef_fn = _load_proprioception(provider, task)
        T_cam_robot = _load_T_cam_robot()
        return cls(
            provider=provider,
            object_list=task.object_list,
            sequence_id=task.folder_name,
            gripper_fn=gripper_fn,
            eef_fn=eef_fn,
            T_cam_robot=T_cam_robot,
            video_path=provider.color_path,
        )


def _load_proprioception(
    provider: VideoRgbdFrameProvider, task: Task
) -> tuple[GripperFn | None, EefFn | None]:
    gripper_fn: GripperFn | None = None
    gripper_result = provider.load_gripper_states()
    if gripper_result is not None:
        zarr_ts, gripper_closed = gripper_result
        zarr_ts_rel = zarr_ts - zarr_ts[0]

        def _gripper_fn(frame_id: int, timestamp: float) -> bool:
            idx = int(np.searchsorted(zarr_ts_rel, timestamp).clip(0, len(zarr_ts_rel) - 1))
            return bool(gripper_closed[idx])

        gripper_fn = _gripper_fn

    eef_fn: EefFn | None = None
    zarr_path = Path(task.task_root) / "replay_buffer.zarr"
    if zarr_path.exists():
        zr = zarr.open_group(str(zarr_path), mode="r")
        eef_ts = np.array(zr["data/timestamp"][:])
        eef_poses = np.array(zr["data/robot_eef_pose"][:, :3])
        eef_ts_rel = eef_ts - eef_ts[0]

        def _eef_fn(frame_id: int, timestamp: float) -> np.ndarray:
            idx = int(np.searchsorted(eef_ts_rel, timestamp).clip(0, len(eef_ts_rel) - 1))
            return eef_poses[idx]

        eef_fn = _eef_fn

    return gripper_fn, eef_fn


def _load_T_cam_robot() -> np.ndarray | None:
    t_path = Path(__file__).resolve().parent / "annotations" / "T_cam_robot.npy"
    if t_path.exists():
        return np.load(str(t_path))
    return None


def run_episode(ep: EpisodeInput, detector: GroundingDinoDetector, out_dir: Path) -> Path:
    started_at = datetime.now()
    run_id = f"{started_at.strftime('%Y%m%d_%H%M%S')}_{ep.sequence_id}"
    run_root = setup_workspace(out_dir, run_id=run_id)

    metadata = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "sequence_id": ep.sequence_id,
        "object_list": ep.object_list,
        "git_commit": _git_commit(),
        "config": RUN_CONFIG,
    }
    (run_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Run ID: {run_id}  →  {run_root}")

    frame0 = ep.provider.get_frame(0)
    print(
        f"Loaded frame {frame0.step_idx} with RGB shape {frame0.rgb.shape} "
        f"and depth shape {frame0.depth.shape}"
    )

    jsonl_dir = run_root / "jsonl"
    detection_writer = JsonlWriter(jsonl_dir / "detections.jsonl")
    tracking_writer = JsonlWriter(jsonl_dir / "tracking.jsonl")
    validation_writer = JsonlWriter(jsonl_dir / "validation.jsonl")

    runner = DetectionRunner(
        detector=detector,
        strategy=PromptStrategy(),
        log_dir=run_root / "state_summary" / "detection",
        jsonl_writer=detection_writer,
    )

    class _EpTask:
        object_list = ep.object_list
        folder_name = ep.sequence_id

    ep_task = _EpTask()
    detection_result = runner.run(frame0, ep_task, trigger_reason=TriggerReason.INIT)

    output_dir = run_root / "images"
    detection_img = detection_result_to_pil(frame0, detection_result)
    detection_path = output_dir / f"detection_step_{frame0.step_idx}.png"
    detection_img.save(detection_path)
    print(f"Detection result: {detection_result}")
    print(f"Saved detection image to {detection_path.resolve()}")

    n = ep.provider.n_frames
    frames_iter = ((i, ep.provider.get_frame(i).rgb) for i in range(n))

    tracked_output = run_root / "videos" / f"tracked_{ep.sequence_id}.mp4"
    track_video_with_yoloe_redetect(
        frames=frames_iter,
        initial_detection_result=detection_result,
        output_path=tracked_output,
        frame_step=1,
        sequence_id=ep.sequence_id,
        detection_writer=detection_writer,
        tracking_writer=tracking_writer,
        validation_writer=validation_writer,
        redetect_every_n_frames=RUN_CONFIG["redetect_every_n_frames"],
        provider=ep.provider,
        detection_runner=runner,
        task=ep_task,
        redetect_on_lost=RUN_CONFIG["redetect_on_lost"],
        redetect_on_invalid=RUN_CONFIG["redetect_on_invalid"],
        validator=CompositeTrackingValidator(),
        validate_with_depth=RUN_CONFIG["validate_with_depth"],
        dedupe_by_label=RUN_CONFIG["dedupe_by_label"],
    )

    # --- Scene graph ---
    validation_jsonl = jsonl_dir / "validation.jsonl"
    sg_out = jsonl_dir / "scene_graph.jsonl"

    written = assemble_scene_graph(
        tracking_path=validation_jsonl,
        depth_fn=lambda fid: ep.provider.get_frame(fid).depth,
        out_path=sg_out,
        detection_path=jsonl_dir / "detections.jsonl",
        gripper_fn=ep.gripper_fn,
        eef_fn=ep.eef_fn,
        T_cam_robot=ep.T_cam_robot,
    )
    print(f"Scene graph: {written} frames → {sg_out}")

    if ep.video_path is not None:
        render_sg_mp4(
            sg_path=sg_out,
            video_path=ep.video_path,
            out_dir=run_root / "videos",
            fps=5,
            keyframes_only=True,
            out_filename=f"scene_graph_{ep.sequence_id}.mp4",
        )

    return run_root
