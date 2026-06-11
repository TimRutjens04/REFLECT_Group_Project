import json
import subprocess
from datetime import datetime
from pathlib import Path

from data_loader.rgbd_loader import VideoRgbdFrameProvider
from data_loader.task_loader import Task
from data_loader.workspace import setup_workspace
from detector.GroundingDinoDetector import GroundingDinoDetector
from detector.runner import DetectionRunner
from detector.prompt_strategy import PromptStrategy
from models.base import JsonlWriter
from models.detection import TriggerReason
from scripts.notebook_helpers import detection_result_to_pil
from tracker.validator import CompositeTrackingValidator
from tracker.yoloe_tracker import track_video_with_yoloe_redetect
from scene_graph.build_scene_graphs import assemble as assemble_scene_graph

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


def run_task(task: Task, data_dir: Path, detector: GroundingDinoDetector) -> Path:
    started_at = datetime.now()
    run_id = f"{started_at.strftime('%Y%m%d_%H%M%S')}_{task.folder_name}"
    run_root = setup_workspace(data_dir, run_id=run_id)

    metadata = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "task_id": task.task_id,
        "task_name": task.name,
        "sequence_id": task.folder_name,
        "object_list": task.object_list,
        "git_commit": _git_commit(),
        "config": RUN_CONFIG,
    }
    (run_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Run ID: {run_id}  →  {run_root}")

    provider = VideoRgbdFrameProvider(task)

    # --- Detection on frame 0 ---
    frame0 = provider.get_frame(0)
    print(
        f"Loaded frame {frame0.step_idx} with RGB shape {frame0.rgb.shape} and depth shape {frame0.depth.shape}"
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

    detection_result = runner.run(frame0, task, trigger_reason=TriggerReason.INIT)

    output_dir = run_root / "images"

    detection_img = detection_result_to_pil(frame0, detection_result)
    detection_path = output_dir / f"detection_step_{frame0.step_idx}.png"
    detection_img.save(detection_path)
    print(f"Detection result: {detection_result}")
    print(f"Saved detection image to {detection_path.resolve()}")

    color_video = provider.color_path

    tracked_output = run_root / "videos" / f"tracked_{task.folder_name}.mp4"
    track_video_with_yoloe_redetect(
        video_path=color_video,
        initial_detection_result=detection_result,
        output_path=tracked_output,
        frame_step=1,
        sequence_id=task.folder_name,
        detection_writer=detection_writer,
        tracking_writer=tracking_writer,
        validation_writer=validation_writer,
        redetect_every_n_frames=RUN_CONFIG["redetect_every_n_frames"],
        provider=provider,
        detection_runner=runner,
        task=task,
        redetect_on_lost=RUN_CONFIG["redetect_on_lost"],
        redetect_on_invalid=RUN_CONFIG["redetect_on_invalid"],
        validator=CompositeTrackingValidator(),
        validate_with_depth=RUN_CONFIG["validate_with_depth"],
        dedupe_by_label=RUN_CONFIG["dedupe_by_label"],
    )

    # --- Scene graph ---
    import zarr
    import numpy as np

    validation_jsonl = jsonl_dir / "validation.jsonl"
    sg_out = jsonl_dir / "scene_graph.jsonl"

    # depth fn: pull depth frame from the same provider used above
    def depth_fn(frame_id: int):
        return provider.get_frame(frame_id).depth

    # gripper fn: zarr gripper states, timestamps normalised to relative 0-base
    gripper_fn = None
    gripper_result = provider.load_gripper_states()
    if gripper_result is not None:
        zarr_ts, gripper_closed = gripper_result
        zarr_ts_rel = zarr_ts - zarr_ts[0]
        def _gripper_fn(frame_id: int, timestamp: float) -> bool:
            idx = int(np.searchsorted(zarr_ts_rel, timestamp).clip(0, len(zarr_ts_rel) - 1))
            return bool(gripper_closed[idx])
        gripper_fn = _gripper_fn

    # eef fn: robot EEF XYZ from zarr replay buffer
    eef_fn = None
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

    # T_cam_robot: camera-robot extrinsics (optional)
    T_cam_robot = None
    t_path = Path(__file__).resolve().parent / "annotations" / "T_cam_robot.npy"
    if t_path.exists():
        T_cam_robot = np.load(str(t_path))

    written = assemble_scene_graph(
        tracking_path=validation_jsonl,
        depth_fn=depth_fn,
        out_path=sg_out,
        detection_path=jsonl_dir / "detections.jsonl",
        gripper_fn=gripper_fn,
        eef_fn=eef_fn,
        T_cam_robot=T_cam_robot,
    )
    print(f"Scene graph: {written} frames → {sg_out}")
    # track_video_with_yoloe(
    #     video_path=color_video,
    #     detection_result=detection_result,
    #     output_path=Path("real_world/videos")
    #     / f"tracked_no_redetect_{task.folder_name}.mp4",
    #     frame_step=1,
    #     sequence_id=task.folder_name,
    #     detection_writer=detection_writer,
    #     tracking_writer=tracking_writer,
    # )
    # tracked_output = Path("real_world/videos") / f"trackedsam2_{task.folder_name}.mp4"
    # track_video_with_sam2(
    #     video_path=color_video,
    #     detection_result=detection_result,
    #     output_path=tracked_output,
    #     model_name="sam2_b.pt",
    #     frame_step=1,
    # )
    # tracked_output = Path("real_world/videos") / f"tracked_{task.folder_name}.mp4"
    # track_video_with_sam2(
    #     video_path=color_video,
    #     output_path=tracked_output,
    #     detection_runner=runner,
    #     provider=provider,
    #     task=task,
    #     redetect_every=30,
    # )
    return run_root
