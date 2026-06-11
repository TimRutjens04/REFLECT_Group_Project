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
from depth.pipeline_depth import run_depth_scene_graph

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

    # --- Depth + scene graph ---
    # Consumes the tracker/validator handoff JSONL and reads RGB/depth frames
    # through the same data loader provider used above.
    validation_jsonl = jsonl_dir / "validation.jsonl"
    run_depth_scene_graph(
        tracking_jsonl=validation_jsonl,
        provider=provider,
        output_dir=run_root / "depth",
    )
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
