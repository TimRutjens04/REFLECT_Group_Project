from pathlib import Path

from data_loader.rgbd_loader import VideoRgbdFrameProvider
from data_loader.task_loader import TaskLoader
from data_loader.workspace import setup_workspace
from detector.GroundingDinoDetector import GroundingDinoDetector, DetectorConfig
from detector.runner import DetectionRunner
from detector.prompt_strategy import PromptStrategy
from models.base import JsonlWriter
from models.detection import TriggerReason
from sam2_tracker_redetect import track_video_with_sam2
from scripts.notebook_helpers import detection_result_to_pil
from validator import CompositeTrackingValidator
from yoloe_tracker import track_video_with_yoloe_redetect
from yoloe_tracker2 import track_video_with_yoloe
from detector.locateanything import LocateAnythingDetector


def main():
    setup_workspace(Path("/home/coder/datasets"))
    loader = TaskLoader(Path("/home/coder/datasets"))
    task = loader.get(1)
    provider = VideoRgbdFrameProvider(task)

    # --- Detection on frame 0 ---
    frame0 = provider.get_frame(0)
    print(
        f"Loaded frame {frame0.step_idx} with RGB shape {frame0.rgb.shape} and depth shape {frame0.depth.shape}"
    )

    jsonl_dir = Path("real_world/jsonl")
    detection_writer = JsonlWriter(jsonl_dir / "detections.jsonl")
    tracking_writer = JsonlWriter(jsonl_dir / "tracking.jsonl")

    detector = GroundingDinoDetector(DetectorConfig())
    detector.load()

    runner = DetectionRunner(
        detector=detector,
        strategy=PromptStrategy(),
        log_dir=Path("real_world/state_summary/detection"),
        jsonl_writer=detection_writer,
    )

    detection_result = runner.run(frame0, task, trigger_reason=TriggerReason.INIT)

    output_dir = Path("real_world/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_img = detection_result_to_pil(frame0, detection_result)
    detection_path = output_dir / f"detection_step_{frame0.step_idx}.png"
    detection_img.save(detection_path)
    print(f"Detection result: {detection_result}")
    print(f"Saved detection image to {detection_path.resolve()}")

    # --- Debug: YOLOe visual-prompt predict on frame 0 ---
    color_video = provider.color_path

    tracked_output = Path("real_world/videos") / f"tracked_{task.folder_name}.mp4"
    # track_video_with_yoloe_redetect(
    #     video_path=color_video,
    #     initial_detection_result=detection_result,
    #     output_path=tracked_output,
    #     frame_step=1,
    #     sequence_id=task.folder_name,
    #     detection_writer=detection_writer,
    #     tracking_writer=tracking_writer,
    #     redetect_every_n_frames=30,
    #     provider=provider,
    #     detection_runner=runner,
    #     task=task,
    #     redetect_on_lost=False,
    #     redetect_on_invalid=True,
    #     validator=CompositeTrackingValidator(),
    #     validate_with_depth=True,
    # )
    track_video_with_yoloe(
        video_path=color_video,
        detection_result=detection_result,
        output_path=Path("real_world/videos")
        / f"tracked_no_redetect_{task.folder_name}.mp4",
        frame_step=1,
        sequence_id=task.folder_name,
        detection_writer=detection_writer,
        tracking_writer=tracking_writer,
    )
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


if __name__ == "__main__":
    main()
