from data_loader.rgbd_loader import VideoRgbdFrameProvider
from data_loader.task_loader import TaskLoader
from data_loader.workspace import setup_workspace
from pathlib import Path
from detector.Sam2Detector import Sam2Detector, Sam2DetectorConfig
from detector.runner import DetectionRunner
from detector.sam2_prompt_strategy import Sam2PromptStrategy
from interfaces.IFrameInput import RgbdFrame
from scripts.notebook_helpers import detection_result_to_pil


def main():
    setup_workspace(Path("/home/coder/datasets"))
    loader = TaskLoader(Path("/home/coder/datasets"))
    task = loader.get(1)
    provider = VideoRgbdFrameProvider(task)
    frame: RgbdFrame = provider.get_frame(0)
    print(
        f"Loaded frame {frame.step_idx} with RGB shape {frame.rgb.shape} and depth shape {frame.depth.shape}"
    )

    detector = Sam2Detector(Sam2DetectorConfig())
    detector.load()

    runner = DetectionRunner(
        detector=detector,
        strategy=Sam2PromptStrategy(),
        log_path=Path("real_world/state_summary/detection.json"),
    )

    result = runner.run(frame, task)

    output_path = Path("real_world/images") / f"detectionsam2_step_{frame.step_idx}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = detection_result_to_pil(frame, result)
    annotated.save(output_path)

    print(f"Detection result: {result}")
    print(f"Saved annotated detection image to {output_path.resolve()}")


if __name__ == "__main__":
    main()
