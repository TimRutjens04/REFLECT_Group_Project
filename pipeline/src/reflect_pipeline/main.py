from pathlib import Path

from reflect_pipeline.data_loader.task_loader import TaskLoader
from reflect_pipeline.detector.GroundingDinoDetector import GroundingDinoDetector, DetectorConfig
from reflect_pipeline.run_pipeline import run_task

# Which task to run: set to a task ID (e.g. 1) for a single task, or None for all tasks.
TASK_ID: int | None = 1


def main():
    data_dir = Path(__file__).resolve().parents[1] / "example_data"
    loader = TaskLoader(data_dir)

    task_ids = loader.all_task_ids() if TASK_ID is None else [TASK_ID]

    detector = GroundingDinoDetector(DetectorConfig())
    detector.load()

    for tid in task_ids:
        task = loader.get(tid)
        print(f"\n=== Task {tid}: {task.name} ({task.folder_name}) ===")
        run_task(task, data_dir, detector)


if __name__ == "__main__":
    main()
