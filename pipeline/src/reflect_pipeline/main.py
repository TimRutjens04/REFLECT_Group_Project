import argparse
from pathlib import Path

from reflect_pipeline.data_loader.task_loader import TaskLoader
from reflect_pipeline.detector.GroundingDinoDetector import GroundingDinoDetector, DetectorConfig
from reflect_pipeline.run_pipeline import run_task

_EXAMPLE_DATA = Path(__file__).resolve().parents[3] / "example_data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to data/ directory (contains real_data/ and tasks_real_world.json). Defaults to example_data/ if not provided.")
    parser.add_argument("--task-id", type=int, default=1, help="Task ID to run (default: 1)")
    args = parser.parse_args()

    if args.data_dir is not None:
        data_dir = args.data_dir
    elif _EXAMPLE_DATA.exists():
        data_dir = _EXAMPLE_DATA
        print(f"No --data-dir provided, using example_data: {data_dir}")
    else:
        raise FileNotFoundError("No --data-dir provided and example_data/ not found. Pass --data-dir <path/to/data>.")

    loader = TaskLoader(data_dir)
    task_ids = loader.all_task_ids() if args.task_id is None else [args.task_id]

    detector = GroundingDinoDetector(DetectorConfig())
    detector.load()

    for tid in task_ids:
        task = loader.get(tid)
        print(f"\n=== Task {tid}: {task.name} ({task.folder_name}) ===")
        run_task(task, data_dir, detector)


if __name__ == "__main__":
    main()
