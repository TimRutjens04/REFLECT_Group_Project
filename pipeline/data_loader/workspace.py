from __future__ import annotations
from pathlib import Path


def setup_workspace(data_dir: Path, work_dir: Path | None = None, run_id: str | None = None) -> Path:
    data_dir = Path(data_dir)
    real_data_dir = data_dir / "real_data"
    task_json_path = data_dir / "tasks_real_world.json"
    work_dir = Path(work_dir) if work_dir else Path(".").resolve()

    outputs_root = work_dir / "outputs"
    outputs_root.mkdir(exist_ok=True)

    if run_id:
        run_root = outputs_root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        out_root = run_root
    else:
        out_root = outputs_root

    data_link = out_root / "data"
    if not data_link.exists():
        data_link.symlink_to(real_data_dir)

    task_link = out_root / "tasks_real_world.json"
    if not task_link.exists():
        task_link.symlink_to(task_json_path)

    for sub in ("state_summary", "images", "scene", "jsonl", "videos"):
        (out_root / sub).mkdir(exist_ok=True)

    return out_root
