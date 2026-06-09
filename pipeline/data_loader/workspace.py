from __future__ import annotations
from pathlib import Path


def setup_workspace(data_dir: Path, work_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir)
    real_data_dir = data_dir / "real_data"
    task_json_path = data_dir / "tasks_real_world.json"
    work_dir = Path(work_dir) if work_dir else Path(".").resolve()

    rw_root = work_dir / "real_world"
    rw_root.mkdir(exist_ok=True)

    data_link = rw_root / "data"
    if not data_link.exists():
        data_link.symlink_to(real_data_dir)

    task_link = rw_root / "tasks_real_world.json"
    if not task_link.exists():
        task_link.symlink_to(task_json_path)

    for sub in ("state_summary", "images", "scene"):
        (rw_root / sub).mkdir(exist_ok=True)

    return rw_root
