from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Task:
    task_id: int
    name: str
    folder_name: str
    object_list: list[str]
    actions: list[Any]
    success_condition: str
    gt_failure_reason: str
    task_root: Path
    raw: dict[str, Any]


class TaskLoader:
    def __init__(self, data_dir: Path, task_json_path: Path | None = None):
        self.data_dir = Path(data_dir)
        self.real_data_dir = self.data_dir / "real_data"
        self.task_json_path = (
            Path(task_json_path)
            if task_json_path
            else self.data_dir / "tasks_real_world.json"
        )
        with open(self.task_json_path, "r") as f:
            self._tasks: dict[str, dict[str, Any]] = json.load(f)

    def get(self, task_id: int) -> Task:
        key = f"Task {task_id}"
        if key not in self._tasks:
            raise KeyError(f"{key} not found in {self.task_json_path}")
        raw = self._tasks[key]
        return Task(
            task_id=task_id,
            name=raw["name"],
            folder_name=raw["general_folder_name"],
            object_list=raw["object_list"],
            actions=raw["actions"],
            success_condition=raw["success_condition"],
            gt_failure_reason=raw["gt_failure_reason"],
            task_root=self.real_data_dir / raw["general_folder_name"],
            raw=raw,
        )

    def all_task_ids(self) -> list[int]:
        return sorted(
            int(k.split()[1]) for k in self._tasks if k.startswith("Task ")
        )

    def available_folders(self) -> list[str]:
        return sorted(d.name for d in self.real_data_dir.iterdir() if d.is_dir())
