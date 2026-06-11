from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import json


@dataclass
class FrameBase:
    """
    Shared keys for every JSONL row.

    Every pipeline module should inherit from this so rows can be merged on:
    sequence_id, frame_id, timestamp.
    """

    sequence_id: str
    frame_id: int
    timestamp: float


@dataclass
class BBoxXYXY:
    """
    Bounding box in pixel coordinates: [x1, y1, x2, y2].
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


def to_jsonable(value: Any) -> Any:
    """
    Convert dataclasses and enums into plain JSON-compatible values.
    """

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]

    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}

    return value


class JsonlWriter:
    """
    Append one dataclass row as one JSON object per line.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: FrameBase) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")

    def write_many(self, rows: list[FrameBase]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


class JsonFrameWriter:
    """
    Write one JSON file per frame to a directory: frame_{id:04d}.json.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, row: FrameBase) -> None:
        path = self.output_dir / f"frame_{row.frame_id:04d}.json"
        path.write_text(json.dumps(to_jsonable(row), ensure_ascii=False, indent=2))
