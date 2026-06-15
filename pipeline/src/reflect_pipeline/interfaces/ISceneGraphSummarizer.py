from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SceneGraphInput:
    nodes: list[str]
    edges: list[tuple[str, str, str]]
    metadata: dict[str, Any] | None = None


@dataclass
class SceneSummary:
    l1_summary: str
    l2_summary: str


class SceneGraphSummarizer(ABC):
    @abstractmethod
    def summarize(
        self,
        scene_graph: SceneGraphInput,
    ) -> SceneSummary:
        pass
