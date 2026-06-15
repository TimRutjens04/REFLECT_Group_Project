from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RgbdFrame:
    rgb: np.ndarray
    depth: np.ndarray
    step_idx: int
    metadata: dict[str, Any] | None = None


class RgbdFrameProvider(ABC):
    @abstractmethod
    def get_frame(self, step_idx: int) -> RgbdFrame:
        pass
