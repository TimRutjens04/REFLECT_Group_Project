"""Synthetic test scene: coloured shapes YOLOE can find via text prompts.

Verified empirically: YOLOE text-prompt mode detects rendered shapes with
prompts like "red square" / "blue rectangle" / "green circle" at conf 0.6+.

The red square starts left of the blue rectangle and drifts into it, so the
relation evolves left_of/near -> inside over ~50 frames.
"""

from __future__ import annotations

import numpy as np

import cv2

SYNTH_W, SYNTH_H = 640, 480
SYNTH_PROMPTS = ["red square", "blue rectangle", "green circle"]


def synthetic_frame(i: int, w: int = SYNTH_W, h: int = SYNTH_H) -> np.ndarray:
    """Frame i of the synthetic clip (BGR). Mild noise keeps BoTSORT honest."""
    rng = np.random.default_rng(i)
    img = np.full((h, w, 3), (110, 105, 100), dtype=np.uint8)
    noise = rng.integers(-8, 8, (h, w, 3))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # static blue rectangle ("bowl")
    cv2.rectangle(img, (220, 180), (420, 360), (180, 90, 30), -1)
    # red square drifting from the left into the blue rectangle
    x = min(40 + i * 4, 300)
    cv2.rectangle(img, (x, 230), (x + 60, 290), (40, 40, 200), -1)
    # static green circle off to the side
    cv2.circle(img, (540, 140), 40, (60, 180, 60), -1)
    return img


class SyntheticSource:
    def __init__(self, max_frames: int | None = None):
        self._i = 0
        self._max = max_frames

    def read(self) -> np.ndarray | None:
        if self._max is not None and self._i >= self._max:
            return None
        frame = synthetic_frame(self._i)
        self._i += 1
        return frame

    def release(self) -> None:
        pass
