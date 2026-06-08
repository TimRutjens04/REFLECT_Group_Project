from __future__ import annotations

import numpy as np

REID_SIMILARITY_THRESHOLD = 0.85
REID_EMA_ALPHA = 0.05


class ObjectReIdMatcher:
    """
    Stores one reference embedding per object slot (indexed by vocab position)
    and decides whether a re-detected object is the same one that was lost.

    Embeddings must be L2-normalised before being passed in (detector.py does this).
    EMA update keeps the reference fresh as appearance drifts across frames.
    """

    def __init__(
        self,
        similarity_threshold: float = REID_SIMILARITY_THRESHOLD,
        ema_alpha: float = REID_EMA_ALPHA,
    ) -> None:
        self._threshold = similarity_threshold
        self._alpha = ema_alpha
        self._refs: dict[int, np.ndarray] = {}

    def register(self, obj_idx: int, embedding: np.ndarray) -> None:
        """Store initial reference embedding for obj_idx (called at seed time)."""
        self._refs[obj_idx] = self._normalize(embedding)

    def is_same_object(self, obj_idx: int, embedding: np.ndarray) -> bool:
        """True if embedding is similar enough to the stored reference."""
        if obj_idx not in self._refs:
            return False
        sim = float(np.dot(self._refs[obj_idx], self._normalize(embedding)))
        return sim >= self._threshold

    def update(self, obj_idx: int, embedding: np.ndarray) -> None:
        """EMA update of reference — call after confirmed same-object re-acquisition."""
        if obj_idx not in self._refs:
            self.register(obj_idx, embedding)
            return
        updated = (1.0 - self._alpha) * self._refs[obj_idx] + self._alpha * self._normalize(embedding)
        self._refs[obj_idx] = self._normalize(updated)

    @staticmethod
    def _normalize(e: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(e)
        return e / n if n > 0 else e
