"""Appearance re-identification for keeping stable object ids across re-primes.

Copied from code/reid.py. The pipeline's detector does not expose GDino query
embeddings, so :func:`crop_embedding` is provided as the embedding source: an
L2-normalised HSV colour histogram of the bbox crop, comparable with the same
cosine-similarity logic the matcher already uses.
"""

from __future__ import annotations

import cv2
import numpy as np

# Tuned for the sat-weighted hue embedding below against a FROZEN frame-0
# reference (putAppleBowl1 replay): same-object similarity stays >= ~0.46
# across the whole video (occlusion, in-bowl), different-object similarity
# is ~0.00-0.02. The original 0.85 was tuned for GDino query embeddings.
REID_SIMILARITY_THRESHOLD = 0.40
REID_EMA_ALPHA = 0.05


class ObjectReIdMatcher:
    """
    Stores one reference embedding per object key (e.g. label or vocab index)
    and decides whether a re-detected object is the same one that was lost.

    Embeddings must be L2-normalised before being passed in.
    EMA update keeps the reference fresh as appearance drifts across frames.
    """

    def __init__(
        self,
        similarity_threshold: float = REID_SIMILARITY_THRESHOLD,
        ema_alpha: float = REID_EMA_ALPHA,
    ) -> None:
        self._threshold = similarity_threshold
        self._alpha = ema_alpha
        self._refs: dict[str | int, np.ndarray] = {}

    def register(self, key: str | int, embedding: np.ndarray) -> None:
        """Store initial reference embedding for key (called at seed time)."""
        self._refs[key] = self._normalize(embedding)

    def is_same_object(self, key: str | int, embedding: np.ndarray) -> bool:
        """True if embedding is similar enough to the stored reference."""
        if key not in self._refs:
            return False
        sim = float(np.dot(self._refs[key], self._normalize(embedding)))
        return sim >= self._threshold

    def best_match(
        self, keys, embedding: np.ndarray
    ) -> tuple[str | int | None, float]:
        """Most similar stored reference among ``keys`` that passes the threshold.

        Returns ``(key, similarity)``, or ``(None, best_similarity)`` when no
        candidate reaches the threshold.
        """
        emb = self._normalize(embedding)
        best_key: str | int | None = None
        best_sim = -1.0
        for key in keys:
            ref = self._refs.get(key)
            if ref is None:
                continue
            sim = float(np.dot(ref, emb))
            if sim > best_sim:
                best_key, best_sim = key, sim
        if best_sim < self._threshold:
            return None, best_sim
        return best_key, best_sim

    def update(self, key: str | int, embedding: np.ndarray) -> None:
        """EMA update of reference — call after confirmed same-object re-acquisition."""
        if key not in self._refs:
            self.register(key, embedding)
            return
        updated = (1.0 - self._alpha) * self._refs[key] + self._alpha * self._normalize(
            embedding
        )
        self._refs[key] = self._normalize(updated)

    @staticmethod
    def _normalize(e: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(e)
        return e / n if n > 0 else e


def crop_embedding(
    frame_rgb: np.ndarray,
    bbox_xyxy,
    bins: int = 24,
    center_margin: float = 0.15,
) -> np.ndarray | None:
    """L2-normalised saturation-weighted hue histogram of the bbox crop.

    Cheap appearance descriptor used in place of detector query embeddings,
    chosen for invariance against a frozen seed reference:
      - hue only — robust to the lighting/brightness changes that made full
        HSV histograms drift far from the frame-0 reference over a video;
      - saturation weighting — grey/white pixels (table, specular highlights)
        barely vote, so the object's colour dominates;
      - ``center_margin`` shrinks the box on each side before cropping, which
        keeps background pixels (e.g. the bowl around an apple placed inside
        it) from contaminating the histogram.
    """
    x1, y1, x2, y2 = bbox_xyxy
    dx = (x2 - x1) * center_margin
    dy = (y2 - y1) * center_margin
    h, w = frame_rgb.shape[:2]
    xi1 = int(max(x1 + dx, 0))
    yi1 = int(max(y1 + dy, 0))
    xi2 = int(min(x2 - dx, w))
    yi2 = int(min(y2 - dy, h))
    crop = frame_rgb[yi1:yi2, xi1:xi2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[..., 0].ravel()
    sat = hsv[..., 1].ravel()
    hist, _ = np.histogram(hue, bins=bins, range=(0, 180), weights=sat)
    n = np.linalg.norm(hist)
    if n == 0:
        return None
    return (hist / n).astype(np.float32)
