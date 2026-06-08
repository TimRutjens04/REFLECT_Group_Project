from __future__ import annotations

import os

import numpy as np

from interfaces import RgbdFrame, RgbdFrameProvider


class AlignedEpisodeFrameProvider(RgbdFrameProvider):
    """
    Serves RgbdFrames from aligned/<ep>.npz + depth_state/<ep>.npz.
    depth is zero-filled when depth_state/ does not yet exist (Georgi's stage).
    """

    def __init__(self, episode_id: str, aligned_dir: str, depth_dir: str) -> None:
        aligned = np.load(os.path.join(aligned_dir, f"{episode_id}.npz"), allow_pickle=False)
        self._frames         = aligned["frames"]          # (N, H, W, 3) uint8
        self._timestamps     = aligned["timestamps"]      # (N,)
        self._failure_labels = aligned["failure_labels"]  # (N,)

        depth_path = os.path.join(depth_dir, f"{episode_id}.npz")
        if os.path.exists(depth_path):
            d = np.load(depth_path, allow_pickle=False)
            self._depths: np.ndarray | None = d["depth"] if "depth" in d else None
        else:
            self._depths = None

        h, w = self._frames[0].shape[:2]
        self._zero_depth = np.zeros((h, w), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._frames)

    def get_frame(self, step_idx: int) -> RgbdFrame:
        depth = (
            self._depths[step_idx]
            if self._depths is not None
            else self._zero_depth
        )
        return RgbdFrame(
            rgb=self._frames[step_idx],
            depth=depth,
            step_idx=step_idx,
            metadata={
                "timestamp":     float(self._timestamps[step_idx]),
                "failure_label": bool(self._failure_labels[step_idx]),
            },
        )
