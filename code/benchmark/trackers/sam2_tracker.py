"""
SAM 2 video-predictor tracker for the benchmark harness.

SAM 2 is an offline tracker: it needs all frames upfront via a video file
and processes them in one shot. It does not fit the per-frame step() interface,
so this module exposes run_video() instead.

The harness calls run_video() separately (passing the original video path) and
folds the predictions into the comparison video alongside step-based trackers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from sam2.build_sam import build_sam2_video_predictor
    sam2_available = True
except ImportError:
    sam2_available = False

_CHECKPOINT = Path(__file__).resolve().parents[3] / "checkpoints" / "sam2.1_hiera_small.pt"
_CONFIG     = "configs/sam2.1/sam2.1_hiera_s.yaml"


class Sam2Tracker:
    """
    SAM 2 offline video tracker.

    Usage:
        tracker = Sam2Tracker()
        preds_per_obj, id_switches = tracker.run_video(video_path, seed_bboxes, n_frames)

    preds_per_obj: list[list[np.ndarray | None]]  — indexed [obj][frame]
    id_switches:   list[int]                        — always 0 (SAM 2 memory bank maintains identity)
    """

    @property
    def name(self) -> str:
        return "SAM 2"

    def __init__(self) -> None:

        if not sam2_available:
            raise RuntimeError("sam2 package not installed")
        if not _CHECKPOINT.exists():
            raise RuntimeError(f"SAM 2 checkpoint not found: {_CHECKPOINT}")

        # SAM 2 MPS has a known pin_memory bug on Apple Silicon — fall back to CPU.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._predictor = build_sam2_video_predictor(
            _CONFIG, str(_CHECKPOINT), device=self._device
        )

    def run_video(
        self,
        frames: list[np.ndarray],
        seed_bboxes: list[np.ndarray],
    ) -> tuple[list[list[np.ndarray | None]], list[int]]:
        """
        Track all objects through all frames.

        frames:      BGR frames already loaded into memory.
        seed_bboxes: one xyxy float32 bbox per object, drawn on frame 0.
        """
        n_obj    = len(seed_bboxes)
        n_frames = len(frames)

        with tempfile.TemporaryDirectory() as tmp:
            _write_jpegs(frames, tmp)

            with torch.inference_mode():
                state = self._predictor.init_state(
                    tmp, offload_video_to_cpu=True, offload_state_to_cpu=True
                )

                for obj_id, bbox in enumerate(seed_bboxes, start=1):
                    self._predictor.add_new_points_or_box(
                        state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=bbox.astype(np.float32),
                    )

                raw: dict[int, dict[int, np.ndarray | None]] = {
                    oid: {} for oid in range(1, n_obj + 1)
                }

                for fi, obj_ids, masks in self._predictor.propagate_in_video(state):
                    # obj_ids: list[int], masks: (N, 1, H, W) tensor
                    for i, oid in enumerate(obj_ids):
                        raw[oid][fi] = _mask_to_xyxy(masks[i])

                self._predictor.reset_state(state)

        preds_per_obj: list[list[np.ndarray | None]] = []
        for oid in range(1, n_obj + 1):
            preds_per_obj.append([raw[oid].get(fi) for fi in range(n_frames)])

        return preds_per_obj, [0] * n_obj


def _write_jpegs(frames: list[np.ndarray], directory: str) -> None:
    """Write frames as zero-padded JPEGs so SAM 2 can load them as a JPEG folder."""
    for i, f in enumerate(frames):
        cv2.imwrite(str(Path(directory) / f"{i:05d}.jpg"), f)


def _mask_to_xyxy(mask: torch.Tensor) -> np.ndarray | None:
    """Convert a (1, H, W) bool mask to xyxy bbox, or None if mask is empty."""
    m = mask.squeeze(0).cpu().numpy() > 0
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
