#!/usr/bin/env python3
"""
Stage 4 — Temporal State Tracking

Input:  detect/<episode>.npz    (boxes, n_dets, label_ids, label_vocab)
        segment/<episode>.npz   (masks_small, mask_valid, orig_hw)
Output: track/<episode>.npz

Tracking algorithm
------------------
Greedy IoU matching between consecutive frames assigns a persistent track_id
to each detection. A new ID is allocated when no prior mask overlaps the current
mask above IOU_THRESHOLD.

held_by_gripper heuristic
--------------------------
An object is flagged as held_by_gripper when its centroid (in the small-mask
coordinate space) has been moving with consistent velocity across the last
N_FRAME_WINDOW frames AND the displacement magnitude exceeds HELD_MOVE_THRESHOLD.
This captures "object is being carried by the robot arm" without needing a
dedicated gripper segmentation mask.

npz keys
--------
track_ids            (N, MAX_DET)  int32   — persistent ID (-1 = no detection)
tracking_confidence  (N, MAX_DET)  float32 — IoU with matched prior mask (1.0 = new track)
held_by_gripper      (N, MAX_DET)  bool    — centroid-motion held heuristic
id_switch_frames     (K,)          int32   — frame indices where ≥1 ID switch detected
timestamps           (N,)          float64 — from aligned
failure_labels       (N,)          bool    — from aligned
fps_base                           float   — from aligned

Configurable thresholds (all in this file header — never hardcoded inline)
--------------------------------------------------------------------------
IOU_THRESHOLD        float  mask IoU required to associate a detection to a prior track
N_FRAME_WINDOW       int    look-back window for centroid velocity estimation
HELD_MOVE_THRESHOLD  float  min centroid displacement (px in small-mask space) per frame
                            to consider an object as moving / being held
HELD_CONSISTENCY     float  0–1 cosine similarity threshold for velocity direction
                            consistency to flag held_by_gripper

Usage
-----
  poetry run python3 code/track.py
  poetry run python3 code/track.py boilWater-1
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DETECT_DIR = ROOT / "detect"
SEGMENT_DIR = ROOT / "segment"
ALIGNED_DIR = ROOT / "aligned"
TRACK_DIR = ROOT / "track"
TRACK_DIR.mkdir(exist_ok=True)

# ── configurable thresholds ────────────────────────────────────────────────────
IOU_THRESHOLD = 0.40       # min mask IoU to associate detection to existing track
N_FRAME_WINDOW = 3         # frames of history for velocity estimation
HELD_MOVE_THRESHOLD = 3.0  # min displacement (px, small-mask coords) per frame
HELD_CONSISTENCY = 0.80    # cosine similarity threshold for direction consistency
MAX_DET = 20               # must match detect.py / segment.py


# ── IoU utilities ──────────────────────────────────────────────────────────────

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two binary masks of the same shape."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    """Return (row, col) centroid of a binary mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return np.array([ys.mean(), xs.mean()], dtype=np.float32)


# ── greedy IoU tracker ─────────────────────────────────────────────────────────

class GreedyTracker:
    """
    Assigns persistent track IDs to per-frame detections using greedy IoU matching.

    State per track
    ---------------
    track_mask : most recent mask (small coords)
    track_age  : frames since track was last seen
    history    : deque of (frame_idx, centroid) for velocity estimation
    """

    def __init__(self) -> None:
        self._next_id = 0
        self._tracks: dict[int, dict] = {}   # track_id → state dict

    def update(
        self,
        frame_idx: int,
        masks: np.ndarray,     # (k, Hs, Ws) bool — valid masks this frame
        n_dets: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Match `n_dets` current detections to existing tracks.

        Returns
        -------
        track_ids   (MAX_DET,) int32   — track ID per slot (-1 = pad)
        match_ious  (MAX_DET,) float32 — IoU with matched prior mask
        """
        k = n_dets
        track_ids = np.full(MAX_DET, -1, dtype=np.int32)
        match_ious = np.zeros(MAX_DET, dtype=np.float32)

        if k == 0:
            # age all existing tracks
            for tid in list(self._tracks.keys()):
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > 5:
                    del self._tracks[tid]
            return track_ids, match_ious

        # build IoU matrix: active_tracks × current_dets
        active_ids = list(self._tracks.keys())
        n_tracks = len(active_ids)
        iou_matrix = np.zeros((n_tracks, k), dtype=np.float32)

        for ti, tid in enumerate(active_ids):
            prior_mask = self._tracks[tid]["mask"]
            for di in range(k):
                iou_matrix[ti, di] = _mask_iou(
                    prior_mask.astype(bool), masks[di].astype(bool)
                )

        # greedy matching: repeatedly take highest-IoU pair
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        det_to_track: dict[int, int] = {}
        det_to_iou: dict[int, float] = {}

        if n_tracks > 0:
            flat_order = np.argsort(-iou_matrix.ravel())
            for flat_idx in flat_order:
                ti, di = divmod(int(flat_idx), k)
                iou_val = iou_matrix[ti, di]
                if iou_val < IOU_THRESHOLD:
                    break
                if ti in assigned_tracks or di in assigned_dets:
                    continue
                det_to_track[di] = active_ids[ti]
                det_to_iou[di] = iou_val
                assigned_tracks.add(ti)
                assigned_dets.add(di)

        # assign IDs and update tracks
        for di in range(k):
            if di in det_to_track:
                tid = det_to_track[di]
                iou_val = det_to_iou[di]
            else:
                # new track
                tid = self._next_id
                self._next_id += 1
                iou_val = 1.0   # new track gets 1.0 sentinel

            centroid = _mask_centroid(masks[di].astype(bool))
            history = self._tracks.get(tid, {}).get("history", [])
            history = (history + [(frame_idx, centroid)])[-N_FRAME_WINDOW - 1:]

            self._tracks[tid] = {
                "mask": masks[di],
                "age": 0,
                "history": history,
            }
            track_ids[di] = tid
            match_ious[di] = iou_val

        # age unmatched tracks
        for tid in list(self._tracks.keys()):
            if tid not in [det_to_track.get(di, -99) for di in range(k)]:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > 5:
                    del self._tracks[tid]

        return track_ids, match_ious

    def get_centroid_history(self, tid: int) -> list[tuple[int, np.ndarray | None]]:
        if tid in self._tracks:
            return self._tracks[tid]["history"]
        return []


# ── held_by_gripper estimation ─────────────────────────────────────────────────

def _estimate_held(
    tid: int,
    tracker: GreedyTracker,
) -> bool:
    """
    Return True if centroid velocity over N_FRAME_WINDOW frames is:
    - large enough (> HELD_MOVE_THRESHOLD per frame) AND
    - consistent in direction (cosine similarity > HELD_CONSISTENCY)

    This captures objects being actively carried without a gripper mask.
    """
    history = tracker.get_centroid_history(tid)
    # need at least 2 valid centroids
    valid = [(fi, c) for fi, c in history if c is not None]
    if len(valid) < 2:
        return False

    # compute pairwise displacements between consecutive valid entries
    displacements = []
    for i in range(len(valid) - 1):
        delta = valid[i + 1][1] - valid[i][0 if False else 1]  # (row_delta, col_delta)
        displacements.append(delta)

    if not displacements:
        return False

    disps = np.array(displacements, dtype=np.float32)  # (m, 2)
    magnitudes = np.linalg.norm(disps, axis=1)         # (m,)

    if magnitudes.mean() < HELD_MOVE_THRESHOLD:
        return False

    # check directional consistency
    if len(disps) < 2:
        return True    # single displacement but above threshold — treat as held

    norms = disps / (magnitudes[:, None] + 1e-8)
    # pairwise cosine similarities
    cos_sims = []
    for i in range(len(norms) - 1):
        cos_sims.append(float(norms[i] @ norms[i + 1]))
    return float(np.mean(cos_sims)) >= HELD_CONSISTENCY


# ── per-episode entry point ────────────────────────────────────────────────────

def process_episode(episode_id: str) -> None:
    aligned_path = ALIGNED_DIR / f"{episode_id}.npz"
    detect_path = DETECT_DIR / f"{episode_id}.npz"
    segment_path = SEGMENT_DIR / f"{episode_id}.npz"
    out_path = TRACK_DIR / f"{episode_id}.npz"

    for path, name in [
        (aligned_path, "aligned"),
        (detect_path, "detect"),
        (segment_path, "segment"),
    ]:
        if not path.exists():
            print(f"  [skip] {episode_id}: missing {name} file")
            return
    if out_path.exists():
        print(f"  [skip] {episode_id}: already tracked")
        return

    aligned = np.load(aligned_path, allow_pickle=True)
    det = np.load(detect_path, allow_pickle=True)
    seg = np.load(segment_path, allow_pickle=True)

    timestamps: np.ndarray = aligned["timestamps"]
    failure_labels: np.ndarray = aligned["failure_labels"]
    fps_base = float(aligned["fps_base"])

    all_n_dets: np.ndarray = det["n_dets"]               # (N,)
    masks_small: np.ndarray = seg["masks_small"]         # (N, MAX_DET, Hs, Ws)
    mask_valid: np.ndarray = seg["mask_valid"]           # (N, MAX_DET)

    N = len(timestamps)
    track_ids_out = np.full((N, MAX_DET), -1, dtype=np.int32)
    track_conf_out = np.zeros((N, MAX_DET), dtype=np.float32)
    held_out = np.zeros((N, MAX_DET), dtype=bool)
    id_switch_frames: list[int] = []

    tracker = GreedyTracker()
    prev_track_ids = np.full(MAX_DET, -1, dtype=np.int32)

    for i in tqdm(range(N), desc=f"  {episode_id}", leave=False, unit="fr"):
        k = int(all_n_dets[i])
        # collect valid masks for this frame
        valid_masks = np.zeros((k, *masks_small.shape[2:]), dtype=np.uint8)
        for j in range(k):
            if mask_valid[i, j]:
                valid_masks[j] = masks_small[i, j]

        tids, confs = tracker.update(i, valid_masks, k)
        track_ids_out[i] = tids
        track_conf_out[i] = confs

        # detect ID switches: same detection slot got a different track ID
        for j in range(k):
            if prev_track_ids[j] != -1 and tids[j] != -1:
                if prev_track_ids[j] != tids[j]:
                    id_switch_frames.append(i)
                    break

        # held_by_gripper
        for j in range(k):
            tid = int(tids[j])
            if tid >= 0:
                held_out[i, j] = _estimate_held(tid, tracker)

        prev_track_ids = tids.copy()

    id_switch_arr = np.array(sorted(set(id_switch_frames)), dtype=np.int32)

    np.savez_compressed(
        out_path,
        track_ids=track_ids_out,
        tracking_confidence=track_conf_out,
        held_by_gripper=held_out,
        id_switch_frames=id_switch_arr,
        timestamps=timestamps,
        failure_labels=failure_labels,
        fps_base=fps_base,
    )
    n_switches = len(id_switch_arr)
    n_held = int(held_out.sum())
    print(
        f"  saved {out_path.name} — "
        f"{n_switches} ID-switch frames, {n_held} held-by-gripper detections"
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    episodes = sorted(p.stem for p in ALIGNED_DIR.glob("*.npz"))
    if not episodes:
        sys.exit("No aligned episodes found.")

    if len(sys.argv) > 1:
        requested = set(sys.argv[1:])
        episodes = [e for e in episodes if e in requested]
        if not episodes:
            sys.exit(f"None of {sys.argv[1:]} found.")

    print(
        f"Stage 4 — Temporal tracking | "
        f"{len(episodes)} episodes | IOU_THRESHOLD={IOU_THRESHOLD}"
    )
    for ep in tqdm(episodes, unit="ep"):
        print(f"\n▶ {ep}")
        process_episode(ep)
    print("\nStage 4 complete.")


if __name__ == "__main__":
    main()
