#!/usr/bin/env python3
"""
Stage 3 — Depth Estimation (Depth Anything V2) + Object State (SigLIP 2)

Input:  aligned/<episode>.npz   (frames)
        segment/<episode>.npz   (masks_small, mask_valid, orig_hw)
        detect/<episode>.npz    (boxes, n_dets, label_vocab)
Output: depth_state/<episode>.npz

Depth Anything V2 produces *relative* (not metric) depth maps.
SigLIP 2 classifies each masked object crop against a fixed set of state labels.

npz keys
--------
depth_maps     (N, H2, W2)         float32  — relative depth at H//2 × W//2
obj_depth      (N, MAX_DET)        float32  — mean depth within each object mask
obj_state      (N, MAX_DET)        int32    — index into STATE_VOCAB (-1 = no mask)
obj_state_prob (N, MAX_DET)        float32  — sigmoid probability of chosen state
timestamps     (N,)                float64  — from aligned
failure_labels (N,)                bool     — from aligned
fps_base                           float    — from aligned
state_vocab    (S,)                str      — e.g. ["open","closed","held","free",...]

Usage
-----
  poetry run python3 code/depth_state.py
  poetry run python3 code/depth_state.py boilWater-1
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    AutoProcessor,
    AutoModel,
)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
ALIGNED_DIR = ROOT / "aligned"
DETECT_DIR = ROOT / "detect"
SEGMENT_DIR = ROOT / "segment"
DEPTH_STATE_DIR = ROOT / "depth_state"
DEPTH_STATE_DIR.mkdir(exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
SIGLIP_MODEL_IDS = [
    "google/siglip-base-patch16-224",      # ~200M params, sufficient for state classification
]
STATE_VOCAB = ["open", "closed", "held", "free", "full", "empty", "on", "off"]
DEPTH_SCALE = 2      # store depth at 1/2 resolution
MAX_DET = 20         # must match detect.py / segment.py
MIN_CROP_PX = 8      # skip crops smaller than this in either dim

# ── device ─────────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── model handles (cached) ─────────────────────────────────────────────────────
_depth_proc: AutoImageProcessor | None = None
_depth_model: AutoModelForDepthEstimation | None = None
_sig_proc: AutoProcessor | None = None
_sig_model: AutoModel | None = None


def _get_depth_model() -> tuple[AutoImageProcessor, AutoModelForDepthEstimation]:
    global _depth_proc, _depth_model
    if _depth_model is None:
        print(f"Loading Depth Anything V2 ({DEPTH_MODEL_ID}) on {DEVICE}…")
        _depth_proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        _depth_model = AutoModelForDepthEstimation.from_pretrained(
            DEPTH_MODEL_ID
        ).to(DEVICE)
        _depth_model.eval()
    return _depth_proc, _depth_model  # type: ignore[return-value]


def _get_siglip_model() -> tuple[AutoProcessor, AutoModel]:
    global _sig_proc, _sig_model
    if _sig_model is None:
        loaded = False
        for model_id in SIGLIP_MODEL_IDS:
            try:
                print(f"Loading SigLIP ({model_id}) on {DEVICE}…")
                _sig_proc = AutoProcessor.from_pretrained(model_id)
                _sig_model = AutoModel.from_pretrained(model_id).to(DEVICE)
                _sig_model.eval()
                loaded = True
                break
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] could not load {model_id}: {e}")
        if not loaded:
            raise RuntimeError("No SigLIP model available. Check your internet connection.")
    return _sig_proc, _sig_model  # type: ignore[return-value]


# ── depth estimation ───────────────────────────────────────────────────────────

@torch.inference_mode()
def estimate_depth(frame: np.ndarray) -> np.ndarray:
    """
    Run Depth Anything V2 on a single RGB frame.

    Returns a float32 relative depth map at H//DEPTH_SCALE × W//DEPTH_SCALE.
    Larger values = further away (normalized 0–1 per frame).
    """
    proc, model = _get_depth_model()
    image = Image.fromarray(frame)
    inputs = proc(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    # interpolate to original resolution then downsample for storage
    H, W, _ = frame.shape
    pred = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze()                          # (H, W)
    depth_np = pred.cpu().numpy().astype(np.float32)
    # normalize per frame to [0, 1]
    dmin, dmax = depth_np.min(), depth_np.max()
    if dmax > dmin:
        depth_np = (depth_np - dmin) / (dmax - dmin)
    # downsample for storage
    Hs, Ws = H // DEPTH_SCALE, W // DEPTH_SCALE
    depth_small = cv2.resize(depth_np, (Ws, Hs), interpolation=cv2.INTER_AREA)
    return depth_small


def _mask_mean_depth(depth_full: np.ndarray, mask_small: np.ndarray) -> float:
    """
    Compute mean depth value within an object mask.

    depth_full is at H//DEPTH_SCALE × W//DEPTH_SCALE (same as mask_small at //4 → not same).
    We resize mask_small to depth_full size before computing.
    """
    Hd, Wd = depth_full.shape
    if mask_small.shape != (Hd, Wd):
        mask_rs = cv2.resize(
            mask_small.astype(np.float32), (Wd, Hd), interpolation=cv2.INTER_NEAREST
        )
        mask_rs = mask_rs > 0.5
    else:
        mask_rs = mask_small.astype(bool)
    if mask_rs.sum() == 0:
        return float("nan")
    return float(depth_full[mask_rs].mean())


# ── object state classification ────────────────────────────────────────────────

@torch.inference_mode()
def classify_state(
    crops: list[Image.Image],   # object image crops
    state_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score each crop against state_labels with SigLIP 2 (sigmoid zero-shot).

    Returns
    -------
    state_ids   (k,)      int32   index of highest-sigmoid state
    state_probs (k,)      float32 sigmoid probability of chosen state
    """
    proc, model = _get_siglip_model()
    k = len(crops)
    state_ids = np.full(k, -1, dtype=np.int32)
    state_probs = np.zeros(k, dtype=np.float32)

    if k == 0:
        return state_ids, state_probs

    # Batch all (crop, state) pairs
    all_inputs = proc(
        text=state_labels,
        images=crops,
        return_tensors="pt",
        padding="max_length",
    ).to(DEVICE)

    outputs = model(**all_inputs)
    # logits_per_image: (num_crops, num_labels) — sigmoid for independent scoring
    logits = outputs.logits_per_image        # (k, S)
    probs = torch.sigmoid(logits).cpu().numpy()   # (k, S)

    for i in range(k):
        best = int(np.argmax(probs[i]))
        state_ids[i] = best
        state_probs[i] = float(probs[i, best])

    return state_ids, state_probs


def _extract_crop(frame: np.ndarray, box_xyxy: np.ndarray) -> Image.Image | None:
    """Crop frame to bounding box with basic bounds checking."""
    H, W, _ = frame.shape
    x1 = max(0, int(box_xyxy[0]))
    y1 = max(0, int(box_xyxy[1]))
    x2 = min(W, int(box_xyxy[2]))
    y2 = min(H, int(box_xyxy[3]))
    if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
        return None
    return Image.fromarray(frame[y1:y2, x1:x2])


# ── per-episode entry point ────────────────────────────────────────────────────

def process_episode(episode_id: str) -> None:
    aligned_path = ALIGNED_DIR / f"{episode_id}.npz"
    detect_path = DETECT_DIR / f"{episode_id}.npz"
    segment_path = SEGMENT_DIR / f"{episode_id}.npz"
    out_path = DEPTH_STATE_DIR / f"{episode_id}.npz"

    for path, name in [
        (aligned_path, "aligned"),
        (detect_path, "detect"),
        (segment_path, "segment"),
    ]:
        if not path.exists():
            print(f"  [skip] {episode_id}: missing {name} file")
            return
    if out_path.exists():
        print(f"  [skip] {episode_id}: already processed")
        return

    aligned = np.load(aligned_path, allow_pickle=True)
    det = np.load(detect_path, allow_pickle=True)
    seg = np.load(segment_path, allow_pickle=True)

    frames: np.ndarray = aligned["frames"]           # (N, H, W, 3)
    timestamps: np.ndarray = aligned["timestamps"]
    failure_labels: np.ndarray = aligned["failure_labels"]
    fps_base = float(aligned["fps_base"])

    all_boxes: np.ndarray = det["boxes"]             # (N, MAX_DET, 4)
    all_n_dets: np.ndarray = det["n_dets"]           # (N,)

    masks_small: np.ndarray = seg["masks_small"]     # (N, MAX_DET, Hs, Ws)
    mask_valid: np.ndarray = seg["mask_valid"]       # (N, MAX_DET)

    N, H, W, _ = frames.shape
    Hs, Ws = H // DEPTH_SCALE, W // DEPTH_SCALE

    depth_maps = np.zeros((N, Hs, Ws), dtype=np.float32)
    obj_depth = np.full((N, MAX_DET), np.nan, dtype=np.float32)
    obj_state = np.full((N, MAX_DET), -1, dtype=np.int32)
    obj_state_prob = np.zeros((N, MAX_DET), dtype=np.float32)

    for i in tqdm(range(N), desc=f"  {episode_id}", leave=False, unit="fr"):
        # --- depth ---
        depth_maps[i] = estimate_depth(frames[i])

        k = int(all_n_dets[i])
        if k == 0:
            continue

        # --- per-object depth from mask ---
        for j in range(k):
            if not mask_valid[i, j]:
                continue
            obj_depth[i, j] = _mask_mean_depth(depth_maps[i], masks_small[i, j])

        # --- object state via SigLIP 2 ---
        crops = []
        valid_js = []
        for j in range(k):
            crop = _extract_crop(frames[i], all_boxes[i, j])
            if crop is not None:
                crops.append(crop)
                valid_js.append(j)

        if crops:
            state_ids, state_probs = classify_state(crops, STATE_VOCAB)
            for m, j in enumerate(valid_js):
                obj_state[i, j] = state_ids[m]
                obj_state_prob[i, j] = state_probs[m]

    np.savez_compressed(
        out_path,
        depth_maps=depth_maps,
        obj_depth=obj_depth,
        obj_state=obj_state,
        obj_state_prob=obj_state_prob,
        timestamps=timestamps,
        failure_labels=failure_labels,
        fps_base=fps_base,
        state_vocab=np.array(STATE_VOCAB),
    )
    valid_depths = int((~np.isnan(obj_depth)).sum())
    print(f"  saved {out_path.name} — depth maps {Hs}×{Ws}, {valid_depths} object depths")


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
        f"Stage 3 — Depth Anything V2 + SigLIP 2 | "
        f"{len(episodes)} episodes | device={DEVICE}"
    )
    for ep in tqdm(episodes, unit="ep"):
        print(f"\n▶ {ep}")
        process_episode(ep)
    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
