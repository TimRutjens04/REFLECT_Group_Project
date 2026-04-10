#!/usr/bin/env python3
"""
Stage 2 — Instance Segmentation (SAM 2)

Input:  aligned/<episode>.npz  (frames)
        detect/<episode>.npz   (boxes, n_dets)
Output: segment/<episode>.npz

Masks stored at 1/4 original resolution to keep file sizes manageable.
SAM 2 is run in image mode per-frame using bounding-box prompts from Stage 1.
The same loaded model is reused in Stage 4 for temporal tracking.

npz keys
--------
masks_small    (N, MAX_DET, H4, W4)  uint8    — binary masks at H//4 × W//4
mask_scores    (N, MAX_DET)          float32  — SAM 2 IoU-predicted score
mask_valid     (N, MAX_DET)          bool     — True for valid (non-pad) masks
timestamps     (N,)                  float64  — from aligned
failure_labels (N,)                  bool     — from aligned
fps_base                             float    — from aligned
orig_hw        (2,)                  int32    — [H, W] of original frames

Usage
-----
  poetry run python3 code/segment.py
  poetry run python3 code/segment.py boilWater-1
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
from transformers import AutoModelForMaskGeneration, AutoProcessor

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
ALIGNED_DIR = ROOT / "aligned"
DETECT_DIR = ROOT / "detect"
SEGMENT_DIR = ROOT / "segment"
SEGMENT_DIR.mkdir(exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_ID = "facebook/sam2-hiera-small"
MASK_SCALE = 4          # store at 1/MASK_SCALE resolution (4 → H//4)
MIN_SCORE = 0.5         # minimum SAM IoU score to keep a mask
MAX_DET = 20            # must match detect.py

# ── device ─────────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── model (cached globally, reused by track.py) ────────────────────────────────
_processor: AutoProcessor | None = None
_model: AutoModelForMaskGeneration | None = None


def get_sam2_model() -> tuple[AutoProcessor, AutoModelForMaskGeneration]:
    """Load SAM 2 once; return cached handles."""
    global _processor, _model
    if _model is None:
        print(f"Loading SAM 2 ({MODEL_ID}) on {DEVICE}…")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForMaskGeneration.from_pretrained(MODEL_ID).to(DEVICE)
        _model.eval()
    return _processor, _model  # type: ignore[return-value]


# ── segmentation helpers ───────────────────────────────────────────────────────

def _xyxy_to_xywh(box: np.ndarray) -> list[float]:
    """Convert [x1, y1, x2, y2] → [x, y, w, h] for SAM 2 processor."""
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


@torch.inference_mode()
def segment_frame(
    image: Image.Image,
    boxes_xyxy: np.ndarray,   # (k, 4) valid boxes for this frame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run SAM 2 on one frame with k bounding-box prompts.

    Returns
    -------
    masks  (k, H, W) bool
    scores (k,)      float32  — SAM IoU-predicted quality
    """
    processor, model = get_sam2_model()

    input_boxes = [_xyxy_to_xywh(b) for b in boxes_xyxy]

    inputs = processor(
        images=image,
        input_boxes=[input_boxes],   # batch dim required
        return_tensors="pt",
    ).to(DEVICE)

    outputs = model(**inputs)

    # Avoid processor.post_process_masks — the SAM 2 video processor forwards
    # args to the image processor in a different order, causing size mismatches.
    # Instead: interpolate the raw logit masks directly to the original resolution.
    #
    # pred_masks : (1, k, num_candidates, H_enc, W_enc)
    # iou_scores : (1, k, num_candidates)
    k = len(input_boxes)
    H, W = image.size[1], image.size[0]

    masks_np = np.zeros((k, H, W), dtype=bool)
    scores_np = np.zeros(k, dtype=np.float32)

    pred_masks = outputs.pred_masks   # (1, k, C, He, We)
    iou_scores = outputs.iou_scores   # (1, k, C)

    for i in range(min(k, pred_masks.shape[1])):
        s = iou_scores[0, i]          # (C,)
        best = int(s.argmax())
        scores_np[i] = float(s[best])

        # logit mask for best candidate: (1, 1, He, We)
        logit = pred_masks[0, i, best].unsqueeze(0).unsqueeze(0).float()
        # bilinear upsample to original frame resolution
        upsampled = torch.nn.functional.interpolate(
            logit, size=(H, W), mode="bilinear", align_corners=False
        )
        masks_np[i] = upsampled.squeeze().cpu().numpy() > 0.0

    return masks_np, scores_np


def _downsample_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    """Downsample boolean mask by integer scale factor using area-max pooling."""
    H, W = mask.shape
    Hs, Ws = H // scale, W // scale
    # reshape + any() for max-pooling semantics
    return (
        mask[:Hs * scale, :Ws * scale]
        .reshape(Hs, scale, Ws, scale)
        .any(axis=(1, 3))
        .astype(np.uint8)
    )


# ── per-episode entry point ────────────────────────────────────────────────────

def process_episode(episode_id: str) -> None:
    aligned_path = ALIGNED_DIR / f"{episode_id}.npz"
    detect_path = DETECT_DIR / f"{episode_id}.npz"
    out_path = SEGMENT_DIR / f"{episode_id}.npz"

    for path, name in [(aligned_path, "aligned"), (detect_path, "detect")]:
        if not path.exists():
            print(f"  [skip] {episode_id}: missing {name} file")
            return
    if out_path.exists():
        print(f"  [skip] {episode_id}: already segmented")
        return

    aligned = np.load(aligned_path, allow_pickle=True)
    det = np.load(detect_path, allow_pickle=True)

    frames: np.ndarray = aligned["frames"]            # (N, H, W, 3)
    timestamps: np.ndarray = aligned["timestamps"]
    failure_labels: np.ndarray = aligned["failure_labels"]
    fps_base = float(aligned["fps_base"])

    all_boxes: np.ndarray = det["boxes"]              # (N, MAX_DET, 4)
    all_n_dets: np.ndarray = det["n_dets"]            # (N,)

    N, H, W, _ = frames.shape
    Hs, Ws = H // MASK_SCALE, W // MASK_SCALE

    masks_small = np.zeros((N, MAX_DET, Hs, Ws), dtype=np.uint8)
    mask_scores = np.zeros((N, MAX_DET), dtype=np.float32)
    mask_valid = np.zeros((N, MAX_DET), dtype=bool)

    processor, _ = get_sam2_model()

    for i in tqdm(range(N), desc=f"  {episode_id}", leave=False, unit="fr"):
        k = int(all_n_dets[i])
        if k == 0:
            continue

        image = Image.fromarray(frames[i])
        boxes_k = all_boxes[i, :k]      # (k, 4)

        masks, scores = segment_frame(image, boxes_k)

        for j in range(k):
            mask_valid[i, j] = True
            mask_scores[i, j] = scores[j]
            masks_small[i, j] = _downsample_mask(masks[j], MASK_SCALE)

    np.savez_compressed(
        out_path,
        masks_small=masks_small,
        mask_scores=mask_scores,
        mask_valid=mask_valid,
        timestamps=timestamps,
        failure_labels=failure_labels,
        fps_base=fps_base,
        orig_hw=np.array([H, W], dtype=np.int32),
    )
    n_valid = int(mask_valid.sum())
    print(f"  saved {out_path.name} — {n_valid} masks, stored at {Hs}×{Ws}")


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
        f"Stage 2 — SAM 2 segmentation | "
        f"{len(episodes)} episodes | device={DEVICE}"
    )
    for ep in tqdm(episodes, unit="ep"):
        print(f"\n▶ {ep}")
        process_episode(ep)
    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
