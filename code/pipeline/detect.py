#!/usr/bin/env python3
"""
Stage 1 — Open-Vocabulary Object Detection (Grounding DINO)

Input:  aligned/<episode>.npz  (frames, timestamps, failure_labels)
        task metadata (object queries) from task.json / tasks_real_world.json
Output: detect/<episode>.npz

npz keys
--------
boxes          (N, MAX_DET, 4)  float32  — [x1, y1, x2, y2] pixel coords
scores         (N, MAX_DET)     float32  — confidence (0-padded beyond n_dets)
label_ids      (N, MAX_DET)     int32    — index into label_vocab (-1 = pad)
n_dets         (N,)             int32    — valid detections per frame
label_vocab    (V,)             str      — object vocabulary for this episode
timestamps     (N,)             float64  — from aligned
failure_labels (N,)             bool     — from aligned
fps_base                        float    — from aligned

Usage
-----
  poetry run python3 code/detect.py                  # all episodes
  poetry run python3 code/detect.py boilWater-1      # specific episode(s)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# cummax is not yet implemented on MPS — enable CPU fallback for unsupported ops
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
ALIGNED_DIR = ROOT / "aligned"
DETECT_DIR = ROOT / "detect"
DATA_DIR = ROOT / "data"
DETECT_DIR.mkdir(exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SCORE_THRESHOLD = 0.25   # box + text confidence threshold
MAX_DET = 20             # max detections stored per frame
BATCH_SIZE = 4           # frames per forward pass

# ── device ─────────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── task metadata helpers ──────────────────────────────────────────────────────

def _load_object_queries(episode_id: str) -> list[str]:
    """Return object name strings for the given episode (from task.json)."""
    # -- simulated episodes: data/<task>/<episode>/task.json --
    for task_dir in sorted(DATA_DIR.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for ep_dir in task_dir.iterdir():
            if not ep_dir.is_dir():
                continue
            # episode folder is named e.g. "boilWater-1" directly
            if ep_dir.name == episode_id:
                task_json = ep_dir / "task.json"
                if task_json.exists():
                    with open(task_json) as f:
                        meta = json.load(f)
                    objs = meta.get("object_list", [])
                    if objs:
                        return [str(o) for o in objs]

    # -- real-world episodes: data/tasks_real_world.json --
    real_meta_path = DATA_DIR / "tasks_real_world.json"
    if real_meta_path.exists():
        with open(real_meta_path) as f:
            real_meta = json.load(f)
        for key, ep in real_meta.items():
            if key == episode_id or ep.get("folder") == episode_id:
                objs = ep.get("object_list", ep.get("objects", []))
                if objs:
                    return [str(o) for o in objs]

    # -- fallback: generic manipulation objects --
    print(f"  [warn] no task metadata for {episode_id}, using generic query")
    return ["object", "cup", "bowl", "apple", "bottle", "box"]


def _build_text_query(objects: list[str]) -> str:
    """Format as Grounding DINO text prompt: 'obj1 . obj2 . obj3 .'"""
    return " . ".join(o.lower().strip() for o in objects) + " ."


# ── model (cached globally) ────────────────────────────────────────────────────
_processor: AutoProcessor | None = None
_model: AutoModelForZeroShotObjectDetection | None = None


def _get_model() -> tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
    global _processor, _model
    if _model is None:
        print(f"Loading Grounding DINO ({MODEL_ID}) on {DEVICE}…")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
        _model.eval()
    return _processor, _model  # type: ignore[return-value]


# ── detection ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def detect_frames(
    frames: np.ndarray,      # (N, H, W, 3) uint8
    text_query: str,
    label_vocab: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Grounding DINO on all frames in batches.

    Returns
    -------
    boxes      (N, MAX_DET, 4)  float32  pixel coords [x1,y1,x2,y2]
    scores     (N, MAX_DET)     float32
    label_ids  (N, MAX_DET)     int32    index into label_vocab (-1 = pad)
    n_dets     (N,)             int32
    """
    processor, model = _get_model()
    N, H, W, _ = frames.shape

    boxes_out = np.zeros((N, MAX_DET, 4), dtype=np.float32)
    scores_out = np.zeros((N, MAX_DET), dtype=np.float32)
    label_ids_out = np.full((N, MAX_DET), -1, dtype=np.int32)
    n_dets_out = np.zeros(N, dtype=np.int32)

    vocab_lower = [v.lower().strip() for v in label_vocab]

    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        batch_pil = [Image.fromarray(frames[i]) for i in range(start, end)]
        texts = [text_query] * len(batch_pil)

        inputs = processor(
            images=batch_pil,
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=SCORE_THRESHOLD,
            text_threshold=SCORE_THRESHOLD,
            target_sizes=[(H, W)] * len(batch_pil),
        )

        for j, result in enumerate(results):
            idx = start + j
            det_boxes = result["boxes"].cpu().numpy()   # (k, 4)
            det_scores = result["scores"].cpu().numpy() # (k,)
            det_labels: list[str] = result["labels"]

            if len(det_scores) == 0:
                continue

            # sort by score descending, clip to MAX_DET
            order = np.argsort(-det_scores)[:MAX_DET]
            det_boxes = det_boxes[order]
            det_scores = det_scores[order]
            det_labels = [det_labels[i] for i in order]

            k = len(det_scores)
            boxes_out[idx, :k] = det_boxes
            scores_out[idx, :k] = det_scores
            n_dets_out[idx] = k

            for m, lbl in enumerate(det_labels):
                lbl_lower = lbl.lower().strip()
                best_idx = -1
                for vi, vl in enumerate(vocab_lower):
                    if lbl_lower in vl or vl in lbl_lower:
                        best_idx = vi
                        break
                label_ids_out[idx, m] = best_idx

    return boxes_out, scores_out, label_ids_out, n_dets_out


# ── per-episode entry point ────────────────────────────────────────────────────

def process_episode(episode_id: str) -> None:
    aligned_path = ALIGNED_DIR / f"{episode_id}.npz"
    out_path = DETECT_DIR / f"{episode_id}.npz"

    if not aligned_path.exists():
        print(f"  [skip] {episode_id}: no aligned file found")
        return
    if out_path.exists():
        print(f"  [skip] {episode_id}: already detected")
        return

    data = np.load(aligned_path, allow_pickle=True)
    frames: np.ndarray = data["frames"]           # (N, H, W, 3)
    timestamps: np.ndarray = data["timestamps"]
    failure_labels: np.ndarray = data["failure_labels"]
    fps_base = float(data["fps_base"])

    objects = _load_object_queries(episode_id)
    text_query = _build_text_query(objects)
    print(f"  query: {text_query!r}")
    print(f"  frames: {len(frames)}, vocab: {objects}")

    boxes, scores, label_ids, n_dets = detect_frames(frames, text_query, objects)

    np.savez_compressed(
        out_path,
        boxes=boxes,
        scores=scores,
        label_ids=label_ids,
        n_dets=n_dets,
        label_vocab=np.array(objects),
        timestamps=timestamps,
        failure_labels=failure_labels,
        fps_base=fps_base,
    )
    total_dets = int(n_dets.sum())
    print(f"  saved {out_path.name} — {total_dets} detections across {len(frames)} frames")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    episodes = sorted(p.stem for p in ALIGNED_DIR.glob("*.npz"))
    if not episodes:
        sys.exit("No aligned episodes found in aligned/.")

    if len(sys.argv) > 1:
        requested = set(sys.argv[1:])
        episodes = [e for e in episodes if e in requested]
        if not episodes:
            sys.exit(f"None of {sys.argv[1:]} found in aligned/.")

    print(
        f"Stage 1 — Grounding DINO detection | "
        f"{len(episodes)} episodes | device={DEVICE}"
    )
    for ep in tqdm(episodes, unit="ep"):
        print(f"\n▶ {ep}")
        process_episode(ep)
    print("\nStage 1 complete.")


if __name__ == "__main__":
    main()
