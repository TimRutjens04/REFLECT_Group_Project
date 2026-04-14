#!/usr/bin/env python3
"""
Generate labeled crop dataset from existing AI2-THOR step_N.pickle files.

No live simulator needed — reads the pickles already in data/{task}/{episode}/events/.
Each pickle is an ai2thor.server.Event with:
  - event.frame                  (H, W, 3) uint8 RGB
  - event.instance_detections2D  objectId → (x1, y1, x2, y2)
  - event.metadata["objects"]    list of dicts with capability/state flags

Output: state_classifier/dataset.pkl
  List of dicts, one per labeled object crop:
    {
      "image":       PIL.Image  (cropped object region)
      "object_type": str
      "episode":     str
      "step":        int
      "labels":      dict[int, bool]  pair_idx → True (positive) / False (negative)
      "mask":        list[int]        1 if pair applies to this object, else 0
    }

Usage:
  poetry run python3 code/state_classifier/generate_dataset.py
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from state_classifier.config import (  # noqa: E402
    CAPABILITY_TO_PAIR,
    DATASET_PATH,
    MIN_CROP_PX,
    ROOT,
    STATE_FIELD_TO_PAIR,
)

DATA_DIR = ROOT / "data"
DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)


def _gather_pickle_files() -> list[Path]:
    """Find all step_N.pickle files under data/{task}/{episode}/events/."""
    return sorted(DATA_DIR.glob("*/*/events/step_*.pickle"))


def _step_number(path: Path) -> int:
    m = re.search(r"step_(\d+)\.pickle", path.name)
    return int(m.group(1)) if m else 0


def _episode_id(path: Path) -> str:
    return path.parent.parent.name


def process_pickle(path: Path) -> list[dict]:
    """Extract labeled crops from a single step_N.pickle."""
    with open(path, "rb") as f:
        ev = pickle.load(f)

    frame_rgb = ev.frame           # (H, W, 3) uint8
    H, W = frame_rgb.shape[:2]
    detections = ev.instance_detections2D

    samples = []
    for obj in ev.metadata["objects"]:
        if not obj.get("visible"):
            continue

        obj_id = obj.get("objectId", "")
        if obj_id not in detections:
            continue

        bbox = detections[obj_id]   # (x1, y1, x2, y2) integers
        if bbox is None:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
            continue

        crop = Image.fromarray(frame_rgb[y1:y2, x1:x2])

        # Build mask from capability flags (which state pairs apply to this object)
        mask = [1 if obj.get(cap) else 0 for cap in CAPABILITY_TO_PAIR]

        # Build labels from ground-truth state fields (only for applicable pairs)
        labels: dict[int, bool] = {}
        for field, pair_idx in STATE_FIELD_TO_PAIR.items():
            if mask[pair_idx] and field in obj and obj[field] is not None:
                labels[pair_idx] = bool(obj[field])

        if not any(mask):
            continue  # object has no classifiable state pairs — skip

        samples.append({
            "image":       crop,
            "object_type": obj.get("objectType", "Unknown"),
            "episode":     _episode_id(path),
            "step":        _step_number(path),
            "labels":      labels,
            "mask":        mask,
        })

    return samples


def main() -> None:
    pickle_files = _gather_pickle_files()
    if not pickle_files:
        sys.exit(f"No step_N.pickle files found under {DATA_DIR}")

    print(f"Found {len(pickle_files)} pickle files across "
          f"{len({_episode_id(p) for p in pickle_files})} episodes")

    all_samples: list[dict] = []
    for path in tqdm(pickle_files, unit="pkl", desc="Generating dataset"):
        try:
            all_samples.extend(process_pickle(path))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {path}: {e}")

    print(f"\nTotal labeled crops: {len(all_samples)}")

    # Per-pair positive rate summary
    from state_classifier.config import STATE_PAIRS, N_PAIRS
    pair_pos = [0] * N_PAIRS
    pair_total = [0] * N_PAIRS
    for s in all_samples:
        for pair_idx, is_pos in s["labels"].items():
            pair_total[pair_idx] += 1
            if is_pos:
                pair_pos[pair_idx] += 1

    print("\nPer-pair statistics:")
    for i, (pos_label, neg_label) in enumerate(STATE_PAIRS):
        total = pair_total[i]
        pos   = pair_pos[i]
        if total > 0:
            print(f"  [{i}] {pos_label}/{neg_label}: {total} crops, "
                  f"{pos}/{total} positive ({100*pos/total:.1f}%)")
        else:
            print(f"  [{i}] {pos_label}/{neg_label}: 0 crops")

    with open(DATASET_PATH, "wb") as f:
        pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved → {DATASET_PATH}")


if __name__ == "__main__":
    main()
