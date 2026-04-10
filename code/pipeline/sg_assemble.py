#!/usr/bin/env python3
"""
Stage 5 — Scene Graph Assembly (Rule-Based)

Input:  detect/<episode>.npz       (boxes, n_dets, label_vocab, label_ids)
        segment/<episode>.npz      (masks_small, mask_valid, orig_hw)
        depth_state/<episode>.npz  (obj_depth, obj_state, state_vocab)
        track/<episode>.npz        (track_ids, tracking_confidence, held_by_gripper)
        aligned/<episode>.npz      (timestamps, failure_labels)
Output: scene_graphs_pipeline/<episode>.json

Scene graph format (per-frame list, matches CLAUDE.md schema)
-------------------------------------------------------------
{
  "episode": "boilWater-1",
  "frames": [
    {
      "frame_idx": 3,
      "timestamp": 3.0,
      "failure_label": true,
      "objects": [
        {
          "id": 42,            ← persistent track_id
          "label": "apple",
          "bbox": [x1,y1,x2,y2],
          "depth": 0.42,       ← normalized relative depth [0,1]
          "state": "free",
          "tracking_confidence": 0.94,
          "held_by_gripper": false
        }
      ],
      "spatial_relations": [
        {"subject": 42, "relation": "inside", "object": 7}
      ],
      "localization_flag": {
        "failure_detected": true,
        "type": "Wrong_object",
        "affected_object_ids": [42]
      }
    }
  ]
}

Spatial relation rules (all thresholds configurable below)
-----------------------------------------------------------
above / below     : depth centroid comparison along depth axis (± DEPTH_EPS)
on_top_of         : centroid above AND 2D bbox overlap > ON_TOP_IOU
inside            : A's bbox is mostly contained in B's bbox (containment IoU > INSIDE_IOU)
                    AND depth centroids within INSIDE_DEPTH_RANGE
left_of / right_of: relative X centroid in image coords
near              : 2D Euclidean centroid distance < NEAR_DIST_PX (normalized)
held_by_gripper   : from Stage 4 track output

Localization flag logic
-----------------------
Wrong_object  : a track_id appears in a failure frame that was NOT detected in
                the immediately preceding non-failure frames (novel object at failure)
No_Grasp      : held_by_gripper expected (failure type implies grasping) but False
Slip          : held_by_gripper transitions True→False mid-trajectory
Translation   : object depth changes > TRANSLATION_DEPTH_THRESH between consecutive
                frames (large unexpected displacement)

Usage
-----
  poetry run python3 code/sg_assemble.py
  poetry run python3 code/sg_assemble.py boilWater-1
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
ALIGNED_DIR = ROOT / "aligned"
DETECT_DIR = ROOT / "detect"
SEGMENT_DIR = ROOT / "segment"
DEPTH_STATE_DIR = ROOT / "depth_state"
TRACK_DIR = ROOT / "track"
PIPELINE_GRAPHS_DIR = ROOT / "scene_graphs_pipeline"
PIPELINE_GRAPHS_DIR.mkdir(exist_ok=True)

# ── configurable spatial-relation thresholds ───────────────────────────────────
DEPTH_EPS = 0.10             # depth diff threshold for above/below
ON_TOP_IOU = 0.25            # 2D bbox IoU threshold for on_top_of
INSIDE_IOU = 0.70            # containment IoU: fraction of A's bbox inside B
INSIDE_DEPTH_RANGE = 0.20    # max depth diff for inside relation
NEAR_DIST_PX = 0.15          # normalized centroid distance (fraction of frame width)
HELD_IOU_THRESHOLD = 0.30    # not used directly — held comes from Stage 4

# ── localization thresholds ────────────────────────────────────────────────────
TRANSLATION_DEPTH_THRESH = 0.15   # depth change > this flags Translation
SLIP_LOOKBACK = 15                 # max frames back to search for a prior held=True event
LARGE_OBJ_HELD_THRESH = 0.30      # bbox fraction of frame area above which held is forced False
MAX_DET = 20                       # must match upstream stages

# ── relation count control ─────────────────────────────────────────────────────
NEAR_TOP_K = 2                     # max "near" relations emitted per subject object


# ── spatial relation helpers ───────────────────────────────────────────────────

def _bbox_centroid(box: list[float]) -> tuple[float, float]:
    """Return (cx, cy) in pixel space from [x1,y1,x2,y2]."""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _bbox_iou_2d(a: list[float], b: list[float]) -> float:
    """Standard 2D bbox IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _containment_iou(inner: list[float], outer: list[float]) -> float:
    """Fraction of `inner` bbox that overlaps `outer` (not symmetric)."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_inner = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area_inner if area_inner > 0 else 0.0


def compute_spatial_relations(
    objects: list[dict],
    frame_width: float,
) -> list[dict]:
    """
    Apply rule-based spatial relation rules from CLAUDE.md to a list of object dicts.

    Each object dict has keys: id, bbox [x1,y1,x2,y2], depth, held_by_gripper.
    Returns list of {"subject": id, "relation": str, "object": id | "gripper"}.
    """
    relations: list[dict] = []
    n = len(objects)

    for i in range(n):
        oi = objects[i]
        # held_by_gripper
        if oi.get("held_by_gripper", False):
            relations.append({
                "subject": oi["id"],
                "relation": "held_by_gripper",
                "object": "gripper",
            })

        for j in range(n):
            if i == j:
                continue
            oj = objects[j]

            bi = oi["bbox"]
            bj = oj["bbox"]
            di = oi.get("depth")
            dj = oj.get("depth")
            cxi, cyi = _bbox_centroid(bi)
            cxj, cyj = _bbox_centroid(bj)

            # ── above / below (depth-based) ──────────────────────────────────
            if di is not None and dj is not None and not (np.isnan(di) or np.isnan(dj)):
                depth_diff = di - dj   # smaller depth = closer to camera
                if depth_diff > DEPTH_EPS:
                    # oi is further from camera than oj → oi is "behind" oj
                    # but in robot manipulation: larger Y in image = lower position
                    pass
                # Use image Y for above/below (lower Y value = higher in image = above)
                if cyi < cyj - 10:    # oi centroid is higher in the image
                    if abs(di - dj) < DEPTH_EPS * 3:  # roughly same depth plane
                        iou_2d = _bbox_iou_2d(bi, bj)
                        if iou_2d > ON_TOP_IOU:
                            relations.append({
                                "subject": oi["id"],
                                "relation": "on_top_of",
                                "object": oj["id"],
                            })
                        else:
                            relations.append({
                                "subject": oi["id"],
                                "relation": "above",
                                "object": oj["id"],
                            })

            # ── inside (containment) ──────────────────────────────────────────
            contain = _containment_iou(bi, bj)
            if contain > INSIDE_IOU:
                depth_ok = True
                if di is not None and dj is not None:
                    depth_ok = not (np.isnan(di) or np.isnan(dj))
                    if depth_ok:
                        depth_ok = abs(di - dj) < INSIDE_DEPTH_RANGE
                if depth_ok:
                    relations.append({
                        "subject": oi["id"],
                        "relation": "inside",
                        "object": oj["id"],
                    })

    # ── near: top-K closest pairs per subject (separate pass) ────────────────
    for i in range(n):
        oi = objects[i]
        cxi, cyi = _bbox_centroid(oi["bbox"])
        pair_dists: list[tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            oj = objects[j]
            cxj, cyj = _bbox_centroid(oj["bbox"])
            d = np.sqrt((cxi - cxj) ** 2 + (cyi - cyj) ** 2) / frame_width
            pair_dists.append((d, oj["id"]))
        pair_dists.sort(key=lambda x: x[0])
        emitted = 0
        for d, other_id in pair_dists:
            if emitted >= NEAR_TOP_K or d >= NEAR_DIST_PX:
                break
            relations.append({"subject": oi["id"], "relation": "near", "object": other_id})
            emitted += 1

    return relations


# ── localization flag logic ────────────────────────────────────────────────────

def compute_localization_flag(
    frame_idx: int,
    failure_label: bool,
    objects: list[dict],
    track_ids_history: dict[int, list[int]],   # track_id → list of frame_indices seen
    held_history: dict[str, list[bool]],        # label → per-frame held flags (all frames so far)
    prev_depth: dict[int, float],              # track_id → depth at frame_idx-1
    task_vocab: set[str],                      # expected labels for this episode
) -> dict:
    """
    Compute the localization_flag for a frame based on failure taxonomy.

    Priority order (highest-confidence first):
      Slip → Translation → Wrong_object → No_Grasp

    Returns {"failure_detected": bool, "type": str | None, "affected_object_ids": list}
    """
    if not failure_label:
        return {"failure_detected": False, "type": None, "affected_object_ids": []}

    affected: list[int] = []
    failure_type: str | None = None

    for obj in objects:
        tid = obj["id"]
        lbl = obj["label"]
        if tid < 0:
            continue

        # ── Slip: label was held recently but is not held now ─────────────────
        label_hist = held_history.get(lbl, [])
        currently_held = label_hist[-1] if label_hist else False
        lookback = label_hist[-(SLIP_LOOKBACK + 1):-1]
        if len(label_hist) >= 2 and any(lookback) and not currently_held:
            if failure_type is None:
                failure_type = "Slip"
            affected.append(tid)
            continue

        # ── Translation: large depth change between consecutive frames ─────────
        prev_d = prev_depth.get(tid)
        curr_d = obj.get("depth")
        if (
            prev_d is not None
            and curr_d is not None
            and not np.isnan(prev_d)
            and not np.isnan(curr_d)
            and abs(curr_d - prev_d) > TRANSLATION_DEPTH_THRESH
        ):
            if failure_type is None:
                failure_type = "Translation"
            affected.append(tid)
            continue

        # ── Wrong_object: new track_id AND label not in task vocabulary ────────
        seen_frames = track_ids_history.get(tid, [])
        pre_failure_frames = [f for f in seen_frames if f < frame_idx]
        if len(pre_failure_frames) == 0 and lbl not in task_vocab:
            if failure_type is None:
                failure_type = "Wrong_object"
            affected.append(tid)
            continue

        # ── No_Grasp: fallback — object present at failure but not held ────────
        if not obj.get("held_by_gripper", False):
            if failure_type is None:
                failure_type = "No_Grasp"
            affected.append(tid)

    if failure_type is None and failure_label:
        failure_type = "Unknown"

    return {
        "failure_detected": True,
        "type": failure_type,
        "affected_object_ids": list(set(affected)),
    }


# ── per-episode entry point ────────────────────────────────────────────────────

def process_episode(episode_id: str) -> None:
    aligned_path = ALIGNED_DIR / f"{episode_id}.npz"
    detect_path = DETECT_DIR / f"{episode_id}.npz"
    depth_state_path = DEPTH_STATE_DIR / f"{episode_id}.npz"
    track_path = TRACK_DIR / f"{episode_id}.npz"
    out_path = PIPELINE_GRAPHS_DIR / f"{episode_id}.json"

    segment_path = SEGMENT_DIR / f"{episode_id}.npz"  # optional for orig_hw

    missing = []
    for path, name in [
        (aligned_path, "aligned"),
        (detect_path, "detect"),
        (depth_state_path, "depth_state"),
        (track_path, "track"),
    ]:
        if not path.exists():
            missing.append(name)
    if missing:
        print(f"  [skip] {episode_id}: missing {missing}")
        return
    if out_path.exists():
        print(f"  [skip] {episode_id}: already assembled")
        return

    aligned = np.load(aligned_path, allow_pickle=True)
    det = np.load(detect_path, allow_pickle=True)
    ds = np.load(depth_state_path, allow_pickle=True)
    trk = np.load(track_path, allow_pickle=True)

    timestamps: np.ndarray = aligned["timestamps"]
    failure_labels: np.ndarray = aligned["failure_labels"]
    fps_base = float(aligned["fps_base"])

    all_boxes: np.ndarray = det["boxes"]             # (N, MAX_DET, 4)
    all_n_dets: np.ndarray = det["n_dets"]           # (N,)
    label_vocab: list[str] = list(det["label_vocab"])
    label_vocab_set: set[str] = set(label_vocab)
    all_label_ids: np.ndarray = det["label_ids"]     # (N, MAX_DET)

    all_obj_depth: np.ndarray = ds["obj_depth"]      # (N, MAX_DET)
    all_obj_state: np.ndarray = ds["obj_state"]      # (N, MAX_DET)
    state_vocab: list[str] = list(ds["state_vocab"])

    all_track_ids: np.ndarray = trk["track_ids"]             # (N, MAX_DET)
    all_track_conf: np.ndarray = trk["tracking_confidence"]  # (N, MAX_DET)
    all_held: np.ndarray = trk["held_by_gripper"]            # (N, MAX_DET)

    N = len(timestamps)

    # determine frame dimensions
    frame_width = 960.0   # default for AI2THOR sim
    frame_height = 960.0
    if segment_path.exists():
        seg = np.load(segment_path, allow_pickle=True)
        frame_height = float(seg["orig_hw"][0])
        frame_width  = float(seg["orig_hw"][1])
    frame_area = frame_width * frame_height

    # per-track / per-label history for localization flags
    track_ids_history: dict[int, list[int]] = {}
    held_history: dict[str, list[bool]] = {}   # label → per-frame held flags
    prev_depth: dict[int, float] = {}

    frames_out: list[dict] = []

    for i in range(N):
        k = int(all_n_dets[i])
        timestamp = float(timestamps[i])
        fail = bool(failure_labels[i])

        objects: list[dict] = []
        for j in range(k):
            tid = int(all_track_ids[i, j])
            if tid < 0:
                continue

            lid = int(all_label_ids[i, j])
            label = label_vocab[lid] if 0 <= lid < len(label_vocab) else "unknown"

            sid = int(all_obj_state[i, j])
            state_str = state_vocab[sid] if 0 <= sid < len(state_vocab) else "unknown"

            depth_val = float(all_obj_depth[i, j])
            if np.isnan(depth_val):
                depth_val = None   # type: ignore[assignment]

            box = [float(v) for v in all_boxes[i, j]]
            held = bool(all_held[i, j])
            conf = float(all_track_conf[i, j])

            # Fix 2: objects too large to be physically gripped are never held
            bbox_area = (box[2] - box[0]) * (box[3] - box[1])
            if held and bbox_area / frame_area > LARGE_OBJ_HELD_THRESH:
                held = False

            objects.append({
                "id": tid,
                "label": label,
                "bbox": box,
                "depth": depth_val,
                "state": state_str,
                "tracking_confidence": round(conf, 4),
                "held_by_gripper": held,
            })

            # update track_id history
            track_ids_history.setdefault(tid, []).append(i)

        # update label-based held history (OR-aggregate per label per frame)
        labels_held_this_frame: dict[str, bool] = {}
        for obj in objects:
            lbl = obj["label"]
            labels_held_this_frame[lbl] = (
                labels_held_this_frame.get(lbl, False) or obj["held_by_gripper"]
            )
        for lbl, h in labels_held_this_frame.items():
            held_history.setdefault(lbl, []).append(h)

        # spatial relations
        relations = compute_spatial_relations(objects, frame_width)

        # localization flag
        loc_flag = compute_localization_flag(
            i, fail, objects, track_ids_history, held_history, prev_depth,
            label_vocab_set,
        )

        # update prev_depth for next frame
        for obj in objects:
            if obj["depth"] is not None:
                prev_depth[obj["id"]] = obj["depth"]

        frames_out.append({
            "frame_idx": i,
            "timestamp": round(timestamp, 4),
            "failure_label": fail,
            "objects": objects,
            "spatial_relations": relations,
            "localization_flag": loc_flag,
        })

    output = {
        "episode": episode_id,
        "fps_base": fps_base,
        "thresholds": {
            "depth_eps": DEPTH_EPS,
            "on_top_iou": ON_TOP_IOU,
            "inside_iou": INSIDE_IOU,
            "near_dist_norm": NEAR_DIST_PX,
            "translation_depth_thresh": TRANSLATION_DEPTH_THRESH,
        },
        "frames": frames_out,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    n_fail = int(failure_labels.sum())
    n_flags = sum(1 for fr in frames_out if fr["localization_flag"]["failure_detected"])
    print(
        f"  saved {out_path.name} — "
        f"{N} frames, {n_fail} failure frames, {n_flags} localized"
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
        f"Stage 5 — Scene graph assembly | "
        f"{len(episodes)} episodes | rule-based, no learned models"
    )
    for ep in tqdm(episodes, unit="ep"):
        print(f"\n▶ {ep}")
        process_episode(ep)
    print("\nStage 5 complete.")


if __name__ == "__main__":
    main()
