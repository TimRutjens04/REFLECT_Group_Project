"""
CSRT tracker pipeline — REFLECT / RoboFail dataset.

Reads detect/<ep>.npz (Grounding DINO bboxes) and aligned/<ep>.npz (RGB frames),
runs one CSRT tracker per object class across all frames, applies composite failure
detection, and triggers Grounding DINO re-detection when failures are found.

Output: track/<ep>.npz — per-frame bboxes, failure flags, and recovery events.

=== OUTPUT FORMAT ===

  track/<episode_id>.npz
    boxes               (N, n_obj, 4)   float32 — tracked [x1,y1,x2,y2]; NaN if lost
    track_ids           (N, n_obj)      int32   — object vocab index; -1 if not tracked
    tracking_confidence (N, n_obj)      float32 — 1.0 minus 0.25 per active failure bit
    failure_flags       (N, n_obj)      uint8   — bitmask (see FLAG_* constants)
    held_by_gripper     (N, n_obj)      bool    — stub; always False (awaits depth stage)
    recovery_frames     (M,)            int32   — frames where GDINO re-detection fired
    id_switch_frames    (K,)            int32   — frames where any tracker was reinit'd
    timestamps          (N,)            float64 — copied from aligned/
    failure_labels      (N,)            bool    — copied from aligned/
    fps_base            ()              float64 — copied from aligned/
    label_vocab         (n_obj,)        U       — object labels

  Object column order matches label_vocab from detect/<ep>.npz.

Usage:
    python track.py                         # process all episodes in detect/
    python track.py boilWater-1             # single episode by name
    python track.py boilWater-1 --force
    python track.py boilWater-1 --object Pot
"""

import os
import sys
import time
from collections import deque

import numpy as np
import zarr
from tqdm import tqdm

from detector import GroundingDinoDetector
from frame_provider import AlignedEpisodeFrameProvider
from interfaces import DetectedObject, DetectionResult, ObjectDetector
from reid import ObjectReIdMatcher
from tracker import CsrtObjectTracker
from tracking_log import TrackingLog
from validator import CompositeTrackingValidator

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DETECT_DIR    = os.path.join(ROOT, "detect")
ALIGNED_DIR   = os.path.join(ROOT, "aligned")
DEPTH_DIR     = os.path.join(ROOT, "depth_state")
TRACK_DIR     = os.path.join(ROOT, "track")
REAL_DATA_DIR = os.path.join(ROOT, "data", "real_data")

# ── Thresholds (tune against RoboFail ground truth) ───────────────────────────

AREA_CHANGE_THRESH   = 0.50   # flag if bbox area changes > 50 % of init area
DRIFT_THRESH_PX      = 30     # flag if centroid moves > 30 px from previous frame
DEPTH_JUMP_THRESH    = 0.30   # flag if mean depth inside box changes > 0.30 m
REDETECT_INTERVAL    = 30     # force GDINO re-detection after this many frames
GRIP_APPROACH_WINDOW = 10     # frames of depth history for held-object attribution
GRIPPER_CLOSED_STATE = 4      # gripper_state: 0=open, 1=closing, 4=gripping, 5=opening
GDINO_MODEL_ID     = "IDEA-Research/grounding-dino-base"
GDINO_SCORE_THRESH = 0.30

# ── Failure flag bitmask constants ────────────────────────────────────────────

FLAG_CSRT_FAIL    = 1 << 0   # OpenCV CSRT reported failure
FLAG_AREA_CHANGE  = 1 << 1   # bbox area changed > AREA_CHANGE_THRESH from init
FLAG_DRIFT        = 1 << 2   # centroid drifted > DRIFT_THRESH_PX in one step
FLAG_DEPTH_JUMP   = 1 << 3   # mean depth jumped > DEPTH_JUMP_THRESH (stub)
FLAG_FORCED_REDET = 1 << 4   # REDETECT_INTERVAL exceeded; GDINO was called
FLAG_FROZEN       = 1 << 5   # tracker update frozen; holding last valid bbox

# MPS does not implement aten::_cummax_helper used by Grounding DINO's attention
# masking; GDINO always runs on CPU. CSRT itself uses numpy/opencv (no torch device).
GDINO_DEVICE = "cpu"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reason_to_flags(reason: str | None) -> int:
    """Map ValidationResult.reason string to failure bitmask."""
    if not reason:
        return 0
    flags = 0
    _map = {
        "area_change": FLAG_AREA_CHANGE,
        "drift":       FLAG_DRIFT,
        "depth_jump":  FLAG_DEPTH_JUMP,
        "timeout":     FLAG_FORCED_REDET,
    }
    for part in reason.split(";"):
        key = part.strip().split(":")[-1].strip()
        flags |= _map.get(key, 0)
    return flags


def _best_detect_box(det: dict, obj_idx: int, frame_idx: int) -> np.ndarray | None:
    """Return highest-scoring bbox for obj_idx at frame_idx from a detect npz dict."""
    n = int(det["n_dets"][frame_idx])
    if n == 0:
        return None
    label_ids = det["label_ids"][frame_idx, :n]
    scores    = det["scores"][frame_idx, :n]
    boxes     = det["boxes"][frame_idx, :n]
    mask = label_ids == obj_idx
    if not mask.any():
        return None
    best = int(scores[mask].argmax())
    return boxes[mask][best].astype(np.float32)


def _load_gripper_state(
    episode_id: str,
    frame_timestamps: np.ndarray,
) -> np.ndarray:
    """Return bool (N,) array — True when gripper is closed at each aligned frame.
    Falls back to all-False when no Zarr replay buffer exists (sim/demo episodes)."""
    n = len(frame_timestamps)
    zarr_path = os.path.join(REAL_DATA_DIR, episode_id, "replay_buffer.zarr")
    if not os.path.exists(zarr_path):
        return np.zeros(n, dtype=bool)
    try:
        z         = zarr.open(zarr_path)
        zarr_ts   = z["data/timestamp"][:]
        zarr_gs   = z["data/gripper_state"][:]
        indices   = np.searchsorted(zarr_ts, frame_timestamps).clip(0, len(zarr_ts) - 1)
        return (zarr_gs[indices] == GRIPPER_CLOSED_STATE).astype(bool)
    except Exception as e:
        print(f"  [warn] could not load gripper state for {episode_id}: {e}")
        return np.zeros(n, dtype=bool)


def _identify_held_object(
    n_obj: int,
    centroid_depth_history: dict[int, deque],
    confidences: list[float],
) -> int | None:
    """Return the obj_idx most likely being held based on approach motion in depth.

    Picks the object whose centroid depth dropped the most over the last
    GRIP_APPROACH_WINDOW frames (moved closest to camera = gripper approached it).
    Tiebreaks by highest tracking confidence.
    Returns None if no depth data is available (blanket suppression fallback).
    """
    best_idx   = None
    best_drop  = -float("inf")
    best_conf  = -1.0
    any_depth  = False

    for obj_idx in range(n_obj):
        hist = list(centroid_depth_history.get(obj_idx, []))
        if len(hist) < 2 or all(d == 0.0 for d in hist):
            continue
        any_depth = True
        depth_drop = hist[0] - hist[-1]   # positive = moved closer
        conf = confidences[obj_idx]
        if depth_drop > best_drop or (depth_drop == best_drop and conf > best_conf):
            best_drop = depth_drop
            best_conf = conf
            best_idx  = obj_idx

    return best_idx if any_depth else None


# ── Per-episode tracking ───────────────────────────────────────────────────────

def track_episode(
    episode_id: str,
    detector: ObjectDetector,
    object_prompts: list[str] | None = None,
    context_labels: list[str] | None = None,
) -> None:
    """
    Track objects in one of three modes:
      - No object_prompts: reads vocab from detect/<ep>.npz (full pipeline).
      - Single object_prompts entry: GDINO seeds frame 0; slug output file.
      - Multiple object_prompts entries: GDINO seeds all objects; detect/ not
        required; each object uses the rest as context for disambiguation.

    context_labels: additional category names added to the GDINO prompt for
    disambiguation on top of the vocab (mainly useful in single-object mode).
    """
    if object_prompts:
        slug     = "_".join(p.replace(" ", "_") for p in object_prompts)
        out_name = f"{episode_id}_{slug}.npz"
    else:
        out_name = f"{episode_id}.npz"
    out_path = os.path.join(TRACK_DIR, out_name)

    provider = AlignedEpisodeFrameProvider(episode_id, ALIGNED_DIR, DEPTH_DIR)
    n = len(provider)

    aligned        = np.load(os.path.join(ALIGNED_DIR, f"{episode_id}.npz"), allow_pickle=False)
    timestamps     = aligned["timestamps"]
    failure_labels = aligned["failure_labels"]
    fps_base       = float(aligned["fps_base"])

    det = None
    if object_prompts:
        vocab = object_prompts
    else:
        det   = np.load(os.path.join(DETECT_DIR, f"{episode_id}.npz"), allow_pickle=False)
        vocab = list(det["label_vocab"])

    n_obj = len(vocab)

    # Output arrays
    out_boxes = np.full((n, n_obj, 4), np.nan, dtype=np.float32)
    out_tids  = np.full((n, n_obj),    -1,      dtype=np.int32)
    out_conf  = np.zeros((n, n_obj),            dtype=np.float32)
    out_flags = np.zeros((n, n_obj),            dtype=np.uint8)
    out_held  = np.zeros((n, n_obj),            dtype=bool)

    recovery_frames_list: list[int] = []
    id_switch_frames_list: list[int] = []

    t0 = time.perf_counter()

    reid = ObjectReIdMatcher()

    # One tracker + validator per object class
    trackers:   list[CsrtObjectTracker | None]   = [None] * n_obj
    validators: list[CompositeTrackingValidator] = [
        CompositeTrackingValidator(
            area_change_thresh=AREA_CHANGE_THRESH,
            drift_thresh_px=float(DRIFT_THRESH_PX),
            depth_jump_thresh=DEPTH_JUMP_THRESH,
            redetect_interval=REDETECT_INTERVAL,
        )
        for _ in range(n_obj)
    ]

    # ── Gripper state ─────────────────────────────────────────────────────────
    gripper_closed = _load_gripper_state(episode_id, timestamps)
    has_gripper    = gripper_closed.any()
    if has_gripper:
        print(f"    gripper data loaded — {gripper_closed.sum()} closed frames")

    # Per-object depth history for held-object attribution
    centroid_depth_history: dict[int, deque] = {
        i: deque(maxlen=GRIP_APPROACH_WINDOW) for i in range(n_obj)
    }
    held_by_gripper_active: list[bool] = [False] * n_obj
    held_obj_idx: int | None = None

    # Per-object log state
    last_detection_frame: list[int] = [0] * n_obj
    log = TrackingLog(sequence_id=episode_id)

    # ── Seed each tracker ─────────────────────────────────────────────────────
    # Build per-object context: every other label acts as a negative category so
    # GDINO can disambiguate (e.g. "pot" vs "fridge" in one prompt).
    def _ctx(obj_idx: int) -> list[str]:
        base   = context_labels or []
        others = [v for i, v in enumerate(vocab) if i != obj_idx]
        return others + [c for c in base if c not in others]

    # Scan forward up to SEED_SCAN_FRAMES to find the first frame where each
    # object is detectable. Stops as soon as all objects are seeded.
    SEED_SCAN_FRAMES = min(10, n)
    unseeded = set(range(n_obj))

    for seed_fi in range(SEED_SCAN_FRAMES):
        if not unseeded:
            break
        seed_frame = provider.get_frame(seed_fi)

        for obj_idx in list(unseeded):
            label = vocab[obj_idx]
            seed_embed: np.ndarray | None = None
            if object_prompts:
                det_result, embeds = detector.detect_with_embeddings(seed_frame, label, context_labels=_ctx(obj_idx))
                if embeds:
                    seed_embed = embeds[0]
            else:
                bbox0 = _best_detect_box(det, obj_idx, seed_fi)
                if bbox0 is not None:
                    det_result = DetectionResult(detections=[
                        DetectedObject(label=label, score=1.0, bbox_2d=bbox0)
                    ])
                else:
                    det_result, embeds = detector.detect_with_embeddings(seed_frame, label, context_labels=_ctx(obj_idx))
                    if embeds:
                        seed_embed = embeds[0]

            if not det_result.detections:
                continue

            tracker = CsrtObjectTracker()
            tracker.initialize(seed_frame, det_result)
            trackers[obj_idx] = tracker
            unseeded.discard(obj_idx)
            if seed_embed is not None:
                reid.register(obj_idx, seed_embed)

            seed_bbox = det_result.detections[0].bbox_2d
            out_boxes[seed_fi, obj_idx] = seed_bbox
            out_tids[seed_fi,  obj_idx] = obj_idx
            out_conf[seed_fi,  obj_idx] = 1.0
            if seed_fi > 0:
                print(f"    [{label}] seeded at frame {seed_fi}")

    for obj_idx in unseeded:
        print(f"  [warn] '{vocab[obj_idx]}' not detected in first {SEED_SCAN_FRAMES} frames — skipping")

    # ── Per-object freeze state ────────────────────────────────────────────────
    # When a failure flag fires we freeze the CSRT model (stop calling
    # tracker.track()) and hold the last valid bbox until GDINO re-acquires
    # the object. This prevents the appearance model from being corrupted by
    # whatever is inside the bbox during occlusion.
    frozen:          list[bool]              = [False] * n_obj
    last_valid_bbox: list[np.ndarray | None] = [None]  * n_obj

    # ── Frames 1 … N-1 ────────────────────────────────────────────────────────
    for fi in tqdm(range(1, n), desc=episode_id, leave=False):
        frame = provider.get_frame(fi)

        # ── Gripper transition detection (once per frame, before per-object loop)
        prev_closed = gripper_closed[fi - 1]
        curr_closed = gripper_closed[fi]

        if curr_closed and not prev_closed:
            # Gripper just closed — attribute the held object
            confs = [float(out_conf[fi - 1, i]) for i in range(n_obj)]
            held_obj_idx = _identify_held_object(n_obj, centroid_depth_history, confs)
            if held_obj_idx is not None:
                held_by_gripper_active[held_obj_idx] = True
                frozen[held_obj_idx] = True
            else:
                # No depth data — blanket suppression
                for i in range(n_obj):
                    held_by_gripper_active[i] = True
                    frozen[i] = True

        elif not curr_closed and prev_closed:
            # Gripper opened — release held objects back to normal frozen state
            for i in range(n_obj):
                if held_by_gripper_active[i]:
                    held_by_gripper_active[i] = False
                    # Remains frozen=True; GDINO will re-acquire on next frame

        for obj_idx, label in enumerate(vocab):
            tracker = trackers[obj_idx]
            if tracker is None:
                continue

            bbox           = last_valid_bbox[obj_idx]
            flags          = 0
            tracker_status = "ok"

            if held_by_gripper_active[obj_idx]:
                # ── Held: suppress GDINO, freeze position ────────────────────
                flags          = FLAG_FROZEN
                tracker_status = "held"
                out_held[fi, obj_idx] = True

            elif frozen[obj_idx]:
                # ── Frozen: model preserved; try GDINO every frame ───────────
                flags          = FLAG_FROZEN
                tracker_status = "searching"

                new_det, new_embeds = detector.detect_with_embeddings(frame, label, context_labels=_ctx(obj_idx))
                if new_det.detections:
                    last_detection_frame[obj_idx] = fi
                    old_track_id = tracker._track_id
                    if new_embeds and reid.is_same_object(obj_idx, new_embeds[0]):
                        tracker.reinitialize(frame, new_det)
                        reid.update(obj_idx, new_embeds[0])
                        tracker_status = "recovered"
                    else:
                        tracker.initialize(frame, new_det)
                        id_switch_frames_list.append(fi)
                        if new_embeds:
                            reid.register(obj_idx, new_embeds[0])
                        tracker_status = "redetected"
                    validators[obj_idx].reset(old_track_id)
                    bbox = new_det.detections[0].bbox_2d
                    last_valid_bbox[obj_idx] = bbox
                    frozen[obj_idx] = False
                    flags = FLAG_FORCED_REDET
                    recovery_frames_list.append(fi)

            else:
                # ── Normal: run CSRT update ───────────────────────────────────
                tracking_result = tracker.track(frame)

                if not tracking_result.tracked_objects:
                    flags |= FLAG_CSRT_FAIL
                    bbox = None
                else:
                    val_result = validators[obj_idx].validate(frame, tracking_result)
                    if not val_result.is_valid:
                        flags |= _reason_to_flags(val_result.reason)
                    bbox = tracking_result.tracked_objects[0].bbox_2d

                    # Update centroid depth history for gripper attribution
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    h, w = frame.depth.shape[:2]
                    xi = int(np.clip(cx, 0, w - 1))
                    yi = int(np.clip(cy, 0, h - 1))
                    centroid_depth_history[obj_idx].append(float(frame.depth[yi, xi]))

                if flags:
                    tracker_status = "frozen"
                    frozen[obj_idx] = True
                    last_valid_bbox[obj_idx] = bbox

                    new_det, new_embeds = detector.detect_with_embeddings(frame, label, context_labels=_ctx(obj_idx))
                    if new_det.detections:
                        last_detection_frame[obj_idx] = fi
                        old_track_id = (
                            tracking_result.tracked_objects[0].track_id
                            if tracking_result.tracked_objects
                            else tracker._track_id
                        )
                        if new_embeds and reid.is_same_object(obj_idx, new_embeds[0]):
                            tracker.reinitialize(frame, new_det)
                            reid.update(obj_idx, new_embeds[0])
                            tracker_status = "recovered"
                        else:
                            tracker.initialize(frame, new_det)
                            id_switch_frames_list.append(fi)
                            if new_embeds:
                                reid.register(obj_idx, new_embeds[0])
                            tracker_status = "redetected"
                        validators[obj_idx].reset(old_track_id)
                        bbox = new_det.detections[0].bbox_2d
                        last_valid_bbox[obj_idx] = bbox
                        frozen[obj_idx] = False
                        flags |= FLAG_FORCED_REDET
                        recovery_frames_list.append(fi)
                else:
                    last_valid_bbox[obj_idx] = bbox

            # ── Write per-frame arrays ────────────────────────────────────────
            out_flags[fi, obj_idx] = np.uint8(flags)
            if bbox is not None:
                out_boxes[fi, obj_idx] = bbox
                out_tids[fi,  obj_idx] = obj_idx
                n_bits = bin(flags).count("1")
                out_conf[fi, obj_idx] = max(0.0, 1.0 - 0.25 * n_bits)

            # ── Log record ────────────────────────────────────────────────────
            log.record(
                frame_id=fi,
                timestamp=float(timestamps[fi]),
                object_id=f"{label}_{obj_idx}",
                label=label,
                bbox_xyxy=bbox.astype(int).tolist() if bbox is not None else [],
                tracker_confidence=float(out_conf[fi, obj_idx]),
                tracker_status=tracker_status,
                bbox_size_change_flag=bool(flags & FLAG_AREA_CHANGE),
                drift_flag=bool(flags & FLAG_DRIFT),
                recovery_trigger=bool(flags & FLAG_FORCED_REDET),
                held_by_gripper=bool(out_held[fi, obj_idx]),
                last_detection_frame=last_detection_frame[obj_idx],
            )

    recovery_arr  = np.array(sorted(set(recovery_frames_list)), dtype=np.int32)
    id_switch_arr = np.array(sorted(set(id_switch_frames_list)), dtype=np.int32)

    np.savez_compressed(
        out_path,
        boxes               = out_boxes,
        track_ids           = out_tids,
        tracking_confidence = out_conf,
        failure_flags       = out_flags,
        held_by_gripper     = out_held,
        recovery_frames     = recovery_arr,
        id_switch_frames    = id_switch_arr,
        timestamps          = timestamps,
        failure_labels      = failure_labels,
        fps_base            = np.float64(fps_base),
        label_vocab         = np.array(vocab),
    )

    log_path = out_path.replace(".npz", "_log.json")
    log.save_json(log_path)

    elapsed = time.perf_counter() - t0
    print(
        f"  saved {out_path}  "
        f"({n} frames, {n_obj} objects, "
        f"{len(recovery_arr)} recoveries, "
        f"{elapsed:.1f}s)"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(TRACK_DIR, exist_ok=True)

    args  = sys.argv[1:]
    force = "--force" in args
    args  = [a for a in args if a != "--force"]

    # --object accepts one or more comma-separated labels:
    #   --object "Pot."                       → single-object mode
    #   --object "Pot.,StoveBurner,Fridge"    → multi-object / no detect/ needed
    object_prompts: list[str] | None = None
    if "--object" in args:
        idx            = args.index("--object")
        object_prompts = [p.strip() for p in args[idx + 1].split(".") if p.strip()]
        args           = args[:idx] + args[idx + 2:]

    # --context "bowl,knife"  — extra disambiguation categories (single-object mode only;
    # in multi-object mode the vocab already provides mutual context via _ctx()).
    context_labels: list[str] | None = None
    if "--context" in args:
        idx            = args.index("--context")
        context_labels = [c.strip() for c in args[idx + 1].split(",") if c.strip()]
        args           = args[:idx] + args[idx + 2:]

    target = args[0] if args else None

    if target:
        episodes = [target]
    elif object_prompts:
        episodes = sorted(
            f[:-4] for f in os.listdir(ALIGNED_DIR) if f.endswith(".npz")
        )
    else:
        episodes = sorted(
            f[:-4] for f in os.listdir(DETECT_DIR) if f.endswith(".npz")
        )

    prompt_info = f"  objects={object_prompts}" if object_prompts else ""
    ctx_info    = f"  context={context_labels}" if context_labels else ""
    print(
        f"CSRT tracking: {len(episodes)} episode(s)  device={GDINO_DEVICE}"
        + prompt_info + ctx_info
        + ("  [force]" if force else "")
    )
    print(f"Loading {GDINO_MODEL_ID} ...")
    detector = GroundingDinoDetector(score_thresh=GDINO_SCORE_THRESH, device=GDINO_DEVICE)
    detector.load()

    skipped = 0
    for ep in episodes:
        if object_prompts:
            slug     = "_".join(p.replace(" ", "_") for p in object_prompts)
            out_name = f"{ep}_{slug}.npz"
        else:
            out_name = f"{ep}.npz"
        out_path     = os.path.join(TRACK_DIR,  out_name)
        aligned_path = os.path.join(ALIGNED_DIR, f"{ep}.npz")
        detect_path  = os.path.join(DETECT_DIR,  f"{ep}.npz")

        if os.path.exists(out_path) and not force:
            print(f"  [skip] {ep} — already computed")
            skipped += 1
            continue
        if not os.path.exists(aligned_path):
            print(f"  [skip] {ep} — no aligned file")
            skipped += 1
            continue
        if not object_prompts and not os.path.exists(detect_path):
            print(f"  [skip] {ep} — no detect file (run without --object or add detect/)")
            skipped += 1
            continue

        print(f"  {ep}")
        track_episode(ep, detector, object_prompts=object_prompts, context_labels=context_labels)

    print(f"\nDone. {len(episodes) - skipped} computed, {skipped} skipped.")


if __name__ == "__main__":
    main()
