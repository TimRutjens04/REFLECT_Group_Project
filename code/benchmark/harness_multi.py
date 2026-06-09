"""
Multi-object tracking benchmark harness — REFLECT project.

Compares CSRT, ByteTrack, and ReID+Kalman on N objects tracked in parallel.
Objects are always seeded manually on frame 0 (no GDINO on frame 0).

--detect-prompts enables GDINO re-detection on the test video at the
configured interval (required for ByteTrack and ReID+Kalman to actually
track — without it their Kalman filters just predict at the seed position).

For quantitative metrics, supply a clean companion video (--gt-video) with
per-object GDINO prompts (--prompts).

Usage:
    python code/benchmark/harness_multi.py \\
        --video occluded.mp4 \\
        --labels "mug 1. mug 2" \\
        --detect-prompts "coffee mug. coffee mug" \\
        [--gt-video clean.mp4 --prompts "coffee mug. coffee mug"] \\
        [--config code/benchmark/config.yaml] \\
        [--out results/benchmark] \\
        [--play]

All period-separated strings: one entry per object.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.metrics import TrackMetrics, compute_metrics, print_table
from benchmark.trackers.csrt import CsrtTracker
from benchmark.trackers.bytetrack import ByteTrackTracker
from benchmark.trackers.sam2_tracker import Sam2Tracker, sam2_available

ROOT = Path(__file__).resolve().parents[2]

# Distinct per-object colours (BGR)
_OBJ_COLOURS: list[tuple[int, int, int]] = [
    (50,  205,  50),
    (0,   200, 255),
    (255, 100,  50),
    (200,  50, 255),
    (255, 255,  50),
    (50,  100, 255),
    (255,  50, 150),
    (100, 255, 200),
]


def _colour(oi: int) -> tuple[int, int, int]:
    return _OBJ_COLOURS[oi % len(_OBJ_COLOURS)]


# ── Video helpers ─────────────────────────────────────────────────────────────

def load_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


# ── Manual bbox selection ─────────────────────────────────────────────────────

def select_bboxes(frame0: np.ndarray, labels: list[str]) -> list[np.ndarray]:
    """Interactively draw one bbox per label on frame0; returns xyxy float32 arrays."""
    h, w = frame0.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fs    = max(0.5, w / 1200)
    thick = max(1, w // 400)
    bboxes: list[np.ndarray] = []

    for i, label in enumerate(labels):
        display = frame0.copy()
        for j, (prev_label, prev_bb) in enumerate(zip(labels[:i], bboxes)):
            x1, y1, x2, y2 = prev_bb.astype(int)
            c = _colour(j)
            cv2.rectangle(display, (x1, y1), (x2, y2), c, thick)
            cv2.putText(display, prev_label, (x1 + 2, max(y1 - 4, 14)), font, fs * 0.7, c, thick)

        print(f"\n[{i + 1}/{len(labels)}] Draw box around '{label}'")
        print("  Click + drag | SPACE/ENTER to confirm | ESC to abort")

        roi = cv2.selectROI(
            f"[{i + 1}/{len(labels)}] '{label}' — SPACE to confirm, ESC to abort",
            display, showCrosshair=True, fromCenter=False,
        )
        cv2.destroyAllWindows()

        x, y, bw, bh = roi
        if bw == 0 or bh == 0:
            raise RuntimeError(f"No selection for '{label}' — aborted.")

        bb = np.array([float(x), float(y), float(x + bw), float(y + bh)], dtype=np.float32)
        bboxes.append(bb)
        print(f"  -> {bb.astype(int).tolist()}")

    return bboxes


# ── GT from clean companion video (GDINO, no re-detection on test video) ──────

def _to_rgbd(bgr: np.ndarray):
    from interfaces import RgbdFrame
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return RgbdFrame(rgb=rgb, depth=np.zeros((h, w), dtype=np.float32), step_idx=0, metadata={})


def build_gt_from_clean(
    gt_frames: list[np.ndarray],
    prompts: list[str],
    manual_bboxes: list[np.ndarray],
    cfg: dict,
    n_test_frames: int,
    interval: int = 1,
) -> list[list[np.ndarray | None]]:
    from detector import GroundingDinoDetector
    det = GroundingDinoDetector(score_thresh=cfg["gdino"]["score_thresh"], device="cpu")
    det.load()

    n = len(prompts)
    gt: list[list[np.ndarray | None]] = [[None] * n_test_frames for _ in range(n)]

    for oi in range(n):
        gt[oi][0] = manual_bboxes[oi]

    n_gt = min(len(gt_frames), n_test_frames)
    frames_to_run = [fi for fi in range(1, n_gt) if fi % interval == 0]
    print(f"  Running GDINO on {len(frames_to_run)} / {n_gt} frames (interval={interval}) ...")
    for fi in frames_to_run:
        bgr = gt_frames[fi]
        for oi, prompt in enumerate(prompts):
            dr = det.detect(_to_rgbd(bgr), prompt)
            if dr.detections:
                gt[oi][fi] = dr.detections[0].bbox_2d

    return gt


def build_dets_per_object(
    frames: list[np.ndarray],
    detect_prompts: list[str],
    manual_bboxes: list[np.ndarray],
    cfg: dict,
) -> list[list[tuple[np.ndarray, float] | None]]:
    """
    Build per-object detection lists for the test video.
    Frame 0: manual bbox (score=1.0).
    Frames at detection_interval: GDINO with each object's prompt.
    All other frames: None (tracker uses internal motion model).
    """
    from detector import GroundingDinoDetector
    interval = cfg["gdino"]["detection_interval"]
    det = GroundingDinoDetector(score_thresh=cfg["gdino"]["score_thresh"], device="cpu")
    det.load()

    n = len(detect_prompts)
    dets: list[list[tuple[np.ndarray, float] | None]] = [[None] * len(frames) for _ in range(n)]

    for oi in range(n):
        dets[oi][0] = (manual_bboxes[oi], 1.0)

    for fi in range(interval, len(frames), interval):
        bgr = frames[fi]
        for oi, prompt in enumerate(detect_prompts):
            dr = det.detect(_to_rgbd(bgr), prompt)
            if dr.detections:
                dets[oi][fi] = (dr.detections[0].bbox_2d, dr.detections[0].score)

    return dets


# ── Per-group tracker run ─────────────────────────────────────────────────────

def run_tracker_group(
    trackers: list,
    frames: list[np.ndarray],
    seed_bboxes: list[np.ndarray],
    dets_per_obj: list[list[tuple[np.ndarray, float] | None]] | None = None,
) -> tuple[list[list[np.ndarray | None]], list[list[float]]]:
    """
    Run N trackers in parallel (one per object).
    Frame 0: seeded with manual bbox.
    Frames 1+: dets_per_obj[oi][fi] if provided, else None (Kalman-only prediction).
    Returns preds[obj][frame] and timings[obj][frame].
    """
    for t in trackers:
        t.reset()

    n = len(trackers)
    preds:   list[list[np.ndarray | None]] = [[] for _ in range(n)]
    timings: list[list[float]]             = [[] for _ in range(n)]

    for fi, frame in enumerate(frames):
        for oi, tracker in enumerate(trackers):
            if fi == 0:
                bbox_in, score = seed_bboxes[oi], 1.0
            elif dets_per_obj is not None:
                det = dets_per_obj[oi][fi]
                bbox_in = det[0] if det else None
                score   = det[1] if det else None
            else:
                bbox_in, score = None, None
            t0 = time.perf_counter()
            bbox_out, _ = tracker.step(frame, bbox_in, score)
            timings[oi].append((time.perf_counter() - t0) * 1000.0)
            preds[oi].append(bbox_out)

    return preds, timings


# ── Comparison video ──────────────────────────────────────────────────────────

def write_comparison_video(
    frames: list[np.ndarray],
    tracker_preds: dict[str, list[list[np.ndarray | None]]],
    gt_per_obj: list[list[np.ndarray | None]] | None,
    object_labels: list[str],
    out_path: str,
    playback_fps: float,
) -> None:
    if not frames:
        return

    h, w = frames[0].shape[:2]
    tracker_names = list(tracker_preds.keys())
    n_t     = len(tracker_names)
    panel_w = (w // max(n_t, 1)) & ~1  # force even width for codec compatibility
    out_w   = panel_w * n_t

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, playback_fps, (out_w, h))
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = max(0.5, panel_w / 900)
    thick  = max(1, panel_w // 400)

    for fi, bgr in enumerate(frames):
        canvas = np.zeros((h, out_w, 3), dtype=np.uint8)

        sx, sy = panel_w / w, h / h  # sy always 1 — panels are full height

        for ti, tname in enumerate(tracker_names):
            cx    = ti * panel_w
            panel = cv2.resize(bgr, (panel_w, h))

            for oi in range(len(object_labels)):
                pred = tracker_preds[tname][oi][fi]
                if pred is not None:
                    px1 = int(pred[0] * sx); py1 = int(pred[1])
                    px2 = int(pred[2] * sx); py2 = int(pred[3])
                    cv2.rectangle(panel, (px1, py1), (px2, py2), _colour(oi), thick + 1)

            # Tracker name banner at the top of each column
            (tw, th), _ = cv2.getTextSize(tname, font, fs, thick)
            cv2.rectangle(panel, (0, 0), (tw + 10, th + 14), (30, 30, 30), -1)
            cv2.putText(panel, tname, (5, th + 7), font, fs, (255, 255, 255), thick,
                        cv2.LINE_AA)

            # Frame counter bottom-left
            cv2.putText(panel, f"f{fi:04d}", (5, h - 8), font, fs * 0.6,
                        (160, 160, 160), 1, cv2.LINE_AA)

            canvas[:, cx:cx + panel_w] = panel

        writer.write(canvas)

    writer.release()


# ── Live playback ─────────────────────────────────────────────────────────────

def play_video(path: str, fps: float) -> None:
    """Play a video file in a cv2 window. Press Q or ESC to quit, SPACE to pause."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open {path} for playback")
        return
    delay = max(1, int(1000 / fps))
    print("\nPlaying comparison video — Q/ESC to quit, SPACE to pause.")
    paused = False
    frame = None
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
        if frame is not None:
            cv2.imshow("Tracker comparison", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-object tracking benchmark (manual seed, no GDINO re-detection)."
    )
    ap.add_argument("--video",   required=True,
                    help="Occluded test video.")
    ap.add_argument("--labels",  required=True,
                    help="Period-separated display names, one per object. "
                         "e.g. \"mug 1. mug 2\"")
    ap.add_argument("--detect-prompts", default=None,
                    help="Period-separated GDINO prompts for re-detection on the TEST video "
                         "at the configured interval. e.g. \"coffee mug. coffee mug\"")
    ap.add_argument("--gt-video", default=None,
                    help="Clean companion video for GT. Requires --prompts.")
    ap.add_argument("--prompts",  default=None,
                    help="Period-separated GDINO prompts for GT generation on --gt-video. "
                         "Must match the number of --labels entries.")
    ap.add_argument("--pseudo-gt", default=None,
                    help="Run GDINO on every frame of the TEST video itself as pseudo-GT. "
                         "Pass period-separated prompts (one per object). "
                         "Mutually exclusive with --gt-video.")
    ap.add_argument("--config", default=str(ROOT / "code/benchmark/config.yaml"))
    ap.add_argument("--out",    default=str(ROOT / "results/benchmark"))
    ap.add_argument("--play",   action="store_true",
                    help="Play the comparison video in a window after benchmarking.")
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(".") if s.strip()]
    n_objects = len(labels)
    if n_objects == 0:
        ap.error("--labels produced no entries after splitting on '.'")

    if args.gt_video and args.pseudo_gt:
        ap.error("--gt-video and --pseudo-gt are mutually exclusive")
    if args.gt_video and not args.prompts:
        ap.error("--gt-video requires --prompts")

    detect_prompts: list[str] = []
    if args.detect_prompts:
        detect_prompts = [s.strip() for s in args.detect_prompts.split(".") if s.strip()]
        if len(detect_prompts) != n_objects:
            ap.error(f"--detect-prompts has {len(detect_prompts)} entries but --labels has {n_objects}")

    prompts: list[str] = []
    if args.prompts:
        prompts = [s.strip() for s in args.prompts.split(".") if s.strip()]
        if len(prompts) != n_objects:
            ap.error(f"--prompts has {len(prompts)} entries but --labels has {n_objects}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    iou_thresh = cfg["benchmark"]["iou_match_thresh"]
    play_fps   = float(cfg["benchmark"]["playback_fps"])

    print(f"Loading video: {args.video}")
    frames, src_fps = load_frames(args.video)
    h, w = frames[0].shape[:2]
    print(f"  {len(frames)} frames @ {src_fps:.1f} fps  ({w}x{h})")
    print(f"  Objects ({n_objects}): {labels}")

    print("\nManual bbox selection on frame 0 ...")
    seed_bboxes = select_bboxes(frames[0], labels)

    dets_per_obj: list[list[tuple[np.ndarray, float] | None]] | None = None
    if detect_prompts:
        interval = cfg["gdino"]["detection_interval"]
        print(f"\nRunning GDINO on test video (every {interval} frames) for re-detection ...")
        dets_per_obj = build_dets_per_object(frames, detect_prompts, seed_bboxes, cfg)
    else:
        print("\nNo --detect-prompts — trackers will run from manual seed only.")

    gt_per_obj: list[list[np.ndarray | None]] | None = None
    if args.gt_video:
        print(f"\nBuilding GT from companion video: {args.gt_video}")
        gt_frames, _ = load_frames(args.gt_video)
        gt_per_obj = build_gt_from_clean(gt_frames, prompts, seed_bboxes, cfg, len(frames))
    elif args.pseudo_gt:
        pseudo_prompts = [s.strip() for s in args.pseudo_gt.split(".") if s.strip()]
        if len(pseudo_prompts) != n_objects:
            ap.error(f"--pseudo-gt has {len(pseudo_prompts)} entries but --labels has {n_objects}")
        pseudo_gt_interval = cfg["benchmark"].get("pseudo_gt_interval", 10)
        print(f"\nBuilding pseudo-GT: GDINO every {pseudo_gt_interval} frames of {args.video} ...")
        gt_per_obj = build_gt_from_clean(frames, pseudo_prompts, seed_bboxes, cfg, len(frames), interval=pseudo_gt_interval)
    else:
        print("\nNo GT source — comparison video only (no quantitative metrics).")

    print("\nBuilding trackers ...")
    tracker_groups: dict[str, list] = {
        "CSRT":      [CsrtTracker()        for _ in range(n_objects)],
        "ByteTrack": [ByteTrackTracker(cfg) for _ in range(n_objects)],
    }

    all_preds:   dict[str, list[list[np.ndarray | None]]] = {}
    all_metrics: list[TrackMetrics] = []

    for tname, tgroup in tracker_groups.items():
        print(f"Running {tname} ({n_objects} parallel tracker(s)) ...")
        preds_per_obj, timings_per_obj = run_tracker_group(
            tgroup, frames, seed_bboxes, dets_per_obj
        )
        all_preds[tname] = preds_per_obj

        if gt_per_obj is not None:
            for oi in range(n_objects):
                m = compute_metrics(
                    f"{tname}/{labels[oi]}",
                    preds_per_obj[oi],
                    gt_per_obj[oi],
                    tgroup[oi].id_switches,
                    timings_per_obj[oi],
                    iou_thresh,
                )
                all_metrics.append(m)

    if sam2_available:
        print("Running SAM 2 (offline, full video) ...")
        sam2 = Sam2Tracker()
        t0 = time.perf_counter()
        sam2_preds, sam2_id_sw = sam2.run_video(frames, seed_bboxes)
        sam2_ms_per_frame = (time.perf_counter() - t0) * 1000.0 / len(frames)
        all_preds["SAM 2"] = sam2_preds
        if gt_per_obj is not None:
            for oi in range(n_objects):
                m = compute_metrics(
                    f"SAM 2/{labels[oi]}",
                    sam2_preds[oi],
                    gt_per_obj[oi],
                    sam2_id_sw[oi],
                    [sam2_ms_per_frame] * len(frames),
                    iou_thresh,
                )
                all_metrics.append(m)
    else:
        print("SAM 2 not installed — skipping (pip install sam2 to enable).")

    if all_metrics:
        print_table(all_metrics)

    stem    = Path(args.video).stem
    out_dir = Path(args.out) / f"{stem}_multi"
    out_dir.mkdir(parents=True, exist_ok=True)

    if all_metrics:
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump([vars(m) for m in all_metrics], f, indent=2)
        print(f"Metrics -> {metrics_path}")

    vis = str(out_dir / "comparison_tracked.mp4")
    if cfg["benchmark"]["visualize"]:
        print(f"Writing comparison video -> {vis}")
        write_comparison_video(frames, all_preds, gt_per_obj, labels, vis, play_fps)

    print("Done.")

    if args.play and cfg["benchmark"]["visualize"]:
        play_video(vis, play_fps)


if __name__ == "__main__":
    main()
