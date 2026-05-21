"""
Tracking benchmark harness — REFLECT project.

Compares CSRT, ByteTrack, and ReID+Kalman on a provided video.
GDINO runs every K frames; trackers use their motion models in between.
GT boxes come from GDINO on a companion clean video (--gt-video); without
it, GDINO on the test video itself is used as pseudo-GT.

Usage:
    python code/benchmark/harness.py \\
        --video path/to/occluded.mp4 \\
        --prompt "cooking pot" \\
        [--gt-video path/to/clean.mp4] \\
        [--config code/benchmark/config.yaml] \\
        [--out results/benchmark]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector import GroundingDinoDetector
from interfaces import RgbdFrame

from benchmark.metrics import TrackMetrics, compute_metrics, print_table
from benchmark.trackers.base import BenchmarkTracker
from benchmark.trackers.csrt import CsrtTracker
from benchmark.trackers.bytetrack import ByteTrackTracker
from benchmark.trackers.reid_kalman import ReidKalmanTracker

ROOT = Path(__file__).resolve().parents[2]

COLOURS = {
    "CSRT":        (50,  205,  50),
    "ByteTrack":   (0,   200, 255),
    "ReID+Kalman": (255, 100,  50),
}


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


def _to_rgbd(bgr: np.ndarray) -> RgbdFrame:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return RgbdFrame(rgb=rgb, depth=np.zeros((h, w), dtype=np.float32),
                     step_idx=0, metadata={})


# ── GDINO helpers ─────────────────────────────────────────────────────────────

def build_detector(cfg: dict) -> GroundingDinoDetector:
    det = GroundingDinoDetector(score_thresh=cfg["gdino"]["score_thresh"], device="cpu")
    det.load()
    return det


def run_gdino_all(
    frames: list[np.ndarray],
    prompt: str,
    detector: GroundingDinoDetector,
    interval: int,
) -> list[tuple[np.ndarray, float] | None]:
    results: list[tuple[np.ndarray, float] | None] = [None] * len(frames)
    for i, bgr in enumerate(frames):
        if i % interval != 0:
            continue
        dr = detector.detect(_to_rgbd(bgr), prompt)
        if dr.detections:
            best = dr.detections[0]
            results[i] = (best.bbox_2d, best.score)
    return results


# ── Per-tracker run ───────────────────────────────────────────────────────────

def run_tracker(
    tracker: BenchmarkTracker,
    frames: list[np.ndarray],
    gdino_dets: list[tuple[np.ndarray, float] | None],
) -> tuple[list[np.ndarray | None], list[float]]:
    tracker.reset()
    preds: list[np.ndarray | None] = []
    timings: list[float] = []
    for bgr, det in zip(frames, gdino_dets):
        t0 = time.perf_counter()
        bbox_out, _ = tracker.step(bgr, det[0] if det else None, det[1] if det else None)
        timings.append((time.perf_counter() - t0) * 1000.0)
        preds.append(bbox_out)
    return preds, timings


# ── Visualisation ─────────────────────────────────────────────────────────────

def write_comparison_video(
    frames: list[np.ndarray],
    tracker_preds: dict[str, list[np.ndarray | None]],
    gt: list[np.ndarray | None],
    out_path: str,
    playback_fps: float,
) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    n_t = len(tracker_preds)
    panel_w = w // max(n_t, 1)
    strip_h = h // 3
    out_h = h + strip_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, playback_fps, (w, out_h))

    font  = cv2.FONT_HERSHEY_SIMPLEX
    fs    = max(0.4, w / 1800)
    thick = max(1, w // 600)

    for fi, bgr in enumerate(frames):
        canvas = np.zeros((out_h, w, 3), dtype=np.uint8)

        top = bgr.copy()
        if gt[fi] is not None:
            x1, y1, x2, y2 = gt[fi].astype(int)
            cv2.rectangle(top, (x1, y1), (x2, y2), (200, 200, 200), thick + 1)
            cv2.putText(top, "GT", (x1 + 2, max(y1 - 4, 14)), font, fs, (0, 0, 0), thick)
        cv2.putText(top, f"f{fi:04d}", (6, h - 6), font, fs * 0.8, (160, 160, 160), thick)
        canvas[:h] = top

        for ti, (name, preds) in enumerate(tracker_preds.items()):
            cx = ti * panel_w
            panel = cv2.resize(bgr, (panel_w, strip_h))
            pred = preds[fi]
            if pred is not None:
                sx, sy = panel_w / w, strip_h / h
                px1, py1 = int(pred[0]*sx), int(pred[1]*sy)
                px2, py2 = int(pred[2]*sx), int(pred[3]*sy)
                colour = COLOURS.get(name, (255, 255, 255))
                cv2.rectangle(panel, (px1, py1), (px2, py2), colour, max(1, thick-1))
            cv2.putText(panel, name, (4, 14), font, fs * 0.7,
                        COLOURS.get(name, (255, 255, 255)), thick)
            canvas[h:, cx:cx + panel_w] = panel

        writer.write(canvas)

    writer.release()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",    required=True)
    ap.add_argument("--prompt",   required=True)
    ap.add_argument("--gt-video", default=None,
                    help="Clean companion video for GT. Without this, GDINO on "
                         "the test video at interval=1 is used as pseudo-GT.")
    ap.add_argument("--config", default=str(ROOT / "code/benchmark/config.yaml"))
    ap.add_argument("--out",    default=str(ROOT / "results/benchmark"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    interval     = cfg["gdino"]["detection_interval"]
    iou_thresh   = cfg["benchmark"]["iou_match_thresh"]
    play_fps     = float(cfg["benchmark"]["playback_fps"])

    print(f"Loading video: {args.video}")
    frames, src_fps = load_frames(args.video)
    h, w = frames[0].shape[:2]
    print(f"  {len(frames)} frames @ {src_fps:.1f}fps  ({w}×{h})")

    print(f"Loading GDINO ...")
    detector = build_detector(cfg)

    if args.gt_video:
        print(f"Building GT from: {args.gt_video}")
        gt_frames, _ = load_frames(args.gt_video)
        gt_raw = run_gdino_all(gt_frames, args.prompt, detector, interval=1)
    else:
        print("No --gt-video — using per-frame GDINO on test video as pseudo-GT.")
        gt_raw = run_gdino_all(frames, args.prompt, detector, interval=1)
    gt: list[np.ndarray | None] = [d[0] if d else None for d in gt_raw]

    print(f"Running GDINO on test video (every {interval} frames) ...")
    gdino_dets = run_gdino_all(frames, args.prompt, detector, interval)

    trackers: list[BenchmarkTracker] = [
        CsrtTracker(),
        ByteTrackTracker(cfg),
        ReidKalmanTracker(cfg),
    ]

    all_preds: dict[str, list[np.ndarray | None]] = {}
    all_metrics: list[TrackMetrics] = []

    for tracker in trackers:
        print(f"Running {tracker.name} ...")
        preds, timings = run_tracker(tracker, frames, gdino_dets)
        all_preds[tracker.name] = preds
        m = compute_metrics(tracker.name, preds, gt, tracker.id_switches,
                            timings, iou_thresh)
        all_metrics.append(m)

    print_table(all_metrics)

    stem    = Path(args.video).stem
    out_dir = Path(args.out) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump([vars(m) for m in all_metrics], f, indent=2)
    print(f"Metrics → {metrics_path}")

    if cfg["benchmark"]["visualize"]:
        vis = str(out_dir / "comparison_tracked.mp4")
        print(f"Writing comparison video → {vis}")
        write_comparison_video(frames, all_preds, gt, vis, play_fps)

    print("Done.")


if __name__ == "__main__":
    main()
