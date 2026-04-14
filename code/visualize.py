#!/usr/bin/env python3
"""
Visualization — Per-Stage Overlay Video

For each episode renders a side-by-side MP4:
  Left   : RGB frame + Grounding DINO bboxes + SAM 2 mask overlays
           + track IDs + held_by_gripper indicator
  Right  : Depth Anything V2 depth map (plasma colormap) with object
           state labels printed at each object centroid
  Footer : frame number, timestamp, failure label, localization flag,
           object state summary

Output: visuals/<episode>.mp4

Usage
-----
  poetry run python3 code/visualize.py                   # all episodes
  poetry run python3 code/visualize.py boilWater-1       # one episode
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import subprocess

import cv2
import numpy as np
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ALIGNED_DIR      = ROOT / "aligned"
DETECT_DIR       = ROOT / "detect"
SEGMENT_DIR      = ROOT / "segment"
DEPTH_STATE_DIR  = ROOT / "depth_state"
TRACK_DIR        = ROOT / "track"
GRAPHS_DIR       = ROOT / "scene_graphs_pipeline"
VISUALS_DIR      = ROOT / "visuals"
VISUALS_DIR.mkdir(exist_ok=True)

# ── layout ─────────────────────────────────────────────────────────────────────
FOOTER_H    = 90       # pixels — info strip below the two panels
MASK_ALPHA  = 0.35     # opacity for mask overlays
FONT        = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_FPS   = 4        # playback fps regardless of capture fps (slow enough to read)

# ── colour palette — 32 visually distinct BGR colours ──────────────────────────
# Generated from HSV with fixed S/V, evenly spaced hues
def _make_palette(n: int = 32) -> list[tuple[int, int, int]]:
    palette = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        palette.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return palette

PALETTE = _make_palette(32)


def _track_color(track_id: int) -> tuple[int, int, int]:
    return PALETTE[track_id % len(PALETTE)]


def _plasma(value: float) -> tuple[int, int, int]:
    """Map a [0,1] float to BGR using a hand-sampled plasma colormap."""
    PLASMA = [          # BGR samples at 0, 0.25, 0.5, 0.75, 1.0
        (148, 12, 13),
        (96, 37, 131),
        (41, 127, 185),
        (37, 210, 148),
        (13, 240, 240),
    ]
    v = max(0.0, min(1.0, value))
    idx = v * (len(PLASMA) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(PLASMA) - 1)
    t = idx - lo
    b = int(PLASMA[lo][0] * (1 - t) + PLASMA[hi][0] * t)
    g = int(PLASMA[lo][1] * (1 - t) + PLASMA[hi][1] * t)
    r = int(PLASMA[lo][2] * (1 - t) + PLASMA[hi][2] * t)
    return (b, g, r)


# ── rendering helpers ──────────────────────────────────────────────────────────

def _apply_masks(
    frame: np.ndarray,          # (H, W, 3) uint8 — modified in-place clone
    masks_small: np.ndarray,    # (MAX_DET, Hs, Ws) uint8
    mask_valid: np.ndarray,     # (MAX_DET,) bool
    track_ids: np.ndarray,      # (MAX_DET,) int32
    n_dets: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Blend per-object masks onto the frame as transparent colour overlays."""
    out = frame.copy()
    for j in range(n_dets):
        if not mask_valid[j]:
            continue
        tid = int(track_ids[j])
        color = _track_color(tid)
        # resize mask to full frame resolution
        mask_full = cv2.resize(
            masks_small[j].astype(np.uint8),
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        overlay = out.copy()
        overlay[mask_full] = color
        out = cv2.addWeighted(out, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
    return out


def _draw_detections(
    frame: np.ndarray,   # (H, W, 3) uint8 — modified in-place
    boxes: np.ndarray,   # (MAX_DET, 4) float32
    n_dets: int,
    label_vocab: list[str],
    label_ids: np.ndarray,   # (MAX_DET,) int32
    track_ids: np.ndarray,   # (MAX_DET,) int32
    track_conf: np.ndarray,  # (MAX_DET,) float32
    held: np.ndarray,        # (MAX_DET,) bool
    state_vocab: list[str],
    obj_states: np.ndarray,  # (MAX_DET,) int32
) -> None:
    for j in range(n_dets):
        tid = int(track_ids[j])
        if tid < 0:
            continue
        color = _track_color(tid)
        x1, y1, x2, y2 = [int(v) for v in boxes[j]]

        # bbox — thick if held
        thickness = 3 if held[j] else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # held indicator — yellow star marker at top-centre
        if held[j]:
            cx = (x1 + x2) // 2
            cv2.drawMarker(frame, (cx, y1 - 6), (0, 220, 220),
                           cv2.MARKER_STAR, 14, 2)

        # label text: "<label> #<tid>"
        lid = int(label_ids[j])
        lbl = label_vocab[lid] if 0 <= lid < len(label_vocab) else "?"
        sid = int(obj_states[j])
        state = state_vocab[sid] if 0 <= sid < len(state_vocab) else ""
        text = f"{lbl} #{tid} {state}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        ty = max(y1 - 4, th + 2)
        cv2.rectangle(frame, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), color, -1)
        cv2.putText(frame, text, (x1 + 1, ty), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _render_depth_panel(
    depth_map: np.ndarray,   # (Hd, Wd) float32
    H: int,
    W: int,
    boxes: np.ndarray,       # (MAX_DET, 4)
    n_dets: int,
    track_ids: np.ndarray,
    obj_depth: np.ndarray,   # (MAX_DET,) float32
    state_vocab: list[str],
    obj_states: np.ndarray,  # (MAX_DET,) int32
) -> np.ndarray:
    """Render plasma depth map at (H, W) with depth values at object centroids."""
    # resize depth to panel size
    depth_rs = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
    panel = np.zeros((H, W, 3), dtype=np.uint8)
    for row in range(H):
        for col in range(W):
            panel[row, col] = _plasma(float(depth_rs[row, col]))

    # print depth value and state at each object centroid
    for j in range(n_dets):
        tid = int(track_ids[j])
        if tid < 0:
            continue
        x1, y1, x2, y2 = [int(v) for v in boxes[j]]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        d = float(obj_depth[j])
        depth_txt = f"{d:.2f}" if not np.isnan(d) else "?"
        sid = int(obj_states[j])
        state = state_vocab[sid] if 0 <= sid < len(state_vocab) else ""
        label = f"d={depth_txt} {state}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.4, 1)
        cv2.rectangle(panel, (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2),
                      (30, 30, 30), -1)
        cv2.putText(panel, label, (cx, cy), FONT, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    # colour bar legend (right edge)
    bar_w = 12
    for row in range(H):
        v = 1.0 - row / H
        panel[row, W - bar_w:W] = _plasma(v)
    cv2.putText(panel, "far",  (W - bar_w - 22, 12),   FONT, 0.35, (200, 200, 200), 1)
    cv2.putText(panel, "near", (W - bar_w - 28, H - 5), FONT, 0.35, (200, 200, 200), 1)

    return panel


def _render_footer(
    frame_idx: int,
    timestamp: float,
    fail: bool,
    flag: dict,
    n_objs: int,
    relations: list[dict],
    W_total: int,
) -> np.ndarray:
    """Render the info strip below the two panels."""
    strip = np.zeros((FOOTER_H, W_total, 3), dtype=np.uint8)
    strip[:] = (30, 30, 30)

    # failure highlight
    if fail:
        strip[:, :] = (0, 0, 80)   # dark red tint
        cv2.rectangle(strip, (0, 0), (W_total - 1, FOOTER_H - 1), (0, 0, 200), 3)

    # left: frame / timestamp
    cv2.putText(strip, f"Frame {frame_idx:3d}  t={timestamp:.1f}s",
                (10, 22), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    status = "FAILURE" if fail else "normal"
    status_color = (50, 50, 220) if fail else (100, 200, 100)
    cv2.putText(strip, status, (10, 52), FONT, 0.65, status_color, 2, cv2.LINE_AA)

    # centre: localization flag
    flag_type = flag.get("type") or "—"
    flag_detected = flag.get("failure_detected", False)
    flag_color = (50, 50, 220) if flag_detected else (120, 120, 120)
    flag_txt = f"Flag: {flag_type}" if flag_detected else "Flag: —"
    cv2.putText(strip, flag_txt, (W_total // 2 - 80, 30),
                FONT, 0.6, flag_color, 2, cv2.LINE_AA)
    ids = flag.get("affected_object_ids", [])
    if ids:
        cv2.putText(strip, f"affected ids: {ids}", (W_total // 2 - 80, 58),
                    FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # right: relation count
    rel_types: dict[str, int] = {}
    for r in relations:
        rel_types[r["relation"]] = rel_types.get(r["relation"], 0) + 1
    rel_summary = "  ".join(f"{k}:{v}" for k, v in sorted(rel_types.items()))
    cv2.putText(strip, f"objs:{n_objs}  rels:{len(relations)}",
                (W_total - 280, 28), FONT, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(strip, rel_summary[:50],
                (W_total - 280, 56), FONT, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    # panel divider
    cv2.line(strip, (W_total // 2, 0), (W_total // 2, FOOTER_H), (60, 60, 60), 1)

    return strip


# ── per-episode entry point ────────────────────────────────────────────────────

def visualize_episode(episode_id: str) -> None:
    aligned_path     = ALIGNED_DIR     / f"{episode_id}.npz"
    detect_path      = DETECT_DIR      / f"{episode_id}.npz"
    segment_path     = SEGMENT_DIR     / f"{episode_id}.npz"
    depth_state_path = DEPTH_STATE_DIR / f"{episode_id}.npz"
    track_path       = TRACK_DIR       / f"{episode_id}.npz"
    graph_path       = GRAPHS_DIR      / f"{episode_id}.json"
    out_path         = VISUALS_DIR     / f"{episode_id}.mp4"

    # require at minimum: aligned + detect
    missing = [n for p, n in [(aligned_path, "aligned"), (detect_path, "detect")]
               if not p.exists()]
    if missing:
        print(f"  [skip] {episode_id}: missing {missing}")
        return
    if out_path.exists():
        print(f"  [skip] {episode_id}: already rendered")
        return

    # ── load data ──────────────────────────────────────────────────────────────
    aligned = np.load(aligned_path, allow_pickle=True)
    det     = np.load(detect_path,  allow_pickle=True)

    frames:         np.ndarray = aligned["frames"]          # (N, H, W, 3)
    timestamps:     np.ndarray = aligned["timestamps"]
    failure_labels: np.ndarray = aligned["failure_labels"]

    all_boxes:     np.ndarray = det["boxes"]                # (N, MAX_DET, 4)
    all_n_dets:    np.ndarray = det["n_dets"]               # (N,)
    label_vocab:   list[str]  = list(det["label_vocab"])
    all_label_ids: np.ndarray = det["label_ids"]            # (N, MAX_DET)

    N, H, W, _ = frames.shape

    # optional: segment
    has_seg = segment_path.exists()
    if has_seg:
        seg = np.load(segment_path, allow_pickle=True)
        all_masks_small: np.ndarray = seg["masks_small"]   # (N, MAX_DET, Hs, Ws)
        all_mask_valid:  np.ndarray = seg["mask_valid"]     # (N, MAX_DET)
    else:
        all_masks_small = np.zeros((N, 1, 1, 1), dtype=np.uint8)
        all_mask_valid  = np.zeros((N, 1), dtype=bool)

    # optional: depth + state
    has_ds = depth_state_path.exists()
    if has_ds:
        ds = np.load(depth_state_path, allow_pickle=True)
        all_depth_maps: np.ndarray = ds["depth_maps"]      # (N, Hd, Wd)
        all_obj_depth:  np.ndarray = ds["obj_depth"]        # (N, MAX_DET)
        all_obj_state:  np.ndarray = ds["obj_state"]        # (N, MAX_DET)
        state_vocab:    list[str]   = list(ds["state_vocab"])
    else:
        all_depth_maps = np.zeros((N, H // 2, W // 2), dtype=np.float32)
        all_obj_depth  = np.full((N, 1), np.nan, dtype=np.float32)
        all_obj_state  = np.full((N, 1), -1, dtype=np.int32)
        state_vocab    = []

    # optional: track
    has_trk = track_path.exists()
    if has_trk:
        trk = np.load(track_path, allow_pickle=True)
        all_track_ids:  np.ndarray = trk["track_ids"]              # (N, MAX_DET)
        all_track_conf: np.ndarray = trk["tracking_confidence"]    # (N, MAX_DET)
        all_held:       np.ndarray = trk["held_by_gripper"]         # (N, MAX_DET)
    else:
        all_track_ids  = np.arange(1, dtype=np.int32).reshape(1, 1).repeat(N, 0)
        all_track_conf = np.ones((N, 1), dtype=np.float32)
        all_held       = np.zeros((N, 1), dtype=bool)

    # optional: scene graph JSON
    frame_graphs: list[dict] = []
    if graph_path.exists():
        with open(graph_path) as f:
            graph_data = json.load(f)
        frame_graphs = graph_data.get("frames", [])
    empty_flag = {"failure_detected": False, "type": None, "affected_object_ids": []}

    # ── ffmpeg pipe — H.264 output playable in QuickTime / Finder ─────────────
    W_total = W * 2
    H_total = H + FOOTER_H
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W_total}x{H_total}",
        "-pix_fmt", "bgr24",
        "-r", str(VIDEO_FPS),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",   # required for QuickTime compatibility
        "-crf", "20",            # quality (lower = better, 18-28 is typical)
        "-preset", "fast",
        str(out_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i in tqdm(range(N), desc=f"  {episode_id}", leave=False, unit="fr"):
        k    = int(all_n_dets[i])
        fail = bool(failure_labels[i])
        ts   = float(timestamps[i])

        frame_rgb = frames[i].copy()   # (H, W, 3) uint8 RGB

        # ── left panel: masks then bboxes ─────────────────────────────────────
        left = _apply_masks(
            frame_rgb,
            all_masks_small[i],
            all_mask_valid[i],
            all_track_ids[i],
            k,
            H, W,
        )
        _draw_detections(
            left,
            all_boxes[i],
            k,
            label_vocab,
            all_label_ids[i],
            all_track_ids[i],
            all_track_conf[i],
            all_held[i],
            state_vocab,
            all_obj_state[i],
        )
        if fail:
            cv2.rectangle(left, (0, 0), (W - 1, H - 1), (0, 0, 220), 6)
        cv2.putText(left, "Detection + Segmentation + Tracking",
                    (8, H - 10), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # ── right panel: depth map ─────────────────────────────────────────────
        right = _render_depth_panel(
            all_depth_maps[i],
            H, W,
            all_boxes[i],
            k,
            all_track_ids[i],
            all_obj_depth[i],
            state_vocab,
            all_obj_state[i],
        )
        if fail:
            cv2.rectangle(right, (0, 0), (W - 1, H - 1), (0, 0, 220), 6)
        cv2.putText(right, "Depth Map (plasma) + Object States",
                    (8, H - 10), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # ── footer ─────────────────────────────────────────────────────────────
        fg = frame_graphs[i] if i < len(frame_graphs) else {}
        flag = fg.get("localization_flag", empty_flag)
        relations = fg.get("spatial_relations", [])
        footer = _render_footer(i, ts, fail, flag, k, relations, W_total)

        # ── compose canvas (BGR for ffmpeg) ────────────────────────────────────
        canvas = np.zeros((H_total, W_total, 3), dtype=np.uint8)
        canvas[:H, :W]  = cv2.cvtColor(left,  cv2.COLOR_RGB2BGR)
        canvas[:H, W:]  = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
        canvas[H:, :]   = footer
        proc.stdin.write(canvas.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"  saved {out_path.name}  ({N} frames @ {VIDEO_FPS}fps, H.264)")


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

    print(f"Visualization | {len(episodes)} episodes → visuals/")
    for ep in tqdm(episodes, unit="ep"):
        print(f"\n▶ {ep}")
        visualize_episode(ep)
    print("\nDone.")


if __name__ == "__main__":
    main()
