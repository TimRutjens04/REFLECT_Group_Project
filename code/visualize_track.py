"""
Tracking visualizer — REFLECT / RoboFail dataset.

Overlays CSRT tracking results on episode frames and writes an MP4.

Colour coding per bounding box:
  Green        — tracked cleanly (no failure flags)
  Yellow       — area-change or drift flag triggered
  Orange       — CSRT reported failure (bit 0)
  Red          — forced GDINO re-detection (bit 4)
  (flags OR'd: worst colour wins)

Each box is labelled:  ObjectName  conf:0.XX  [flag codes]

Usage:
    python visualize_track.py                    # all episodes in track/
    python visualize_track.py boilWater-1        # single episode
    python visualize_track.py boilWater-1 --fps 10
"""

import os
import sys

import cv2
import numpy as np

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACK_DIR   = os.path.join(ROOT, "track")
DETECT_DIR  = os.path.join(ROOT, "detect")
ALIGNED_DIR = os.path.join(ROOT, "aligned")
VIS_DIR     = os.path.join(ROOT, "visuals", "tracked")

# Failure bitmask constants (mirror track.py)
FLAG_CSRT_FAIL    = 1 << 0
FLAG_AREA_CHANGE  = 1 << 1
FLAG_DRIFT        = 1 << 2
FLAG_DEPTH_JUMP   = 1 << 3
FLAG_FORCED_REDET = 1 << 4
FLAG_FROZEN       = 1 << 5

# BGR colours
_COLOUR_CLEAN  = (50,  205, 50)    # lime green — clean tracking
_COLOUR_WARN   = (0,   200, 255)   # yellow     — soft failure flag
_COLOUR_FAIL   = (0,   128, 255)   # orange     — CSRT lost
_COLOUR_REDET  = (0,   0,   220)   # red        — GDINO re-detection fired
_COLOUR_FROZEN = (180, 180, 180)   # grey       — frozen, holding last position

_FLAG_CODES = {
    FLAG_CSRT_FAIL:    "CSRT",
    FLAG_AREA_CHANGE:  "AREA",
    FLAG_DRIFT:        "DRFT",
    FLAG_DEPTH_JUMP:   "DPTH",
    FLAG_FORCED_REDET: "REDT",
    FLAG_FROZEN:       "FRZN",
}

DEFAULT_PLAYBACK_FPS = 5   # 1fps source looks too slow; 5fps is watchable


def _box_colour(flags: int) -> tuple[int, int, int]:
    if flags & FLAG_FORCED_REDET:
        return _COLOUR_REDET
    if flags & FLAG_FROZEN:
        return _COLOUR_FROZEN
    if flags & FLAG_CSRT_FAIL:
        return _COLOUR_FAIL
    if flags & (FLAG_AREA_CHANGE | FLAG_DRIFT | FLAG_DEPTH_JUMP):
        return _COLOUR_WARN
    return _COLOUR_CLEAN


def _flag_str(flags: int) -> str:
    return " ".join(v for k, v in _FLAG_CODES.items() if flags & k)


def visualize_episode(
    episode_id: str,
    playback_fps: int,
    object_prompts: list[str] | None = None,
) -> None:
    if object_prompts:
        slug = "_".join(p.replace(" ", "_") for p in object_prompts)
        track_name = f"{episode_id}_{slug}.npz"
        out_name   = f"{episode_id}_{slug}_tracked.mp4"
    else:
        track_name = f"{episode_id}.npz"
        out_name   = f"{episode_id}_tracked.mp4"

    track_path   = os.path.join(TRACK_DIR,   track_name)
    aligned_path = os.path.join(ALIGNED_DIR, f"{episode_id}.npz")
    out_path     = os.path.join(VIS_DIR,     out_name)

    track   = np.load(track_path,   allow_pickle=False)
    aligned = np.load(aligned_path, allow_pickle=False)

    frames         = aligned["frames"]            # (N, H, W, 3) uint8
    failure_labels = aligned["failure_labels"]    # (N,) bool

    boxes  = track["boxes"]                # (N, n_obj, 4)
    flags  = track["failure_flags"]        # (N, n_obj) uint8
    conf   = track["tracking_confidence"]  # (N, n_obj)
    vocab  = list(track["label_vocab"])    # stored in track npz (self-contained)
    n_obj  = len(vocab)
    n      = len(frames)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, playback_fps, (w, h))

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, w / 1200)
    thickness  = max(1, w // 400)

    for fi in range(n):
        frame_bgr = cv2.cvtColor(frames[fi], cv2.COLOR_RGB2BGR).copy()

        # ── Failure keyframe banner ──────────────────────────────────────────
        if failure_labels[fi]:
            cv2.rectangle(frame_bgr, (0, 0), (w, 36), (0, 0, 180), -1)
            cv2.putText(frame_bgr, "FAILURE KEYFRAME", (8, 26),
                        font, font_scale * 1.0, (255, 255, 255), thickness)

        # ── Frame counter ────────────────────────────────────────────────────
        cv2.putText(frame_bgr, f"frame {fi:03d}", (8, h - 10),
                    font, font_scale * 0.7, (200, 200, 200), thickness)

        # ── Per-object bounding boxes ────────────────────────────────────────
        for oi in range(n_obj):
            bbox = boxes[fi, oi]
            if np.isnan(bbox).any():
                continue

            x1, y1, x2, y2 = bbox.astype(int)
            obj_flags = int(flags[fi, oi])
            colour    = _box_colour(obj_flags)
            c         = float(conf[fi, oi])
            label     = vocab[oi]
            flag_str  = _flag_str(obj_flags)

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, thickness + 1)

            text = f"{label}  {c:.2f}"
            if flag_str:
                text += f"  [{flag_str}]"
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale * 0.7, thickness)
            ty = max(y1 - 4, th + 4)
            cv2.rectangle(frame_bgr,
                          (x1, ty - th - baseline - 2),
                          (x1 + tw + 4, ty + 2),
                          colour, -1)
            cv2.putText(frame_bgr, text, (x1 + 2, ty - baseline),
                        font, font_scale * 0.7, (0, 0, 0), thickness)

        writer.write(frame_bgr)

    writer.release()
    print(f"  saved {out_path}  ({n} frames @ {playback_fps}fps)")


def main() -> None:
    os.makedirs(VIS_DIR, exist_ok=True)

    args    = sys.argv[1:]
    fps_arg = DEFAULT_PLAYBACK_FPS

    if "--fps" in args:
        idx     = args.index("--fps")
        fps_arg = int(args[idx + 1])
        args    = args[:idx] + args[idx + 2:]

    object_prompts: list[str] | None = None
    if "--object" in args:
        idx            = args.index("--object")
        object_prompts = [p.strip() for p in args[idx + 1].split(".") if p.strip()]
        args           = args[:idx] + args[idx + 2:]

    target = args[0] if args else None

    if target:
        episodes = [target]
    else:
        if object_prompts:
            slug   = "_".join(p.replace(" ", "_") for p in object_prompts)
            suffix = f"_{slug}.npz"
        else:
            suffix = ".npz"
        episodes = sorted(
            f[:-len(suffix)] for f in os.listdir(TRACK_DIR) if f.endswith(suffix)
        )

    prompt_info = f"  objects={object_prompts}" if object_prompts else ""
    print(f"Rendering {len(episodes)} episode(s) at {fps_arg}fps{prompt_info} ...")
    for ep in episodes:
        if not os.path.exists(os.path.join(ALIGNED_DIR, f"{ep}.npz")):
            print(f"  [skip] {ep} — no aligned file")
            continue
        print(f"  {ep}")
        visualize_episode(ep, fps_arg, object_prompts=object_prompts)

    print("Done.")


if __name__ == "__main__":
    main()
