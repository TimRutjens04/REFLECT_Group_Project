"""
Manual-seed CSRT tracker demo.

Opens a video, shows the first frame for manual bbox selection, then runs
CSRT tracking through all frames and writes an annotated output video.

Usage:
    python code/demo_track.py <video>
    python code/demo_track.py <video> --label "coffee mug"
    python code/demo_track.py <video> --out visuals/my_output.mp4
    python code/demo_track.py <video> --fps 30

Controls (ROI window):
    Click + drag  — draw bounding box
    SPACE / ENTER — confirm and start tracking
    ESC / c       — cancel and exit
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIS_DIR = os.path.join(ROOT, "visuals", "demo")

_GREEN = (50,  205, 50)
_RED   = (0,   0,   220)


def _parse_args() -> tuple[str, str, str, int | None]:
    args  = sys.argv[1:]
    label = "object"
    out   = None
    fps   = None

    if "--label" in args:
        i     = args.index("--label")
        label = args[i + 1]
        args  = args[:i] + args[i + 2:]

    if "--out" in args:
        i    = args.index("--out")
        out  = args[i + 1]
        args = args[:i] + args[i + 2:]

    if "--fps" in args:
        i   = args.index("--fps")
        fps = int(args[i + 1])
        args = args[:i] + args[i + 2:]

    if not args:
        print("Usage: python code/demo_track.py <video> [--label NAME] [--out PATH] [--fps N]")
        sys.exit(1)

    video_path = args[0]
    if not os.path.isabs(video_path):
        video_path = os.path.join(ROOT, video_path)

    if out is None:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        slug = label.replace(" ", "_")
        out  = os.path.join(VIS_DIR, f"{stem}_{slug}_tracked.mp4")

    return video_path, label, out, fps


def _put_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    font: int,
    scale: float,
    thickness: int,
    colour: tuple[int, int, int],
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, font, scale * 0.7, thickness)
    ty = max(y - 4, th + 4)
    cv2.rectangle(img, (x, ty - th - baseline - 2), (x + tw + 4, ty + 2), colour, -1)
    cv2.putText(img, text, (x + 2, ty - baseline), font, scale * 0.7, (0, 0, 0), thickness)


def main() -> None:
    video_path, label, out_path, fps_override = _parse_args()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        print("If this is a HEVC .MOV, convert it first:")
        print("  ffmpeg -i IMG_0081.MOV -c:v libx264 IMG_0081.mp4")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = float(fps_override) if fps_override else src_fps

    ok, frame0 = cap.read()
    if not ok:
        print("Error: could not read first frame")
        sys.exit(1)

    # Get dimensions from the decoded frame, not cap.get(). iPhone MOVs embed a
    # rotation flag so cap.get() returns pre-rotation dimensions that are swapped
    # vs. what the decoder outputs, causing VideoWriter to produce corrupt frames.
    h, w = frame0.shape[:2]

    # ── Manual bbox selection ──────────────────────────────────────────────────
    print(f"\nDraw a bounding box around '{label}' on the first frame.")
    print("  Click + drag to draw  |  SPACE / ENTER to confirm  |  ESC to cancel\n")

    roi = cv2.selectROI(
        f"Select '{label}' — SPACE to confirm, ESC to cancel",
        frame0,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyAllWindows()

    x, y, bw, bh = roi
    if bw == 0 or bh == 0:
        print("No selection made — exiting.")
        sys.exit(0)

    print(f"  bbox: x={x} y={y} w={bw} h={bh}")

    # ── Init CSRT ─────────────────────────────────────────────────────────────
    tracker = cv2.legacy.TrackerCSRT_create()
    tracker.init(frame0, (float(x), float(y), float(bw), float(bh)))

    # ── Output writer ─────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, w / 1200)
    thickness  = max(1, w // 400)

    # Annotate and write frame 0
    f0 = frame0.copy()
    cv2.rectangle(f0, (x, y), (x + bw, y + bh), _GREEN, thickness + 1)
    _put_label(f0, f"{label}  [manual seed]", x, y, font, font_scale, thickness, _GREEN)
    cv2.putText(f0, "frame 0000", (8, h - 10), font, font_scale * 0.7, (180, 180, 180), thickness)
    writer.write(f0)

    # ── Track frames 1 … N-1 ──────────────────────────────────────────────────
    fi         = 1
    lost_since = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ok_track, (rx, ry, rw, rh) = tracker.update(frame)
        annotated = frame.copy()

        if ok_track and rw > 0 and rh > 0:
            lost_since = 0
            x1, y1 = int(rx), int(ry)
            x2, y2 = int(rx + rw), int(ry + rh)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _GREEN, thickness + 1)
            _put_label(annotated, label, x1, y1, font, font_scale, thickness, _GREEN)
        else:
            lost_since += 1
            cv2.putText(annotated, f"[lost {lost_since}f]", (8, 40),
                        font, font_scale, _RED, thickness)

        cv2.putText(annotated, f"frame {fi:04d}", (8, h - 10),
                    font, font_scale * 0.7, (180, 180, 180), thickness)

        writer.write(annotated)
        fi += 1

    cap.release()
    writer.release()
    print(f"Done. {fi} frames tracked → {out_path}")


if __name__ == "__main__":
    main()
