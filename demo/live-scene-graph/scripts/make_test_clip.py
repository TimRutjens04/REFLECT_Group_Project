"""Generate a short synthetic mp4 (moving coloured shapes) for headless e2e:

    uv run python scripts/make_test_clip.py /tmp/clip.mp4 --frames 90
    uv run python -m livescene.app --source /tmp/clip.mp4 --headless \
        --prompts "red square, blue rectangle, green circle" --out /tmp/out
"""

from __future__ import annotations

import argparse

import cv2

from livescene.synthetic import SYNTH_H, SYNTH_W, synthetic_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", help="output mp4 path")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (SYNTH_W, SYNTH_H)
    )
    for i in range(args.frames):
        writer.write(synthetic_frame(i))
    writer.release()
    print(f"wrote {args.frames} frames to {args.out}")


if __name__ == "__main__":
    main()
