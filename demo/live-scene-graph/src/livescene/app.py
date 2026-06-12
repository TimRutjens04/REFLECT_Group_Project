"""Entry point: live OpenCV split view or headless JSONL/mp4 export.

Live:     uv run python -m livescene.app --source 0
Headless: uv run python -m livescene.app --source synthetic --headless \
              --max-frames 60 --out /tmp/out
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

import cv2

from .config import AppConfig
from .pipeline import LivePipeline
from .prompts import PromptInput, parse_prompt_line
from .render import compose, draw_bbox_panel, draw_graph_panel
from .scene_graph.models import JsonlWriter
from .synthetic import SYNTH_PROMPTS, SyntheticSource

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class WebcamSource:
    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open webcam {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class VideoSource:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class ImageSource:
    """Repeat one still image forever (useful for tuning on a screenshot)."""

    def __init__(self, path: str):
        self.frame = cv2.imread(path)
        if self.frame is None:
            raise RuntimeError(f"cannot read image {path}")

    def read(self) -> np.ndarray | None:
        return self.frame.copy()

    def release(self) -> None:
        pass


class LatestFrameSource:
    """Capture thread that always holds only the newest frame.

    Keeps live inference latency from building a backlog: when processing is
    slower than the camera, stale frames are dropped instead of queued.
    Only used for the live webcam; file/synthetic sources must see every frame.
    """

    def __init__(self, src):
        self._src = src
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stopped:
            frame = self._src.read()
            if frame is None:
                self._stopped = True
                return
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray | None:
        while True:
            with self._lock:
                if self._frame is not None:
                    return self._frame
            if self._stopped:
                return None
            time.sleep(0.005)  # camera warming up; wait for the first frame

    def release(self) -> None:
        self._stopped = True
        self._thread.join(timeout=1.0)
        self._src.release()


def open_source(spec: str, cfg: AppConfig):
    if spec == "synthetic":
        return SyntheticSource()
    if spec.isdigit():
        return WebcamSource(int(spec), cfg.width, cfg.height)
    if Path(spec).suffix.lower() in _IMAGE_EXTS:
        return ImageSource(spec)
    return VideoSource(spec)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="livescene", description=__doc__)
    p.add_argument("--source", default="0",
                   help="webcam index (0), video path, image path, or 'synthetic'")
    p.add_argument("--prompts", default=None,
                   help="comma-separated detection prompts, e.g. 'apple, bowl, mug'")
    p.add_argument("--headless", action="store_true",
                   help="no window: write annotated mp4 + scene_graph.jsonl to --out")
    p.add_argument("--out", default="out", help="output dir for --headless")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = unlimited)")
    p.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    p.add_argument("--conf", type=float, default=None, help="YOLOE confidence threshold")
    p.add_argument("--hfov", type=float, default=None, help="camera horizontal FOV in degrees")
    p.add_argument("--depth-every-k", type=int, default=None, help="run depth every k frames")
    p.add_argument("--depth-input-long-side", type=int, default=None,
                   help="downscale depth input to this long side (speed)")
    p.add_argument("--yoloe-model", default=None)
    p.add_argument("--depth-model", default=None)
    p.add_argument("--width", type=int, default=None, help="webcam capture width")
    p.add_argument("--height", type=int, default=None, help="webcam capture height")
    return p


def config_from_args(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig()
    if args.device:
        cfg.device = args.device
    if args.conf is not None:
        cfg.conf = args.conf
    if args.hfov is not None:
        cfg.hfov_deg = args.hfov
    if args.depth_every_k is not None:
        cfg.depth_every_k = args.depth_every_k
    if args.depth_input_long_side is not None:
        cfg.depth_input_long_side = args.depth_input_long_side
    if args.yoloe_model:
        cfg.yoloe_model = args.yoloe_model
    if args.depth_model:
        cfg.depth_model = args.depth_model
    if args.width:
        cfg.width = args.width
    if args.height:
        cfg.height = args.height
    if args.prompts:
        cfg.prompts = parse_prompt_line(args.prompts)
    elif args.source == "synthetic":
        cfg.prompts = list(SYNTH_PROMPTS)
    return cfg


def run_headless(args: argparse.Namespace, cfg: AppConfig) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "scene_graph.jsonl"
    jsonl_path.unlink(missing_ok=True)  # JsonlWriter appends
    writer = JsonlWriter(jsonl_path)

    pipeline = LivePipeline(cfg)
    source = open_source(args.source, cfg)
    video_writer = None
    max_frames = args.max_frames or 300
    n_frames = 0
    edge_counts: dict[str, int] = {}
    t_start = time.perf_counter()
    last_view = None

    try:
        while n_frames < max_frames:
            frame = source.read()
            if frame is None:
                break
            sg = pipeline.process(frame)
            writer.write(sg)
            for e in sg.edges:
                edge_counts[e.relation] = edge_counts.get(e.relation, 0) + 1

            h, w = frame.shape[:2]
            view = compose(
                draw_bbox_panel(frame, sg), draw_graph_panel(sg, w, h), pipeline.prompts
            )
            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    str(out_dir / "annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    15.0,
                    (view.shape[1], view.shape[0]),
                )
            video_writer.write(view)
            last_view = view
            n_frames += 1
    finally:
        source.release()
        if video_writer is not None:
            video_writer.release()

    if last_view is not None:
        cv2.imwrite(str(out_dir / "last_frame.png"), last_view)
    elapsed = time.perf_counter() - t_start
    fps = n_frames / elapsed if elapsed > 0 else 0.0
    print(f"headless: {n_frames} frames in {elapsed:.1f}s ({fps:.1f} FPS) -> {out_dir}")
    print(f"edge counts: {edge_counts}")
    if n_frames == 0:
        print("error: source produced no frames", file=sys.stderr)
        return 1
    return 0


def run_live(args: argparse.Namespace, cfg: AppConfig) -> int:
    pipeline = LivePipeline(cfg)
    source = open_source(args.source, cfg)
    if isinstance(source, WebcamSource):
        source = LatestFrameSource(source)
    prompt_input = PromptInput(cfg.prompts)
    prompt_input.start()
    seen_version, _ = prompt_input.current()

    window = "live scene graph"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    print(f"prompts: {', '.join(cfg.prompts)}")
    print("type a new comma-separated prompt list + Enter to re-prompt; q in the window quits")

    fps = None
    n_frames = 0
    try:
        while True:
            version, names = prompt_input.current()
            if version != seen_version:
                seen_version = version
                print(f"re-prompting: {', '.join(names)}")
                pipeline.set_prompts(names)

            frame = source.read()
            if frame is None:
                break
            t0 = time.perf_counter()
            sg = pipeline.process(frame, timestamp=time.time())
            dt = time.perf_counter() - t0
            inst = 1.0 / dt if dt > 0 else 0.0
            fps = inst if fps is None else 0.9 * fps + 0.1 * inst

            h, w = frame.shape[:2]
            view = compose(
                draw_bbox_panel(frame, sg), draw_graph_panel(sg, w, h), pipeline.prompts, fps
            )
            cv2.imshow(window, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            n_frames += 1
            if args.max_frames and n_frames >= args.max_frames:
                break
    finally:
        source.release()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.headless:
        return run_headless(args, cfg)
    return run_live(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
