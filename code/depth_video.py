"""
Run Depth Anything V2 Metric Indoor on a video and save per-frame depth maps.

Output: depth/<video_stem>.npz
  depth_maps  (N, H, W)  float32  — metric depth in metres
  fps         ()         float64  — source video FPS

Usage:
    poetry run python code/depth_video.py visuals/reid_can_demo.mp4
    poetry run python code/depth_video.py visuals/reid_can_demo.mp4 --model large
    poetry run python code/depth_video.py visuals/reid_can_demo.mp4 --force
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEPTH_DIR = os.path.join(ROOT, "depth")

_MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run(video_path: str, model_size: str = "small", force: bool = False) -> str:
    os.makedirs(DEPTH_DIR, exist_ok=True)

    stem     = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(DEPTH_DIR, f"{stem}.npz")

    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} already exists (use --force to rerun)")
        return out_path

    model_id = _MODEL_IDS[model_size]
    device   = _pick_device()
    print(f"Model  : {model_id}")
    print(f"Device : {device}")

    print("Loading model …")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model     = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.to(device).eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, bgr0 = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame")
    h, w = bgr0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    depth_maps = np.zeros((n_frames, h, w), dtype=np.float32)

    with torch.no_grad():
        for fi in tqdm(range(n_frames), desc="depth"):
            ok, bgr = cap.read()
            if not ok:
                depth_maps = depth_maps[:fi]
                break

            image  = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out   = model(**inputs)
            depth = out.predicted_depth.squeeze().cpu().numpy().astype(np.float32)

            # Resize from model output resolution back to source resolution
            depth_maps[fi] = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    cap.release()

    np.savez_compressed(out_path, depth_maps=depth_maps, fps=np.float64(fps))
    print(f"Saved  : {out_path}  ({len(depth_maps)} frames, {w}×{h}, metres)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to input video")
    parser.add_argument(
        "--model", choices=["small", "base", "large"], default="small",
        help="Model size (default: small)"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = os.path.join(ROOT, video_path)

    run(video_path, model_size=args.model, force=args.force)


if __name__ == "__main__":
    main()
