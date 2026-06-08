"""SAM2 video tracking in windows: re-detect with GDINO every N frames, track, repeat.

Each window gets a fresh predictor, so VRAM is bounded and objects that left/
reappeared are re-detected. Object IDs are window-local (they reset each window).
"""

from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from models.detection import TriggerReason

_COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (128, 0, 255),
    (0, 255, 255),
]


def track_video_with_sam2(
    video_path: Path,
    output_path: Path,
    detection_runner,
    provider,
    task,
    model_name: str = "sam2_t.pt",
    conf: float = 0.25,
    imgsz: int = 1024,
    redetect_every: int = 90,
) -> None:
    from ultralytics.models.sam import SAM2VideoPredictor

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        imgsz=imgsz,
        model=model_name,
        save=False,
        verbose=False,
    )

    global_idx = 0
    try:
        while True:
            window = _read_window(cap, redetect_every)
            if not window:
                break

            # Re-detect on the window's first frame (provider idx == video idx).
            frame = provider.get_frame(global_idx)
            det = detection_runner.run(
                frame, task, trigger_reason=TriggerReason.TRACKER_LOW_CONFIDENCE
            )

            if det.success and det.detections:
                bboxes = [list(map(float, d.bbox_2d)) for d in det.detections]
                labels = [d.label for d in det.detections]

                window_video = _write_window_video(window, fps, width, height)
                try:
                    predictor = SAM2VideoPredictor(overrides=overrides)
                    results = predictor(
                        source=str(window_video), bboxes=bboxes, stream=True
                    )
                    for frame_bgr, result in zip(window, results):
                        writer.write(_draw(frame_bgr, result, labels))
                finally:
                    window_video.unlink(missing_ok=True)

                del predictor, results
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                for f in window:
                    writer.write(f)

            print(f"  frames {global_idx}-{global_idx + len(window)}")
            global_idx += len(window)
    finally:
        cap.release()
        writer.release()


def _read_window(cap, n: int) -> List[np.ndarray]:
    frames = []
    for _ in range(n):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    return frames


def _write_window_video(
    window: List[np.ndarray], fps: float, width: int, height: int
) -> Path:
    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    video_path = Path(temp_path)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        for frame in window:
            writer.write(frame)
    finally:
        writer.release()
    return video_path


def _draw(frame_bgr, result, labels: List[str]) -> np.ndarray:
    out = frame_bgr.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out
    for k, box in enumerate(result.boxes):
        oid = int(box.id[0].cpu()) if box.id is not None else k
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        color = _COLORS[oid % len(_COLORS)]
        label = labels[oid] if 0 <= oid < len(labels) else f"obj{oid}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{label} #{oid}",
            (x1, max(y1 - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out
