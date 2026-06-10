"""SAM2 visual-prompt video tracking using Grounding DINO detections as box prompts.

Grounding DINO runs on frame 0 to produce bounding boxes. Those boxes are fed to
SAM2's video predictor as prompts on the first frame; SAM2 then propagates them
through the rest of the video via its memory mechanism (no per-frame re-prompting).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np

from interfaces.IDetection import DetectionResult


# Distinct BGR colors for overlaying up to N objects.
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


def _detection_to_bboxes(
    detection_result: DetectionResult,
) -> Tuple[List[List[float]], List[str]]:
    """Convert Grounding DINO detections to SAM2 `bboxes` (xyxy) + label names.

    Returns `(bboxes, label_names)` where `bboxes` is a list of `[x1, y1, x2, y2]`
    (one per object) and `label_names[i]` is the label for object `i`. Object order
    here defines the SAM2 object id order on frame 0.
    """
    dets = detection_result.detections
    bboxes = [list(map(float, d.bbox_2d)) for d in dets]
    label_names = [d.label for d in dets]
    return bboxes, label_names


def _video_meta(video_path: Path) -> Tuple[float, int, int]:
    """Read fps, width, height from the video without consuming SAM2's reader."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height


def track_video_with_sam2(
    video_path: Path,
    detection_result: DetectionResult,
    output_path: Path,
    model_name: str = "sam2_b.pt",
    conf: float = 0.25,
    imgsz: int = 1024,
    frame_step: int = 1,
) -> None:
    """Track objects through `video_path` with SAM2, prompted by Grounding DINO boxes.

    The frame-0 boxes are passed once as `bboxes`; SAM2 propagates them through the
    whole video. `frame_step` only subsamples which frames are *written* to the output
    (SAM2 still processes every frame internally — its memory needs consecutive frames).
    """
    from ultralytics.models.sam import SAM2VideoPredictor

    if not detection_result.success or not detection_result.detections:
        raise ValueError("Grounding DINO produced no detections.")

    bboxes, label_names = _detection_to_bboxes(detection_result)

    fps, width, height = _video_meta(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fps = fps / max(frame_step, 1)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
    )

    overrides = dict(
        conf=conf, task="segment", mode="predict", imgsz=imgsz, model=model_name
    )
    predictor = SAM2VideoPredictor(overrides=overrides)

    try:
        # Single call: prompt on frame 0, SAM2 streams results for every frame.
        results = predictor(source=str(video_path), bboxes=bboxes, stream=True)

        for frame_idx, result in enumerate(results):
            if frame_idx % frame_step != 0:
                continue

            frame_bgr = result.orig_img  # BGR ndarray for this frame
            annotated = _draw_tracks(frame_bgr, result, label_names)

            if frame_idx % 100 == 0:
                n = 0 if result.boxes is None else len(result.boxes)
                print(f"  frame {frame_idx}: {n} masks/boxes")

            writer.write(annotated)
    finally:
        writer.release()


def _draw_tracks(frame_bgr: np.ndarray, result, label_names: List[str]) -> np.ndarray:
    """Overlay SAM2 masks, boxes, ids and Grounding DINO labels onto the frame (BGR)."""
    out = frame_bgr.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return out

    masks = None
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # (N, H, W)

    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        track_id = int(box.id[0].cpu()) if box.id is not None else i
        color = _COLORS[track_id % len(_COLORS)]

        # SAM2 object id follows the frame-0 prompt order, so it indexes label_names.
        label = (
            label_names[track_id]
            if 0 <= track_id < len(label_names)
            else f"obj{track_id}"
        )

        if masks is not None and i < len(masks):
            m = masks[i].astype(bool)
            if m.shape[:2] == out.shape[:2]:
                out[m] = (out[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{label} #{track_id}",
            (x1, max(y1 - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return out
