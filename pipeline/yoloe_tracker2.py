"""YOLOe visual-prompt tracking on a full video using Grounding DINO detections as prompts.

This module provides helpers to convert Grounding DINO detection results into the
visual prompt format expected by YOLOe, run a single debug prediction on frame 0,
and track objects through a video using the baked visual prompts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np

from interfaces.IDetection import DetectionResult
from models.base import JsonlWriter
from models.tracking import TrackedObject, TrackingFlags, TrackingFrame, TrackerStatus


def _detection_to_visual_prompts(
    detection_result: DetectionResult,
) -> Tuple[dict, List[str]]:
    """Convert Grounding DINO detections to the YOLOe `visual_prompts` dict format.

    Returns a tuple `(visual_prompts, label_names)` where `visual_prompts` is
    `{"bboxes": np.ndarray, "cls": np.ndarray}` and `label_names` is a list
    of label strings corresponding to classes.
    """
    dets = detection_result.detections
    if not dets:
        return {
            "bboxes": np.zeros((0, 4), dtype=np.float32),
            "cls": np.array([], dtype=np.int64),
        }, []

    bboxes = np.array([d.bbox_2d for d in dets], dtype=np.float32)
    cls = np.arange(len(dets), dtype=np.int64)
    label_names = [d.label for d in dets]
    return {"bboxes": bboxes, "cls": cls}, label_names


def debug_yoloe_frame0(
    video_path: Path,
    detection_result: DetectionResult,
    output_path: Path,
    model_name: str = "yoloe-11l-seg.pt",
    conf: float = 0.1,
) -> None:
    """Run a single YOLOe visual-prompt predict on frame 0 and save annotated image.

    This is useful for verifying that visual prompts produce detections before
    running the full video. Uses `result.plot()` so masks, boxes, and labels are
    all rendered.
    """
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    if not detection_result.success or not detection_result.detections:
        raise ValueError("No detections to build visual prompts from.")

    visual_prompts, label_names = _detection_to_visual_prompts(detection_result)

    cap = cv2.VideoCapture(str(video_path))
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame 0 from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    model = YOLOE(model_name)
    results = model.predict(
        frame_rgb,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=conf,
        verbose=True,
    )

    result = results[0]
    annotated_bgr = result.plot(
        masks=False
    )  # BGR ndarray with boxes/masks/labels drawn
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated_bgr)


def track_video_with_yoloe(
    video_path: Path,
    detection_result: DetectionResult,
    output_path: Path,
    model_name: str = "yoloe-11l-seg.pt",
    frame_step: int = 1,
    sequence_id: str = "unknown",
    detection_writer: JsonlWriter | None = None,
    tracking_writer: JsonlWriter | None = None,
) -> None:
    """Track objects through `video_path` using YOLOe with visual prompts.

    The function primes the model with frame 0, then calls
    `model.track()` on subsequent frames and writes an annotated output video.
    Optionally writes per-frame DetectionFrame and TrackingFrame rows to JSONL.
    """
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    if not detection_result.success or not detection_result.detections:
        raise ValueError("Grounding DINO produced no detections.")

    visual_prompts, label_names = _detection_to_visual_prompts(detection_result)
    model = YOLOE(model_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ret, ref_bgr = cap.read()
    if not ret:
        raise RuntimeError("Could not read reference frame 0.")
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

    model.predict(
        source=ref_rgb,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        refer_image=ref_rgb,
        conf=0.1,
        verbose=False,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # Per-track state for computing derived tracking fields
    _init_areas: dict[int, float] = {}
    _prev_centers: dict[int, tuple[float, float]] = {}
    _first_seen: dict[int, int] = {}

    try:
        frame_idx = tracker_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = model.track(
                    frame_rgb,
                    persist=True,
                    conf=0.1,
                    tracker="botsort.yaml",
                    verbose=False,
                )
                if tracker_idx % 100 == 0:
                    n = 0 if results[0].boxes is None else len(results[0].boxes)
                    print(f"  frame {frame_idx}/{n_frames}: {n} boxes")
                writer.write(_draw_tracks(frame_bgr, results[0], label_names))

                if tracking_writer:
                    _write_jsonl_rows(
                        result=results[0],
                        frame_idx=frame_idx,
                        fps=fps,
                        sequence_id=sequence_id,
                        label_names=label_names,
                        init_areas=_init_areas,
                        prev_centers=_prev_centers,
                        first_seen=_first_seen,
                        tracking_writer=tracking_writer,
                    )

                tracker_idx += 1
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def _write_jsonl_rows(
    result,
    frame_idx: int,
    fps: float,
    sequence_id: str,
    label_names: List[str],
    init_areas: dict,
    prev_centers: dict,
    first_seen: dict,
    tracking_writer: JsonlWriter | None,
) -> None:
    """Build and write a TrackingFrame for one video frame."""
    timestamp = frame_idx / fps
    boxes = result.boxes

    tracked_objects: list[TrackedObject] = []

    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            cls_id = int(box.cls[0].cpu())
            conf = float(box.conf[0].cpu())
            track_id = int(box.id[0].cpu()) if box.id is not None else -(i + 1)
            label = label_names[cls_id] if cls_id < len(label_names) else f"cls{cls_id}"
            object_id = f"{label}_{track_id}" if track_id >= 0 else f"{label}_{i}"

            if tracking_writer:
                area = (x2 - x1) * (y2 - y1)
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                if track_id not in init_areas:
                    init_areas[track_id] = area
                    first_seen[track_id] = frame_idx

                prev = prev_centers.get(track_id)
                displacement: float | None = None
                if prev is not None:
                    displacement = math.hypot(center[0] - prev[0], center[1] - prev[1])
                prev_centers[track_id] = center

                init_area = init_areas[track_id]
                area_ratio = area / init_area if init_area > 0 else 1.0
                frames_since_redetect = frame_idx - first_seen[track_id]

                tracked_objects.append(
                    TrackedObject(
                        object_id=object_id,
                        bbox_xyxy=[x1, y1, x2, y2],
                        bbox_area_px=area,
                        bbox_area_ratio_to_init=area_ratio,
                        center_xy=list(center),
                        displacement_px=displacement,
                        tracker_confidence=conf,
                        tracker_status=TrackerStatus.OK,
                        frames_since_redetect=frames_since_redetect,
                    )
                )

    if tracking_writer:
        bbox_size_change = any(
            o.bbox_area_ratio_to_init < 0.6 or o.bbox_area_ratio_to_init > 1.67
            for o in tracked_objects
        )
        drift = any(
            o.displacement_px is not None and o.displacement_px > 50
            for o in tracked_objects
        )
        tracking_writer.write(
            TrackingFrame(
                sequence_id=sequence_id,
                frame_id=frame_idx,
                timestamp=timestamp,
                tracked_objects=tracked_objects,
                flags=TrackingFlags(
                    bbox_size_change_flag=bbox_size_change,
                    drift_flag=drift,
                    any_recovery_trigger=bbox_size_change or drift,
                ),
            )
        )


def _draw_tracks(frame_bgr: np.ndarray, result, label_names: List[str]) -> np.ndarray:
    """Draw bounding boxes and track IDs onto the frame (BGR)."""
    out = frame_bgr.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return out

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        cls_id = int(box.cls[0].cpu())
        conf = float(box.conf[0].cpu())
        track_id = int(box.id[0].cpu()) if box.id is not None else -1

        label = label_names[cls_id] if cls_id < len(label_names) else f"cls{cls_id}"
        text = f"{label} {conf:.2f}" if track_id >= 0 else f"{label} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            text,
            (x1, max(y1 - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return out
