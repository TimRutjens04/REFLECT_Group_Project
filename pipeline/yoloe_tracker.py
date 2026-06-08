"""YOLOE tracking with Grounding DINO re-detection fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from interfaces.ITrackingValidator import TrackingValidator
from interfaces.IDetection import DetectionResult
from models.base import JsonlWriter
from models.detection import TriggerReason
from models.tracking import TrackedObject, TrackingFlags, TrackingFrame, TrackerStatus


# --------------------------------------------------------------------------- #
# Lightweight adapters so YOLOE results can be fed to a TrackingValidator that
# expects objects with .bbox_2d / .track_id / .label and a frame with .depth.
# --------------------------------------------------------------------------- #
@dataclass
class _ValTrackedObject:
    bbox_2d: np.ndarray
    track_id: int
    label: str


@dataclass
class _ValTrackingResult:
    tracked_objects: list


@dataclass
class _ValFrame:
    depth: np.ndarray | None = None


def _yoloe_result_to_tracking(result, label_names: list[str]) -> _ValTrackingResult:
    """Convert a YOLOE/BoTSORT result into a validator-readable TrackingResult.

    Boxes without an assigned track id are skipped: the validator is stateful
    per track_id, so only stably-tracked objects can be meaningfully validated.
    """
    objs: list[_ValTrackedObject] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return _ValTrackingResult(tracked_objects=objs)

    for box in boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        cls_id = int(box.cls[0].cpu())
        track_id = int(box.id[0].cpu())
        label = label_names[cls_id] if cls_id < len(label_names) else f"cls{cls_id}"
        objs.append(
            _ValTrackedObject(
                bbox_2d=np.array([x1, y1, x2, y2], dtype=np.float32),
                track_id=track_id,
                label=label,
            )
        )

    return _ValTrackingResult(tracked_objects=objs)


def _frame_depth_from_provider(provider, frame_idx: int) -> np.ndarray | None:
    """Best-effort depth fetch for the validator; returns None on any failure."""
    if provider is None:
        return None
    try:
        pf = provider.get_frame(frame_idx)
    except Exception:
        return None
    return getattr(pf, "depth", None)


def _detection_to_visual_prompts(
    detection_result: DetectionResult,
) -> tuple[dict, list[str]]:
    dets = detection_result.detections

    bboxes = np.array([d.bbox_2d for d in dets], dtype=np.float32)
    cls = np.arange(len(dets), dtype=np.int64)
    label_names = [d.label for d in dets]

    return {"bboxes": bboxes, "cls": cls}, label_names


def _has_yoloe_tracks(result, min_conf: float = 0.1) -> bool:
    if result.boxes is None or len(result.boxes) == 0:
        return False

    confs = result.boxes.conf.cpu().numpy()
    return bool(np.any(confs >= min_conf))


def _prime_yoloe(
    model_name: str,
    frame_rgb: np.ndarray,
    detection_result: DetectionResult,
    conf: float,
):
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    if not detection_result.success or not detection_result.detections:
        raise ValueError("Cannot prime YOLOE without detections.")

    visual_prompts, label_names = _detection_to_visual_prompts(detection_result)

    model = YOLOE(model_name)

    model.predict(
        source=frame_rgb,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        refer_image=frame_rgb,
        conf=conf,
        verbose=False,
    )

    return model, label_names


def track_video_with_yoloe_redetect(
    video_path: Path,
    initial_detection_result: DetectionResult,
    output_path: Path,
    provider=None,
    task=None,
    detection_runner=None,
    model_name: str = "yoloe-11l-seg.pt",
    frame_step: int = 1,
    yoloe_conf: float = 0.1,
    lost_after_n_frames: int = 3,
    occlusion_wait_frames: int = 10,
    redetect_every_n_frames: int | None = None,
    redetect_on_lost: bool = True,
    redetect_on_invalid: bool = True,
    validator: TrackingValidator | None = None,
    validate_with_depth: bool = False,
    validator_settle_frames: int = 5,
    sequence_id: str = "unknown",
    detection_writer: JsonlWriter | None = None,
    tracking_writer: JsonlWriter | None = None,
):
    """Track objects with optional GDino re-detection.

    Re-detection can be triggered in three independent ways:
      - ``redetect_on_lost``: trigger GDino when a tracked label goes missing
        for ``lost_after_n_frames`` consecutive frames (default: True).
      - ``redetect_on_invalid``: trigger GDino when the ``validator`` reports a
        track is misbehaving — bbox area change, centroid drift, depth jump, or
        timeout (default: True). Catches bboxes "doing weird things" while the
        label is still nominally present.
      - ``redetect_every_n_frames``: force a GDino run every N frames regardless
        of tracking state (default: None = disabled).

    Validator notes:
      - If ``redetect_on_invalid`` is True and ``validator`` is None, a default
        :class:`CompositeTrackingValidator` is constructed with its periodic
        ``timeout`` disabled (periodic redetect stays owned by
        ``redetect_every_n_frames``). Inject your own configured validator to
        override thresholds.
      - Depth-based checks only run when ``validate_with_depth`` is True, in
        which case depth is pulled from ``provider.get_frame()``. Otherwise the
        validator runs on bbox geometry only (area + drift).

    When all redetect modes are off, ``provider``, ``task``, and
    ``detection_runner`` are unused and may be omitted.
    """
    _redetect_enabled = (
        redetect_on_lost or redetect_on_invalid or (redetect_every_n_frames is not None)
    )
    if _redetect_enabled and (
        provider is None or task is None or detection_runner is None
    ):
        raise ValueError(
            "provider, task, and detection_runner are required when redetection is enabled."
        )

    if not initial_detection_result.success or not initial_detection_result.detections:
        raise ValueError("Initial Grounding DINO detection produced no detections.")

    # Construct a default validator if requested and none was injected.
    if redetect_on_invalid and validator is None:
        try:
            # Adjust this import path to wherever CompositeTrackingValidator lives.
            from validation.composite_tracking_validator import (
                CompositeTrackingValidator,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import CompositeTrackingValidator automatically. "
                "Either fix this import path or pass a configured `validator=...`."
            ) from exc
        # redetect_interval huge -> disable the validator's own periodic timeout;
        # periodic redetect is handled by redetect_every_n_frames instead.
        validator = CompositeTrackingValidator(redetect_interval=10**9)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    ret, ref_bgr = cap.read()
    if not ret:
        raise RuntimeError("Could not read reference frame 0.")

    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

    model, label_names = _prime_yoloe(
        model_name=model_name,
        frame_rgb=ref_rgb,
        detection_result=initial_detection_result,
        conf=yoloe_conf,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    expected_labels = set(label_names)
    lost_count = {lbl: 0 for lbl in expected_labels}
    redetect_cooldown = 0

    # Per-track state for JSONL derived fields
    _init_areas: dict[int, float] = {}
    _prev_centers: dict[int, tuple[float, float]] = {}
    _first_seen: dict[int, int] = {}

    # Validator bookkeeping
    _validated_ids: set[int] = set()
    frames_since_prime = 0

    state = "TRACKING"
    frame_idx = 0
    reset_tracker_next = True
    last_result = None

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                writer.write(frame_bgr)
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            drawn = frame_bgr.copy()

            if redetect_cooldown > 0:
                redetect_cooldown -= 1

            # --- Determine if a redetect should fire this frame ---
            trigger_redetect = False
            redetect_reason = TriggerReason.TRACKER_LOW_CONFIDENCE

            if (
                redetect_every_n_frames
                and frame_idx > 0
                and frame_idx % redetect_every_n_frames == 0
            ):
                trigger_redetect = True
                redetect_reason = TriggerReason.FRAME_COUNTER_K

            if state == "TRACKING":
                results = model.track(
                    frame_rgb,
                    persist=not reset_tracker_next,
                    conf=yoloe_conf,
                    tracker="botsort.yaml",
                    verbose=False,
                )
                reset_tracker_next = False
                result = results[0]
                last_result = result
                frames_since_prime += 1

                missing: set[str] = set()
                if redetect_on_lost:
                    present = _present_labels(result, label_names, yoloe_conf)
                    for lbl in expected_labels:
                        lost_count[lbl] = 0 if lbl in present else lost_count[lbl] + 1
                    missing = {
                        lbl
                        for lbl in expected_labels
                        if lost_count[lbl] >= lost_after_n_frames
                    }

                # --- Composite validator: catch drifting / exploding / depth-jumping bboxes ---
                invalid_reason: str | None = None
                if validator is not None and redetect_on_invalid:
                    depth = (
                        _frame_depth_from_provider(provider, frame_idx)
                        if validate_with_depth
                        else None
                    )
                    val_frame = _ValFrame(depth=depth)
                    tracking_result = _yoloe_result_to_tracking(result, label_names)
                    for o in tracking_result.tracked_objects:
                        _validated_ids.add(o.track_id)
                    vres = validator.validate(val_frame, tracking_result)
                    if not vres.is_valid:
                        invalid_reason = vres.reason or "validation failed"

                drawn = _draw_tracks(frame_bgr, result, label_names)

                if tracking_writer:
                    _write_jsonl_rows(
                        result=result,
                        frame_idx=frame_idx,
                        fps=fps,
                        sequence_id=sequence_id,
                        label_names=label_names,
                        init_areas=_init_areas,
                        prev_centers=_prev_centers,
                        first_seen=_first_seen,
                        tracking_writer=tracking_writer,
                    )

                # Missing-label trigger
                if (
                    redetect_on_lost
                    and not trigger_redetect
                    and missing
                    and redetect_cooldown <= 0
                ):
                    trigger_redetect = True
                    redetect_reason = TriggerReason.TRACKER_LOW_CONFIDENCE

                # Validator trigger — suppressed during the settle window right
                # after a (re-)prime so it doesn't fire before tracks stabilise.
                if (
                    redetect_on_invalid
                    and not trigger_redetect
                    and invalid_reason is not None
                    and redetect_cooldown <= 0
                    and frames_since_prime >= validator_settle_frames
                ):
                    trigger_redetect = True
                    redetect_reason = TriggerReason.TRACKER_LOW_CONFIDENCE

                if trigger_redetect:
                    bits: list[str] = []
                    if missing:
                        bits.append(f"missing={sorted(missing)}")
                    if invalid_reason:
                        bits.append(f"validator={invalid_reason}")
                    if not bits:
                        bits.append(redetect_reason.value)
                    print(
                        f"[frame {frame_idx}] redetect ({'; '.join(bits)}). Running GDINO."
                    )
                    state = "REDETECT"

            if state == "REDETECT":
                frame_input = provider.get_frame(frame_idx)

                gdino_result = detection_runner.run(
                    frame_input,
                    task,
                    trigger_reason=redetect_reason,
                )

                missing_now = (
                    {
                        lbl
                        for lbl in expected_labels
                        if lost_count[lbl] >= lost_after_n_frames
                    }
                    if redetect_on_lost
                    else set()
                )
                found = (
                    {d.label for d in gdino_result.detections}
                    if gdino_result.detections
                    else set()
                )
                recovered = found & missing_now if missing_now else found

                if gdino_result.success and gdino_result.detections:
                    print(
                        f"[frame {frame_idx}] GDINO recovered {sorted(recovered or found)}. "
                        f"Re-priming YOLOE."
                    )
                    model, label_names = _prime_yoloe(
                        model_name=model_name,
                        frame_rgb=frame_rgb,
                        detection_result=gdino_result,
                        conf=yoloe_conf,
                    )
                    for lbl in set(label_names) & expected_labels:
                        lost_count[lbl] = 0
                    reset_tracker_next = True

                    # Tracker IDs restart after a re-prime: clear validator state
                    # for everything we've validated so far so it reseeds cleanly.
                    frames_since_prime = 0
                    if validator is not None:
                        for tid in _validated_ids:
                            validator.reset(tid)
                        _validated_ids.clear()

                    drawn = _draw_gdino_boxes(frame_bgr, gdino_result)
                else:
                    if missing_now:
                        print(
                            f"[frame {frame_idx}] GDINO did not recover "
                            f"{sorted(missing_now)}. Backing off {occlusion_wait_frames} frames."
                        )
                        drawn = _draw_status(
                            frame_bgr, f"waiting for {sorted(missing_now)}"
                        )
                    else:
                        print(
                            f"[frame {frame_idx}] GDINO returned no detections. "
                            f"Backing off {occlusion_wait_frames} frames."
                        )
                        drawn = _draw_status(frame_bgr, "GDINO: no detections")
                    redetect_cooldown = occlusion_wait_frames

                state = "TRACKING"

            writer.write(drawn)

            if frame_idx % 100 == 0:
                n = 0
                if last_result is not None and last_result.boxes is not None:
                    n = len(last_result.boxes)
                print(
                    f"[frame {frame_idx}/{n_frames}] state={state}, "
                    f"yoloe_boxes={n}, lost={ {k: v for k, v in lost_count.items() if v} }"
                )

            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    print(f"Saved tracked video → {output_path.resolve()}")


def _draw_tracks(
    frame_bgr: np.ndarray,
    result,
    label_names: list[str],
) -> np.ndarray:
    out = frame_bgr.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return out

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        cls_id = int(box.cls[0].cpu())
        conf = float(box.conf[0].cpu())
        track_id = int(box.id[0].cpu()) if box.id is not None else -1

        label = label_names[cls_id] if cls_id < len(label_names) else f"cls{cls_id}"

        text = f"{label} {conf:.2f}"

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


def _draw_gdino_boxes(
    frame_bgr: np.ndarray,
    detection_result: DetectionResult,
) -> np.ndarray:
    out = frame_bgr.copy()

    for det in detection_result.detections:
        x1, y1, x2, y2 = map(int, det.bbox_2d)
        label = det.label

        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            out,
            f"GDINO: {label}",
            (x1, max(y1 - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

    return out


def _draw_status(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    out = frame_bgr.copy()

    cv2.putText(
        out,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def _present_labels(result, label_names: list[str], min_conf: float) -> set[str]:
    if result.boxes is None or len(result.boxes) == 0:
        return set()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    return {
        label_names[c]
        for c, cf in zip(clss, confs)
        if cf >= min_conf and 0 <= c < len(label_names)
    }


def _write_jsonl_rows(
    result,
    frame_idx: int,
    fps: float,
    sequence_id: str,
    label_names: list[str],
    init_areas: dict,
    prev_centers: dict,
    first_seen: dict,
    tracking_writer: JsonlWriter | None,
) -> None:
    """Build and write a TrackingFrame for one YOLOe tracking frame."""
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
