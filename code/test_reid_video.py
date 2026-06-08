"""
Re-ID integration test on a raw video.

Seeds the CSRT tracker from a GDINO detection on frame 0, then tracks frame by
frame.  When CSRT fails (object left frame), GDINO re-detects and the ReID
matcher decides whether it is the same object (same ID) or a new one (ID bump).
The cosine similarity score is printed to the terminal on every re-detection.

Usage:
    poetry run python code/test_reid_video.py <video> --label "monster energy"
    poetry run python code/test_reid_video.py <video> --label "can" --thresh 0.80
    poetry run python code/test_reid_video.py <video> --label "can" --out visuals/reid_test.mp4

If the video is a HEVC .MOV (iPhone), convert first:
    ffmpeg -i recording.MOV -c:v libx264 recording.mp4

Controls (ROI fallback):
    If GDINO finds nothing on frame 0 you will be prompted to draw the bbox
    manually — the embedding will be skipped and re-ID will not fire (prints a
    warning).
"""

from __future__ import annotations

import csv
import os
import sys

import cv2
import numpy as np

from tracking_log import TrackingLog

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIS_DIR = os.path.join(ROOT, "visuals", "reid_test")

# colours (BGR)
_GREEN      = (50,  205, 50)
_ORANGE     = (0,   165, 255)
_BLUE       = (255, 160,  0)
_RED        = (0,   0,   220)
_GRAY       = (180, 180, 180)
_WHITE      = (255, 255, 255)
_GDINO_CAND = (120, 120, 120)   # gray — raw GDINO candidates before filtering


def _parse_args() -> tuple[str, str, str, float]:
    args   = sys.argv[1:]
    label  = "object"
    out    = None
    thresh = 0.85

    if "--label" in args:
        i     = args.index("--label")
        label = args[i + 1]
        args  = args[:i] + args[i + 2:]

    if "--out" in args:
        i   = args.index("--out")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]

    if "--thresh" in args:
        i      = args.index("--thresh")
        thresh = float(args[i + 1])
        args   = args[:i] + args[i + 2:]

    if not args:
        print("Usage: poetry run python code/test_reid_video.py <video> --label NAME [--thresh 0.85] [--out PATH]")
        sys.exit(1)

    video_path = args[0]
    if not os.path.isabs(video_path):
        video_path = os.path.join(ROOT, video_path)

    if out is None:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        slug = label.replace(" ", "_")
        out  = os.path.join(VIS_DIR, f"{stem}_{slug}_reid.mp4")

    return video_path, label, out, thresh


def _frame_to_rgbd(bgr: np.ndarray, step_idx: int, depth: np.ndarray | None = None):
    from interfaces import RgbdFrame
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if depth is None:
        depth = np.zeros(bgr.shape[:2], dtype=np.float32)
    return RgbdFrame(rgb=rgb, depth=depth, step_idx=step_idx)


def _put_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale * 0.7, thickness)
    ty = max(y - 4, th + 4)
    cv2.rectangle(img, (x, ty - th - bl - 2), (x + tw + 4, ty + 2), colour, -1)
    cv2.putText(img, text, (x + 2, ty - bl), font, font_scale * 0.7, (0, 0, 0), thickness)


def _draw_dashed_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    colour: tuple[int, int, int],
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    """Draw a dashed rectangle outline (used for the seed ghost)."""
    pts = [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]
    for (ax, ay), (bx, by) in pts:
        dx, dy = bx - ax, by - ay
        length = max(int((dx ** 2 + dy ** 2) ** 0.5), 1)
        steps  = max(length // (dash_len + gap_len), 1)
        for i in range(steps):
            t0 = i * (dash_len + gap_len) / length
            t1 = min(t0 + dash_len / length, 1.0)
            px0 = int(ax + t0 * dx)
            py0 = int(ay + t0 * dy)
            px1 = int(ax + t1 * dx)
            py1 = int(ay + t1 * dy)
            cv2.line(img, (px0, py0), (px1, py1), colour, 1)


def _draw_diag_panel(
    img: np.ndarray,
    state: str,
    area: float | None,
    area_ratio: float | None,
    drift_px: float | None,
    age_frames: int | None,
    confidence: float | None,
    font_scale: float,
    thickness: int,
) -> None:
    """Semi-transparent diagnostic panel in the bottom-left corner."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = font_scale * 0.65

    lines = [
        f"STATE : {state}",
        f"area  : {int(area) if area is not None else '?'} px"
        + (f"  ({area_ratio:.2f}x seed)" if area_ratio is not None else ""),
        f"drift : {drift_px:.1f} px" if drift_px  is not None else "drift : ?",
        f"age   : {age_frames} f"    if age_frames is not None else "age   : ?",
        f"conf  : {confidence:.2f}"  if confidence is not None else "conf  : ?",
    ]

    line_h  = cv2.getTextSize("A", font, fs, thickness)[0][1] + 6
    pad     = 6
    panel_w = max(cv2.getTextSize(ln, font, fs, thickness)[0][0] for ln in lines) + pad * 2
    panel_h = line_h * len(lines) + pad * 2

    px1, py1 = 8, h - panel_h - 30
    px2, py2 = px1 + panel_w, py1 + panel_h

    overlay = img.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, line in enumerate(lines):
        ty = py1 + pad + (i + 1) * line_h - 2
        if i == 0:
            colour = _GREEN if state == "TRACKING" else (_ORANGE if state in ("FROZEN", "SEARCHING") else _BLUE)
        elif i == 1 and area_ratio is not None and area_ratio < 0.3:
            colour = _RED       # area ratio alarm — likely a sliver
        else:
            colour = _WHITE
        cv2.putText(img, line, (px1 + pad, ty), font, fs, colour, thickness)


def main() -> None:
    video_path, label, out_path, thresh = _parse_args()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── CSV log path ──────────────────────────────────────────────────────────
    stem     = os.path.splitext(os.path.basename(out_path))[0]
    csv_path = os.path.join(os.path.dirname(out_path), f"{stem}_log.csv")

    # ── Depth maps (optional) ─────────────────────────────────────────────────
    video_stem  = os.path.splitext(os.path.basename(video_path))[0]
    depth_path  = os.path.join(ROOT, "depth", f"{video_stem}.npz")
    depth_maps: np.ndarray | None = None
    if os.path.exists(depth_path):
        depth_maps = np.load(depth_path)["depth_maps"]   # (N, H, W) float32, metres
        print(f"Depth maps : {depth_path}  ({len(depth_maps)} frames, metric)")

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"Loading Grounding DINO …")
    from detector import GroundingDinoDetector
    from interfaces import DetectionResult as _DR
    from reid import ObjectReIdMatcher
    from tracker import CsrtObjectTracker
    from validator import CompositeTrackingValidator

    detector = GroundingDinoDetector(score_thresh=0.15)
    detector.load()
    reid      = ObjectReIdMatcher(similarity_threshold=thresh)
    tracker   = CsrtObjectTracker()
    # Defaults are tuned for 1fps RoboFail data — loosen for 30fps video.
    validator = CompositeTrackingValidator(
        area_change_thresh=0.90,
        drift_thresh_px=150.0,
        depth_jump_thresh=0.45,
        redetect_interval=300,
        area_check_grace_frames=5,
    )

    print(f"Re-ID threshold : {thresh}")
    print(f"Label           : '{label}'")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        print("If HEVC .MOV: ffmpeg -i in.MOV -c:v libx264 out.mp4")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame0_bgr = cap.read()
    if not ok:
        print("Error: cannot read frame 0")
        sys.exit(1)

    h, w       = frame0_bgr.shape[:2]
    font_scale = max(0.5, w / 1200)
    thickness  = max(1, w // 400)
    font       = cv2.FONT_HERSHEY_SIMPLEX

    # ── Seed from GDINO on frame 0 ────────────────────────────────────────────
    frame0     = _frame_to_rgbd(frame0_bgr, 0, depth_maps[0] if depth_maps is not None else None)
    det_result, embeds = detector.detect_with_embeddings(frame0, label)

    has_reid = False
    if det_result.detections:
        tracker.initialize(frame0, det_result)
        if embeds:
            reid.register(0, embeds[0])
            has_reid = True
            print(f"[frame 0000] GDINO seed — embedding registered (dim={embeds[0].shape[0]})")
        else:
            print(f"[frame 0000] GDINO seed — no embedding captured (re-ID disabled)")
        seed_bbox  = det_result.detections[0].bbox_2d
        seed_score = det_result.detections[0].score
        print(f"             bbox={seed_bbox.astype(int).tolist()}  score={seed_score:.3f}")
    else:
        print(f"[frame 0000] GDINO found nothing — falling back to manual selection")
        print(f"             (re-ID will not fire without a reference embedding)\n")
        roi = cv2.selectROI(
            f"Select '{label}' — SPACE to confirm, ESC to cancel",
            frame0_bgr, showCrosshair=True, fromCenter=False,
        )
        cv2.destroyAllWindows()
        x, y, bw, bh = roi
        if bw == 0 or bh == 0:
            print("No selection — exiting.")
            sys.exit(0)
        from interfaces import DetectedObject, DetectionResult
        det_result = DetectionResult(detections=[
            DetectedObject(label=label, score=1.0,
                           bbox_2d=np.array([x, y, x + bw, y + bh], dtype=np.float32))
        ])
        tracker.initialize(frame0, det_result)
        seed_bbox = det_result.detections[0].bbox_2d

    # ── Output writer ─────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, src_fps, (w, h))

    # Seed geometry — used for the ghost outline and area ratio denominator
    x1s, y1s, x2s, y2s = seed_bbox
    seed_area   = max((x2s - x1s) * (y2s - y1s), 1.0)
    seed_aspect = (x2s - x1s) / max(y2s - y1s, 1.0)

    # Seed area filter thresholds for GDINO re-detections
    max_area_ratio   = 3.0
    max_aspect_ratio = 2.5

    f0 = frame0_bgr.copy()
    x1, y1, x2, y2 = seed_bbox.astype(int)
    cv2.rectangle(f0, (x1, y1), (x2, y2), _GREEN, thickness + 1)
    _put_label(f0, f"#{tracker._track_id} {label} [seed]", x1, y1, _GREEN, font_scale, thickness)
    _draw_dashed_rect(f0, *seed_bbox.astype(int), _WHITE)
    _draw_diag_panel(f0, "SEEDED", seed_area, 1.0, 0.0, 0, 1.0, font_scale, thickness)
    cv2.putText(f0, "frame 0000", (8, h - 10), font, font_scale * 0.7, _GRAY, thickness)
    writer.write(f0)

    # ── CSV + JSON log setup ──────────────────────────────────────────────────
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "state",
        "x1", "y1", "x2", "y2",
        "area", "area_ratio", "drift_px", "age_frames", "confidence",
        "gdino_n_candidates", "gdino_n_filtered",
    ])
    csv_writer.writerow([
        0, "SEEDED", *seed_bbox.astype(int).tolist(),
        int(seed_area), 1.0, 0.0, 0, 1.0, 0, 0,
    ])

    log      = TrackingLog(sequence_id=video_stem)
    json_path = os.path.join(os.path.dirname(out_path), f"{stem}_log.json")
    log.record(
        frame_id=0, timestamp=0.0,
        object_id=f"{label}_0", label=label,
        bbox_xyxy=seed_bbox.astype(int).tolist(),
        tracker_confidence=1.0, tracker_status="seeded",
        bbox_size_change_flag=False, drift_flag=False,
        recovery_trigger=False, held_by_gripper=False,
        last_detection_frame=0,
    )
    last_detection_frame_idx = 0

    # ── Track ─────────────────────────────────────────────────────────────────
    fi            = 1
    frozen        = False
    last_bbox     = seed_bbox
    redet_count   = 0
    same_id_count = 0

    # Running metrics shown in the panel when the validator hasn't fired yet
    last_area_ratio: float | None = None
    last_drift_px:   float | None = None
    last_age:        int | None   = None
    last_conf:       float | None = 1.0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        d         = depth_maps[fi] if (depth_maps is not None and fi < len(depth_maps)) else None
        frame     = _frame_to_rgbd(bgr, fi, d)
        annotated = bgr.copy()

        # Always draw the seed ghost box
        _draw_dashed_rect(annotated, *seed_bbox.astype(int), _WHITE)

        state_label      = "FROZEN"
        gdino_n_raw      = 0
        gdino_n_filtered = 0
        row_bbox         = last_bbox
        row_conf         = last_conf or 0.0

        if frozen:
            # ── Re-acquire with GDINO + re-ID ─────────────────────────────────
            new_det_raw, new_embeds_raw = detector.detect_with_embeddings(frame, label)
            gdino_n_raw = len(new_det_raw.detections)

            # Draw all raw GDINO candidates as thin gray boxes
            for d in new_det_raw.detections:
                bx1, by1, bx2, by2 = d.bbox_2d.astype(int)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), _GDINO_CAND, 1)

            filtered = [
                d for d in new_det_raw.detections
                if ((d.bbox_2d[2] - d.bbox_2d[0]) * (d.bbox_2d[3] - d.bbox_2d[1])) <= max_area_ratio * seed_area
                and (1 / max_aspect_ratio) <= (
                    (d.bbox_2d[2] - d.bbox_2d[0]) / max(d.bbox_2d[3] - d.bbox_2d[1], 1.0)
                ) / seed_aspect <= max_aspect_ratio
            ]
            gdino_n_filtered = len(filtered)
            new_det          = _DR(detections=filtered)
            new_embeds       = new_embeds_raw[:len(filtered)]

            if new_det.detections:
                redet_count += 1
                if has_reid and new_embeds:
                    new_norm = new_embeds[0] / max(np.linalg.norm(new_embeds[0]), 1e-8)
                    sim      = float(np.dot(reid._refs[0], new_norm))
                    is_same  = sim >= thresh
                else:
                    sim     = float("nan")
                    is_same = False

                old_track_id = tracker._track_id
                if is_same:
                    same_id_count += 1
                    tracker.reinitialize(frame, new_det)
                    reid.update(0, new_embeds[0])
                    colour      = _BLUE
                    verdict     = f"SAME  id=#{tracker._track_id}  sim={sim:.3f}"
                    state_label = "RECOVERED"
                else:
                    tracker.initialize(frame, new_det)
                    if has_reid and new_embeds:
                        reid.register(0, new_embeds[0])
                    colour  = _RED
                    verdict = (
                        f"NEW   id=#{tracker._track_id}  sim={sim:.3f}"
                        if not np.isnan(sim) else
                        f"NEW   id=#{tracker._track_id}  (no ref)"
                    )
                    state_label = "REDETECTED"

                validator.reset(old_track_id)
                frozen        = False
                last_bbox     = new_det.detections[0].bbox_2d
                row_bbox      = last_bbox
                x1, y1, x2, y2 = last_bbox.astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness + 2)
                _put_label(annotated, f"#{tracker._track_id} {label} [redet]",
                           x1, y1, colour, font_scale, thickness)
                cur_area        = (x2 - x1) * (y2 - y1)
                last_area_ratio = cur_area / seed_area
                last_drift_px   = 0.0
                last_age        = 0
                last_conf       = 1.0
                row_conf        = 1.0
                print(f"[frame {fi:04d}] re-detected — {verdict}")
            else:
                if last_bbox is not None:
                    x1, y1, x2, y2 = last_bbox.astype(int)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), _ORANGE, thickness)
                cv2.putText(annotated, "[searching…]", (8, 40),
                            font, font_scale, _ORANGE, thickness)
                state_label = "SEARCHING"
        else:
            # ── Normal CSRT update ────────────────────────────────────────────
            result = tracker.track(frame)
            if result.tracked_objects:
                val = validator.validate(frame, result)
                last_area_ratio = val.area_ratio
                last_drift_px   = val.drift_px
                last_age        = val.frames_since_init
                last_conf       = val.confidence

                if val.is_valid:
                    obj       = result.tracked_objects[0]
                    last_bbox = obj.bbox_2d
                    row_bbox  = last_bbox
                    row_conf  = val.confidence
                    x1, y1, x2, y2 = last_bbox.astype(int)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), _GREEN, thickness + 1)
                    _put_label(annotated, f"#{obj.track_id} {label}",
                               x1, y1, _GREEN, font_scale, thickness)
                    state_label = "TRACKING"
                else:
                    frozen = True
                    cv2.putText(annotated, f"[validator: {val.reason}]", (8, 40),
                                font, font_scale, _ORANGE, thickness)
                    print(f"[frame {fi:04d}] validator triggered ({val.reason}) — searching with GDINO …")
                    state_label = "FROZEN"
            else:
                frozen = True
                cv2.putText(annotated, "[CSRT lost]", (8, 40),
                            font, font_scale, _ORANGE, thickness)
                print(f"[frame {fi:04d}] CSRT lost — searching with GDINO …")
                state_label = "FROZEN"

        # ── Diagnostic panel ──────────────────────────────────────────────────
        cur_area = None
        if row_bbox is not None:
            bx1, by1, bx2, by2 = row_bbox
            cur_area = max((bx2 - bx1) * (by2 - by1), 0.0)

        _draw_diag_panel(
            annotated, state_label,
            cur_area, last_area_ratio, last_drift_px, last_age, last_conf,
            font_scale, thickness,
        )

        cv2.putText(annotated, f"frame {fi:04d}", (8, h - 10),
                    font, font_scale * 0.7, _GRAY, thickness)
        writer.write(annotated)

        # ── CSV row ───────────────────────────────────────────────────────────
        if row_bbox is not None:
            bx1, by1, bx2, by2 = row_bbox.astype(int).tolist()
        else:
            bx1 = by1 = bx2 = by2 = ""
        csv_writer.writerow([
            fi, state_label,
            bx1, by1, bx2, by2,
            int(cur_area) if cur_area is not None else "",
            f"{last_area_ratio:.4f}" if last_area_ratio is not None else "",
            f"{last_drift_px:.2f}"   if last_drift_px   is not None else "",
            last_age                 if last_age         is not None else "",
            f"{last_conf:.4f}"       if last_conf        is not None else "",
            gdino_n_raw,
            gdino_n_filtered,
        ])

        # ── JSON log record ───────────────────────────────────────────────────
        if state_label in ("RECOVERED", "REDETECTED"):
            last_detection_frame_idx = fi
        log.record(
            frame_id=fi,
            timestamp=fi / src_fps,
            object_id=f"{label}_0",
            label=label,
            bbox_xyxy=[bx1, by1, bx2, by2] if row_bbox is not None else [],
            tracker_confidence=float(last_conf) if last_conf is not None else 0.0,
            tracker_status=state_label.lower(),
            bbox_size_change_flag=(last_area_ratio is not None and last_area_ratio < (1 - 0.90)),
            drift_flag=(last_drift_px is not None and last_drift_px > 150.0),
            recovery_trigger=state_label in ("RECOVERED", "REDETECTED"),
            held_by_gripper=False,
            last_detection_frame=last_detection_frame_idx,
        )

        fi += 1

    cap.release()
    writer.release()
    csv_file.close()
    log.save_json(json_path)

    print(f"\n── Summary ──────────────────────────────────────────────────────")
    print(f"  frames tracked : {fi}")
    print(f"  re-detections  : {redet_count}")
    print(f"  same-ID kept   : {same_id_count} / {redet_count}")
    print(f"  final track_id : #{tracker._track_id}")
    print(f"  output video   : {out_path}")
    print(f"  log CSV        : {csv_path}")
    print(f"  log JSON       : {json_path}")


if __name__ == "__main__":
    main()
