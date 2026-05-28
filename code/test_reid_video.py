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

import os
import sys

import cv2
import numpy as np

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIS_DIR = os.path.join(ROOT, "visuals", "reid_test")

# colours (BGR)
_GREEN  = (50,  205, 50)
_ORANGE = (0,   165, 255)
_BLUE   = (255, 160,  0)
_RED    = (0,   0,   220)
_GRAY   = (180, 180, 180)


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


def _frame_to_rgbd(bgr: np.ndarray, step_idx: int):
    from interfaces import RgbdFrame
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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


def main() -> None:
    video_path, label, out_path, thresh = _parse_args()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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
        redetect_interval=300,
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
    frame0     = _frame_to_rgbd(frame0_bgr, 0)
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

    f0 = frame0_bgr.copy()
    x1, y1, x2, y2 = seed_bbox.astype(int)
    cv2.rectangle(f0, (x1, y1), (x2, y2), _GREEN, thickness + 1)
    _put_label(f0, f"#{tracker._track_id} {label} [seed]", x1, y1, _GREEN, font_scale, thickness)
    cv2.putText(f0, "frame 0000", (8, h - 10), font, font_scale * 0.7, _GRAY, thickness)
    writer.write(f0)

    # Seed area and aspect ratio — used to filter bad re-detections.
    max_area_ratio   = 3.0   # reject bboxes > 3× seed area (e.g. monitor FPs)
    max_aspect_ratio = 2.5   # reject bboxes whose w/h deviates > 2.5× from seed
    x1s, y1s, x2s, y2s = seed_bbox
    seed_area        = max((x2s - x1s) * (y2s - y1s), 1.0)
    seed_aspect      = (x2s - x1s) / max(y2s - y1s, 1.0)

    # ── Track ─────────────────────────────────────────────────────────────────
    fi            = 1
    frozen        = False
    last_bbox     = seed_bbox
    redet_count   = 0
    same_id_count = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        frame     = _frame_to_rgbd(bgr, fi)
        annotated = bgr.copy()

        if frozen:
            # ── Re-acquire with GDINO + re-ID ─────────────────────────────────
            new_det, new_embeds = detector.detect_with_embeddings(frame, label)
            # Filter out detections much larger than the seed (background FPs like monitors)
            filtered = [
                d for d in new_det.detections
                if ((d.bbox_2d[2] - d.bbox_2d[0]) * (d.bbox_2d[3] - d.bbox_2d[1])) <= max_area_ratio * seed_area
                and (1 / max_aspect_ratio) <= ((d.bbox_2d[2] - d.bbox_2d[0]) / max(d.bbox_2d[3] - d.bbox_2d[1], 1.0)) / seed_aspect <= max_aspect_ratio
            ]
            new_det    = _DR(detections=filtered)
            new_embeds = new_embeds[:len(filtered)]
            if new_det.detections:
                redet_count += 1
                if has_reid and new_embeds:
                    new_norm = new_embeds[0] / max(np.linalg.norm(new_embeds[0]), 1e-8)
                    sim      = float(np.dot(reid._refs[0], new_norm))
                    is_same  = sim >= thresh
                else:
                    sim     = float("nan")
                    is_same = False

                if is_same:
                    same_id_count += 1
                    tracker.reinitialize(frame, new_det)
                    reid.update(0, new_embeds[0])
                    colour  = _BLUE
                    verdict = f"SAME  id=#{tracker._track_id}  sim={sim:.3f}"
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

                validator.reset(tracker._track_id - 1)
                frozen    = False
                last_bbox = new_det.detections[0].bbox_2d
                x1, y1, x2, y2 = last_bbox.astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness + 2)
                _put_label(annotated, f"#{tracker._track_id} {label} [redet]",
                           x1, y1, colour, font_scale, thickness)
                print(f"[frame {fi:04d}] re-detected — {verdict}")
            else:
                if last_bbox is not None:
                    x1, y1, x2, y2 = last_bbox.astype(int)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), _ORANGE, thickness)
                cv2.putText(annotated, "[searching…]", (8, 40),
                            font, font_scale, _ORANGE, thickness)
        else:
            # ── Normal CSRT update ────────────────────────────────────────────
            result = tracker.track(frame)
            if result.tracked_objects:
                val = validator.validate(frame, result)
                if val.is_valid:
                    obj       = result.tracked_objects[0]
                    last_bbox = obj.bbox_2d
                    x1, y1, x2, y2 = last_bbox.astype(int)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), _GREEN, thickness + 1)
                    _put_label(annotated, f"#{obj.track_id} {label}",
                               x1, y1, _GREEN, font_scale, thickness)
                else:
                    frozen = True
                    cv2.putText(annotated, f"[validator: {val.reason}]", (8, 40),
                                font, font_scale, _ORANGE, thickness)
                    print(f"[frame {fi:04d}] validator triggered ({val.reason}) — searching with GDINO …")
            else:
                frozen = True
                cv2.putText(annotated, "[CSRT lost]", (8, 40),
                            font, font_scale, _ORANGE, thickness)
                print(f"[frame {fi:04d}] CSRT lost — searching with GDINO …")

        cv2.putText(annotated, f"frame {fi:04d}", (8, h - 10),
                    font, font_scale * 0.7, _GRAY, thickness)
        writer.write(annotated)
        fi += 1

    cap.release()
    writer.release()

    print(f"\n── Summary ──────────────────────────────────────────────────────")
    print(f"  frames tracked : {fi}")
    print(f"  re-detections  : {redet_count}")
    print(f"  same-ID kept   : {same_id_count} / {redet_count}")
    print(f"  final track_id : #{tracker._track_id}")
    print(f"  output         : {out_path}")


if __name__ == "__main__":
    main()
