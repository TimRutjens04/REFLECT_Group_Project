from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackMetrics:
    """Aggregated metrics for one tracker on one video."""
    tracker:           str
    mota:              float   # Multiple Object Tracking Accuracy (single-object variant)
    motp:              float   # Mean IoU over True Positive frames
    id_switches:       int
    track_loss_rate:   float   # fraction of frames where pred is None
    recovery_rate:     float   # fraction of occlusion-end events recovered within 3 frames
    mean_ms_per_frame: float   # wall-clock tracker time (GDINO excluded)
    n_frames:          int
    n_gt_frames:       int
    fps:               float = field(init=False)

    def __post_init__(self) -> None:
        self.fps = 1000.0 / self.mean_ms_per_frame if self.mean_ms_per_frame > 0 else 0.0


def compute_metrics(
    tracker_name: str,
    preds:        list[np.ndarray | None],
    gt:           list[np.ndarray | None],
    id_sw:        int,
    timings_ms:   list[float],
    iou_thresh:   float = 0.50,
) -> TrackMetrics:
    """
    MOTA (single-object) = 1 - (FP + FN + IDSW) / N_gt
    MOTP                 = mean IoU over TP frames
    """
    assert len(preds) == len(gt)
    n = len(preds)
    fp = fn = tp = 0
    iou_sum = 0.0
    lost_frames = 0

    for p, g in zip(preds, gt):
        if g is None and p is None:
            continue
        if g is None and p is not None:
            fp += 1
        elif g is not None and p is None:
            fn += 1
            lost_frames += 1
        else:
            iou = _iou(p, g)
            if iou >= iou_thresh:
                tp += 1
                iou_sum += iou
            else:
                fp += 1
                fn += 1

    n_gt = sum(g is not None for g in gt)
    mota     = 1.0 - (fp + fn + id_sw) / (n_gt + 1e-9)
    motp     = iou_sum / (tp + 1e-9)
    loss_rate = lost_frames / (n + 1e-9)
    recovery  = _recovery_rate(preds, gt, iou_thresh)
    mean_ms   = float(np.mean(timings_ms)) if timings_ms else 0.0

    return TrackMetrics(
        tracker=tracker_name,
        mota=round(mota, 4),
        motp=round(motp, 4),
        id_switches=id_sw,
        track_loss_rate=round(loss_rate, 4),
        recovery_rate=round(recovery, 4),
        mean_ms_per_frame=round(mean_ms, 2),
        n_frames=n,
        n_gt_frames=n_gt,
    )


def print_table(results: list[TrackMetrics]) -> None:
    header = (
        f"{'Tracker':<14} {'MOTA':>7} {'MOTP':>7} "
        f"{'ID-SW':>6} {'Loss%':>7} {'Recov':>7} {'ms/f':>7} {'FPS':>6}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(
            f"{r.tracker:<14} {r.mota:>7.3f} {r.motp:>7.3f} "
            f"{r.id_switches:>6d} {r.track_loss_rate*100:>6.1f}% "
            f"{r.recovery_rate:>7.3f} {r.mean_ms_per_frame:>7.1f} {r.fps:>6.1f}"
        )
    print(sep + "\n")


def _recovery_rate(
    preds: list[np.ndarray | None],
    gt:    list[np.ndarray | None],
    iou_thresh: float,
    window: int = 3,
) -> float:
    """Fraction of occlusion-end events recovered (IoU>=thresh) within `window` frames."""
    events = recovered = 0
    n = len(preds)
    for i in range(1, n):
        if gt[i] is not None and (i == 0 or gt[i - 1] is None):
            events += 1
            for j in range(i, min(i + window, n)):
                if preds[j] is not None and gt[j] is not None:
                    if _iou(preds[j], gt[j]) >= iou_thresh:
                        recovered += 1
                        break
    return recovered / (events + 1e-9)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (ua + 1e-6)
