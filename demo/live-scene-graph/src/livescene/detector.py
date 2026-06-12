"""YOLOE open-vocabulary detection + BoTSORT tracking -> TrackedObject list.

YOLOE text-prompt mode re-runs full open-vocab detection every frame and
BoTSORT assigns track ids on top, so there is no separate re-detection state
machine. The per-track derived fields (init area, area ratio, displacement)
mirror the REFLECT tracker's JSONL rows; the scene-graph builder consumes
them to derive node status.
"""

from __future__ import annotations

import math

from .config import AppConfig
from .scene_graph.models import (
    TrackedObject,
    TrackerStatus,
    TrackingFlags,
    TrackingFrame,
)


class YoloeTracker:
    def __init__(self, cfg: AppConfig, device: str):
        from ultralytics import YOLOE

        self.cfg = cfg
        self.device = device
        self.model = YOLOE(cfg.yoloe_model)
        self._labels: list[str] = []
        # Per-track state (survives across frames, cleared on prompt change).
        self._init_area: dict[int, float] = {}
        self._prev_center: dict[int, tuple[float, float]] = {}
        self._first_seen: dict[int, int] = {}
        # Force persist=False on the next track() call so BoTSORT ids reset.
        self._reset_next = True

    def set_prompts(self, names: list[str]) -> None:
        names = [n for n in (s.strip() for s in names) if n]
        if not names:
            return
        self.model.set_classes(names, self.model.get_text_pe(names))
        self._labels = names
        self._reset_next = True
        self._init_area.clear()
        self._prev_center.clear()
        self._first_seen.clear()

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def track(self, frame_bgr, frame_idx: int) -> list[TrackedObject]:
        """Run YOLOE+BoTSORT on one BGR frame (ultralytics' numpy convention)."""
        if not self._labels:
            return []
        res = self.model.track(
            frame_bgr,
            persist=not self._reset_next,
            tracker="botsort.yaml",
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            agnostic_nms=self.cfg.agnostic_nms,
            device=self.device,
            verbose=False,
        )[0]
        self._reset_next = False

        objs: list[TrackedObject] = []
        boxes = res.boxes
        if boxes is None:
            return objs
        for box in boxes:
            if box.id is None:  # untracked detection; builder is per-track-id
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            tid = int(box.id[0].cpu())
            cls = int(box.cls[0].cpu())
            conf = float(box.conf[0].cpu())
            # BoTSORT's second association stage can keep a track alive on
            # detections far below our conf threshold; don't emit those as
            # graph nodes.
            if conf < self.cfg.conf:
                continue
            label = self._labels[cls] if cls < len(self._labels) else f"cls{cls}"
            area = (x2 - x1) * (y2 - y1)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            self._init_area.setdefault(tid, area)
            self._first_seen.setdefault(tid, frame_idx)
            prev = self._prev_center.get(tid)
            disp = None if prev is None else math.hypot(cx - prev[0], cy - prev[1])
            self._prev_center[tid] = (cx, cy)
            init_area = self._init_area[tid]
            objs.append(
                TrackedObject(
                    object_id=f"{label}_{tid}",
                    bbox_xyxy=[x1, y1, x2, y2],
                    bbox_area_px=area,
                    bbox_area_ratio_to_init=area / init_area if init_area > 0 else 1.0,
                    center_xy=[cx, cy],
                    displacement_px=disp,
                    tracker_confidence=conf,
                    tracker_status=TrackerStatus.OK,
                    frames_since_redetect=frame_idx - self._first_seen[tid],
                )
            )
        return objs

    def tracking_frame(
        self, objs: list[TrackedObject], frame_idx: int, timestamp: float
    ) -> TrackingFrame:
        return TrackingFrame(
            sequence_id="live",
            frame_id=frame_idx,
            timestamp=timestamp,
            tracked_objects=objs,
            flags=TrackingFlags(False, False, False),
        )
