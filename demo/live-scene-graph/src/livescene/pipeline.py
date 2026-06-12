"""Per-frame pipeline: YOLOE+BoTSORT -> Depth Anything V2 -> SceneGraphBuilder.

Depth runs every ``depth_every_k`` frames (the bottleneck; it changes slowly),
tracking runs every frame. The builder is rebuilt on prompt change or frame
size change, since track ids restart and intrinsics depend on resolution.
"""

from __future__ import annotations

import numpy as np

import cv2

from .config import AppConfig
from .depth import DepthEstimator
from .detector import YoloeTracker
from .device import pick_device
from .intrinsics import intrinsics_from_fov
from .scene_graph.builder import SceneGraphBuilder
from .scene_graph.models import SceneGraphFrame


class LivePipeline:
    def __init__(self, cfg: AppConfig, device: str | None = None):
        self.cfg = cfg
        self.device = device or cfg.device or pick_device()
        self.detector = YoloeTracker(cfg, self.device)
        self.depth = DepthEstimator(cfg, self.device)
        self.detector.set_prompts(cfg.prompts)
        self._builder: SceneGraphBuilder | None = None
        self._frame_size: tuple[int, int] | None = None
        self._last_depth: np.ndarray | None = None
        self._frame_idx = 0

    def _new_builder(self, width: int, height: int) -> SceneGraphBuilder:
        return SceneGraphBuilder(
            sequence_id="live",
            config=self.cfg.scene_graph,
            intrinsics=intrinsics_from_fov(width, height, self.cfg.hfov_deg),
            # DA-V2 metric is already metres; never let the mm/m heuristic run.
            depth_scale_to_m=self.cfg.depth_scale_to_m,
        )

    def set_prompts(self, names: list[str]) -> None:
        """Re-prompt the detector and start a fresh graph (track ids restart)."""
        self.detector.set_prompts(names)
        if self._frame_size is not None:
            self._builder = self._new_builder(*self._frame_size)

    @property
    def prompts(self) -> list[str]:
        return self.detector.labels

    def process(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> SceneGraphFrame:
        h, w = frame_bgr.shape[:2]
        if self._frame_size != (w, h):
            self._frame_size = (w, h)
            self._builder = self._new_builder(w, h)
            self._last_depth = None

        ts = timestamp if timestamp is not None else self._frame_idx / 30.0
        objs = self.detector.track(frame_bgr, self._frame_idx)

        if self._last_depth is None or self._frame_idx % max(1, self.cfg.depth_every_k) == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._last_depth = self.depth.estimate(rgb)

        tracking_frame = self.detector.tracking_frame(objs, self._frame_idx, ts)
        # Webcam demo: no robot. gripper_closed=False / eef_pos=None keeps the
        # builder's gripper code paths dormant.
        sg = self._builder.build(
            tracking_frame, self._last_depth, gripper_closed=False, eef_pos=None
        )
        self._frame_idx += 1
        return sg
