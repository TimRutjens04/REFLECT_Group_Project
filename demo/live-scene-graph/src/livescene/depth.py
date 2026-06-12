"""Depth Anything V2 metric-indoor depth: RGB frame -> HxW float32 metres.

The Metric-Indoor checkpoints output metres directly; the builder must be
constructed with depth_scale_to_m=1.0 so its mm/m heuristic never rescales.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .config import AppConfig


class DepthEstimator:
    def __init__(self, cfg: AppConfig, device: str):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.proc = AutoImageProcessor.from_pretrained(cfg.depth_model)
        self.model = AutoModelForDepthEstimation.from_pretrained(cfg.depth_model).to(device).eval()
        self.device = device
        self.input_long_side = cfg.depth_input_long_side
        # half() only on CUDA; on MPS/CPU it is often slower or unsupported.
        self._half = device == "cuda"
        if self._half:
            self.model.half()

    @torch.inference_mode()
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """rgb HxWx3 uint8 -> HxW float32 metres (resized to the frame size)."""
        H, W = rgb.shape[:2]
        img = Image.fromarray(rgb)
        if self.input_long_side and max(W, H) > self.input_long_side:
            scale = self.input_long_side / max(W, H)
            img = img.resize((max(1, round(W * scale)), max(1, round(H * scale))))
        inp = self.proc(images=img, return_tensors="pt").to(self.device)
        if self._half:
            inp = {k: v.half() if v.is_floating_point() else v for k, v in inp.items()}
        pred = self.model(**inp).predicted_depth  # (1, h, w)
        d = torch.nn.functional.interpolate(
            pred[:, None], size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
        return d.float().cpu().numpy()
