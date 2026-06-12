"""Smoke test: Depth Anything V2 metric-indoor loads and outputs a frame-sized
float depth map in plausible metres. Skips if the checkpoint can't download.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("transformers")

from fixtures import make_rgb_frame
from livescene.config import AppConfig
from livescene.device import pick_device


@pytest.fixture(scope="module")
def estimator():
    from livescene.depth import DepthEstimator

    try:
        return DepthEstimator(AppConfig(), pick_device())
    except Exception as e:
        pytest.skip(f"Depth model unavailable in this environment: {e}")


def test_depth_output_contract(estimator):
    rgb = make_rgb_frame(640, 480)
    depth = estimator.estimate(rgb)
    assert depth.shape == (480, 640)
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert (depth > 0).all()


def test_depth_plausible_metres_on_real_photo(estimator):
    from ultralytics.utils import ASSETS

    bgr = cv2.imread(str(ASSETS / "bus.jpg"))
    if bgr is None:
        pytest.skip("ultralytics asset image not found")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    depth = estimator.estimate(rgb)
    dt = time.perf_counter() - t0
    print(
        f"\ndepth[{pick_device()}]: {dt * 1000:.0f} ms, "
        f"range {depth.min():.2f}-{depth.max():.2f} m, median {np.median(depth):.2f} m"
    )
    assert depth.shape == rgb.shape[:2]
    # Metric-indoor checkpoint: values must be metres, not millimetres or
    # normalized relative depth.
    assert 0.05 < float(np.median(depth)) < 20.0


def test_depth_downscale_path(estimator):
    estimator.input_long_side = 384
    try:
        rgb = make_rgb_frame(1280, 720)
        depth = estimator.estimate(rgb)
        assert depth.shape == (720, 1280)
    finally:
        estimator.input_long_side = None
