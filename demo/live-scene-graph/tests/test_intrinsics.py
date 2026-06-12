import math

import numpy as np
import pytest

from livescene.intrinsics import intrinsics_from_fov


def test_fx_from_fov_90deg():
    # tan(45 deg) == 1, so fx == W/2 exactly.
    K = intrinsics_from_fov(640, 480, hfov_deg=90.0)
    assert K[0, 0] == pytest.approx(320.0)


def test_fx_from_fov_60deg():
    K = intrinsics_from_fov(1280, 720, hfov_deg=60.0)
    expected = (1280 / 2) / math.tan(math.radians(30.0))
    assert K[0, 0] == pytest.approx(expected)
    assert K[0, 0] == pytest.approx(1108.5, abs=0.1)


def test_square_pixels_and_principal_point():
    K = intrinsics_from_fov(1280, 720, hfov_deg=60.0)
    assert K[1, 1] == K[0, 0]
    assert K[0, 2] == pytest.approx(640.0)
    assert K[1, 2] == pytest.approx(360.0)
    assert K.shape == (3, 3)
    assert K[2, 2] == 1.0


def test_back_projection_roundtrip():
    # A pixel at the image centre back-projects onto the optical axis.
    K = intrinsics_from_fov(640, 480, hfov_deg=60.0)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = 2.0
    x = (cx - cx) * z / fx
    y = (cy - cy) * z / fy
    assert (x, y) == (0.0, 0.0)
    # The right image edge at depth z maps to x = z * tan(hfov/2).
    x_edge = (640 - cx) * z / fx
    assert x_edge == pytest.approx(z * math.tan(math.radians(30.0)))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        intrinsics_from_fov(0, 480, 60.0)
    with pytest.raises(ValueError):
        intrinsics_from_fov(640, 480, 0.0)
    with pytest.raises(ValueError):
        intrinsics_from_fov(640, 480, 180.0)


def test_dtype_float():
    K = intrinsics_from_fov(640, 480, 60.0)
    assert K.dtype == np.float64
