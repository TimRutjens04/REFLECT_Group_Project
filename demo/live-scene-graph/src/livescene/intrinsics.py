"""Camera intrinsics derived from resolution + horizontal FOV (no calibration).

3D positions computed from these are approximate; they are good enough for
coarse spatial relations, not for metrology.
"""

from __future__ import annotations

import math

import numpy as np


def intrinsics_from_fov(width: int, height: int, hfov_deg: float = 60.0) -> np.ndarray:
    """Return a 3x3 pinhole K from frame size and horizontal field of view.

    Assumes square pixels (fy = fx) and the principal point at the image
    centre.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError(f"hfov_deg must be in (0, 180), got {hfov_deg}")
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
