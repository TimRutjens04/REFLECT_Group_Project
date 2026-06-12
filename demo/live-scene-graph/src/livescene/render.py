"""Split-view rendering, pure cv2 (no matplotlib in the live loop).

Left panel: webcam frame + bboxes coloured by node status.
Right panel: the scene graph drawn at image coordinates — each node sits at
its pixel_center, so the graph spatially mirrors the scene.

Palettes are carried over from the REFLECT visualizer (_REL_STYLE,
_STATUS_COLOR, _BBOX_BGR).
"""

from __future__ import annotations

import numpy as np

import cv2

from .scene_graph.models import SceneGraphFrame


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# relation -> (BGR, dashed?, thickness); colours from the REFLECT _REL_STYLE.
_REL_STYLE: dict[str, tuple[tuple[int, int, int], bool, int]] = {
    "near": (_hex_to_bgr("#AAAAAA"), True, 1),
    "inside": (_hex_to_bgr("#3A7EBF"), False, 2),
    "on_top_of": (_hex_to_bgr("#E07B39"), False, 2),
    "above": (_hex_to_bgr("#3A9E5F"), False, 2),
    "below": (_hex_to_bgr("#3A9E5F"), False, 2),
    "left_of": (_hex_to_bgr("#9B59B6"), True, 2),
    "held_by_gripper": (_hex_to_bgr("#E74C3C"), False, 3),
}
_DEFAULT_REL = (_hex_to_bgr("#888888"), True, 1)

# node colour by status; from the REFLECT _STATUS_COLOR.
_STATUS_BGR = {
    "ok": _hex_to_bgr("#2ECC71"),
    "occluded": _hex_to_bgr("#E67E22"),
    "uncertain": _hex_to_bgr("#E74C3C"),
    "drifting": _hex_to_bgr("#F1C40F"),
    "gripper": _hex_to_bgr("#9B59B6"),
}

# bbox colour by status (already BGR in the REFLECT visualizer).
_BBOX_BGR = {
    "ok": (50, 205, 50),
    "occluded": (0, 128, 255),
    "uncertain": (0, 0, 220),
    "drifting": (0, 200, 255),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GRAPH_BG = (28, 26, 24)


def _dashed_line(img, p1, p2, color, thickness, dash=8):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    length = float(np.linalg.norm(p2 - p1))
    if length < 1:
        return
    n = max(1, int(length / dash))
    for i in range(0, n, 2):
        a = p1 + (p2 - p1) * (i / n)
        b = p1 + (p2 - p1) * (min(i + 1, n) / n)
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, cv2.LINE_AA)


def _text(img, text, org, scale=0.45, color=(235, 235, 235), thickness=1, bg=None):
    if bg is not None:
        (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + baseline), bg, -1)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def draw_bbox_panel(frame_bgr: np.ndarray, sg: SceneGraphFrame) -> np.ndarray:
    panel = frame_bgr.copy()
    for node in sg.nodes:
        if len(node.bbox_xyxy) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in node.bbox_xyxy)
        status = node.status.value
        color = _BBOX_BGR.get(status, (180, 180, 180))
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
        label = f"{node.label} {node.confidence:.2f} {node.depth_used_m:.2f}m"
        if status != "ok":
            label += f" [{status}]"
        _text(panel, label, (x1, max(14, y1 - 6)), color=color, bg=(0, 0, 0))
    return panel


def draw_graph_panel(sg: SceneGraphFrame, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), _GRAPH_BG, dtype=np.uint8)
    centers = {n.object_id: (int(n.pixel_center[0]), int(n.pixel_center[1])) for n in sg.nodes}

    for edge in sg.edges:
        a = centers.get(edge.from_object_id)
        b = centers.get(edge.to_object_id)
        if a is None or b is None:
            continue
        color, dashed, thickness = _REL_STYLE.get(edge.relation, _DEFAULT_REL)
        if dashed:
            _dashed_line(panel, a, b, color, thickness)
        else:
            cv2.line(panel, a, b, color, thickness, cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        _text(panel, edge.relation, (mid[0] + 4, mid[1] - 4), color=color, bg=_GRAPH_BG)

    for node in sg.nodes:
        c = centers[node.object_id]
        color = _STATUS_BGR.get(node.status.value, (150, 150, 150))
        cv2.circle(panel, c, 9, color, -1, cv2.LINE_AA)
        cv2.circle(panel, c, 9, (240, 240, 240), 1, cv2.LINE_AA)
        _text(panel, f"{node.label} {node.depth_used_m:.2f}m", (c[0] - 20, c[1] + 24), color=color)

    if sg.localization_flag.failure_detected:
        flag = sg.localization_flag
        msg = f"{flag.type.value if flag.type else 'failure'}: {flag.affected_object_id or '?'}"
        _text(panel, msg, (10, height - 12), scale=0.55, color=(60, 76, 231), bg=_GRAPH_BG)
    return panel


def compose(
    left: np.ndarray,
    right: np.ndarray,
    prompts: list[str],
    fps: float | None = None,
) -> np.ndarray:
    view = np.hstack([left, right])
    header = f"prompts: {', '.join(prompts)}"
    if fps is not None:
        header += f"   |   {fps:.1f} FPS"
    _text(view, header, (10, 22), scale=0.6, bg=(0, 0, 0))
    _text(view, "type new prompts in the terminal + Enter; q quits", (10, 44), scale=0.45, color=(180, 180, 180), bg=(0, 0, 0))
    return view
