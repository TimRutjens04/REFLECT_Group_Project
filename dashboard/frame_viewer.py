"""Draw detection and tracking bboxes on RGB frames using PIL.

Color legend (Change 3):
  #2ca02c green  — DET (Grounding DINO detection)
  #1f77b4 blue   — TRK (tracker box)
  #d62728 red    — depth-flagged (dormant until SHOW_DEPTH_TAB=True)

Scene graph overlay (Frame Viewer "Scene graph" view):
  node circles colored per label via _SG_PALETTE; edges dark gray with
  relation text at the midpoint.
"""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

_DET_COLOR   = "#2ca02c"  # green
_TRK_COLOR   = "#1f77b4"  # blue
_DEPTH_COLOR = "#d62728"  # red


def _load_font(size: int = 13) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    bbox: list,
    outline: str,
    label: str,
    font: ImageFont.FreeTypeFont,
    width: int = 2,
) -> None:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    draw.rectangle([x0, y0, x1, y1], outline=outline, width=width)

    # Filled background behind the label for legibility
    pad = 3
    lx, ly = x0 + 2, max(0, y0 - 18)
    try:
        tb = draw.textbbox((lx, ly), label, font=font)
    except AttributeError:
        # PIL < 9.2: fall back to textsize
        w, h = draw.textsize(label, font=font)  # type: ignore[attr-defined]
        tb = (lx, ly, lx + w, ly + h)
    draw.rectangle(
        [tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad],
        fill=outline,
    )
    draw.text((lx, ly), label, fill="white", font=font)


def draw_bboxes(
    frame_path: str,
    detections: list[dict],
    tracked_objects: list[dict],
    depth_flagged_ids: set[str],
) -> Image.Image:
    """
    Returns a PIL image with bboxes drawn:

      DET boxes  — green (#2ca02c), label: "DET {label} {conf:.2f}"
      TRK boxes  — blue  (#1f77b4), label: "TRK {object_id}"
      Depth-flag — red   (#d62728) outer outline (dormant in v3, code path present)
    """
    img  = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(13)

    # Detection bboxes (green)
    for det in detections:
        bbox = det.get("bbox_xyxy")
        if not bbox:
            continue
        oid   = det.get("object_id", "")
        label = det.get("label") or oid
        conf  = det.get("confidence") or 0.0
        _draw_labeled_box(draw, bbox, _DET_COLOR, f"DET {label} {conf:.2f}", font)
        if oid in depth_flagged_ids:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            draw.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=_DEPTH_COLOR, width=3)

    # Tracker bboxes (blue)
    for obj in tracked_objects:
        bbox = obj.get("bbox_xyxy")
        if not bbox:
            continue
        oid  = obj.get("object_id", "")
        _draw_labeled_box(draw, bbox, _TRK_COLOR, f"TRK {oid}", font)
        if oid in depth_flagged_ids:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            draw.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=_DEPTH_COLOR, width=3)

    return img


# ---------------------------------------------------------------------------
# Scene graph overlay
# ---------------------------------------------------------------------------

_SG_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
]

_SG_NODE_RADIUS = 22
_SG_EDGE_COLOR  = (68, 68, 68, 255)   # #444


def _color_for_label(label: str) -> str:
    # Sum-of-ordinals rather than hash(): Python's hash() is randomized per
    # process, so the same label would change color between Streamlit reruns.
    h = sum(ord(c) for c in label or "") % len(_SG_PALETTE)
    return _SG_PALETTE[h]


def _hex_to_rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def _compact_id(object_id: str) -> str:
    """'dark blue bowl_1' → 'bowl_1'; fallback: first 8 chars of object_id."""
    try:
        label_part, suffix = object_id.rsplit("_", 1)
        return f"{label_part.split()[-1]}_{suffix}"
    except (ValueError, IndexError, AttributeError):
        return str(object_id or "")[:8]


def _valid_center(pc) -> tuple[float, float] | None:
    if pc is None or not isinstance(pc, (list, tuple)) or len(pc) != 2:
        return None
    try:
        u, v = float(pc[0]), float(pc[1])
    except (TypeError, ValueError):
        return None
    if math.isnan(u) or math.isnan(v):
        return None
    return u, v


def _text_extent(draw, text: str, font) -> tuple[float, float, float, float]:
    """Returns (width, height, x_offset, y_offset) for centered placement."""
    try:
        tb = draw.textbbox((0, 0), text, font=font)
        return tb[2] - tb[0], tb[3] - tb[1], tb[0], tb[1]
    except AttributeError:
        # PIL < 9.2: fall back to textsize
        w, h = draw.textsize(text, font=font)  # type: ignore[attr-defined]
        return w, h, 0, 0


def draw_scene_graph_overlay(
    frame_path: str,
    nodes: list[dict],
    edges: list[dict],
) -> Image.Image:
    """
    Overlay scene graph on the RGB frame.
      nodes — list of dicts with keys: object_id, label, pixel_center [u, v], state (optional)
      edges — list of dicts with keys: from, to, relation, distance_3d_m (optional)
    Edges are drawn under the node circles so circles stay readable.
    """
    img     = Image.open(frame_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    W, H    = img.size

    font_rel  = _load_font(12)
    font_dist = _load_font(10)
    font_node = _load_font(11)

    centers: dict[str, tuple[float, float]] = {}
    for node in nodes:
        pc = _valid_center(node.get("pixel_center"))
        if pc is not None:
            centers[node.get("object_id")] = pc

    # ── Edges (under circles) ────────────────────────────────────────────────
    for edge in edges:
        a, b = edge.get("from"), edge.get("to")
        if a not in centers or b not in centers:
            continue
        (x0, y0), (x1, y1) = centers[a], centers[b]
        draw.line([x0, y0, x1, y1], fill=_SG_EDGE_COLOR, width=2)

        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if not (0 <= mx <= W and 0 <= my <= H):
            continue  # clip label rather than draw off-image
        relation = str(edge.get("relation") or "")
        if not relation:
            continue

        lines = [(relation, font_rel)]
        dist = edge.get("distance_3d_m")
        if dist is not None and not (isinstance(dist, float) and math.isnan(dist)):
            lines.append((f"{float(dist):.2f}m", font_dist))

        gap     = 2
        extents = [_text_extent(draw, t, f) for t, f in lines]
        block_w = max(w for w, _, _, _ in extents)
        block_h = sum(h for _, h, _, _ in extents) + gap * (len(lines) - 1)
        pad     = 3
        rect = [mx - block_w / 2 - pad, my - block_h / 2 - pad,
                mx + block_w / 2 + pad, my + block_h / 2 + pad]
        try:
            draw.rounded_rectangle(rect, radius=4, fill=(255, 255, 255, 255))
        except AttributeError:
            # PIL < 8.2: plain rectangle
            draw.rectangle(rect, fill=(255, 255, 255, 255))
        ty = my - block_h / 2
        for (text, f), (w, h, ox, oy) in zip(lines, extents):
            draw.text((mx - w / 2 - ox, ty - oy), text, fill=(34, 34, 34, 255), font=f)
            ty += h + gap

    # ── Node circles on top ──────────────────────────────────────────────────
    r = _SG_NODE_RADIUS
    for node in nodes:
        oid = node.get("object_id")
        if oid not in centers:
            continue
        u, v  = centers[oid]
        color = _color_for_label(node.get("label") or str(oid or ""))
        draw.ellipse(
            [u - r, v - r, u + r, v + r],
            fill=_hex_to_rgba(color, 178),     # ~70% opacity
            outline=_hex_to_rgba(color, 255),
            width=2,
        )
        text = _compact_id(str(oid or ""))
        w, h, ox, oy = _text_extent(draw, text, font_node)
        draw.text((u - w / 2 - ox, v - h / 2 - oy), text,
                  fill=(255, 255, 255, 255), font=font_node)

    return Image.alpha_composite(img, overlay).convert("RGB")
