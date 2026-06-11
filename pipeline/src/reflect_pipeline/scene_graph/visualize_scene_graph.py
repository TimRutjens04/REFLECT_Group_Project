"""
Scene-graph frame-overlay visualizer.

For each selected frame: video frame on left (with bounding boxes + labels),
scene graph on right (nodes positioned by image coords, edges colour-coded by
relation type).  Outputs a PNG keyframe strip and/or an MP4.

Usage
-----
    # Keyframe strip PNG (flag frames + sampled normal)
    poetry run python3 visualize_scene_graph.py \
        --sg /tmp/scene_graph_putAppleBowl1.jsonl \
        --video example_data/real_data/putAppleBowl1/videos/color.mp4 \
        --out /tmp/sg_vis

    # Full MP4 (every frame, left=bbox overlay, right=scene graph)
    poetry run python3 visualize_scene_graph.py \
        --sg ... --video ... --out /tmp/sg_vis --mp4 --fps 10

    # Flag-frames-only MP4
    poetry run python3 visualize_scene_graph.py ... --mp4 --keyframes-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ── relation style ──────────────────────────────────────────────────────────
_REL_STYLE: dict[str, dict] = {
    "near":            {"color": "#AAAAAA", "style": "dashed",  "lw": 1.2},
    "inside":          {"color": "#3A7EBF", "style": "solid",   "lw": 2.0},
    "on_top_of":       {"color": "#E07B39", "style": "solid",   "lw": 2.0},
    "above":           {"color": "#3A9E5F", "style": "solid",   "lw": 1.5},
    "below":           {"color": "#3A9E5F", "style": "solid",   "lw": 1.5},
    "left_of":         {"color": "#9B59B6", "style": "dashed",  "lw": 1.5},
    "held_by_gripper": {"color": "#E74C3C", "style": "solid",   "lw": 2.5},
}
_DEFAULT_REL = {"color": "#888888", "style": "dotted", "lw": 1.0}

# ── node colour by status ───────────────────────────────────────────────────
_STATUS_COLOR = {
    "ok":        "#2ECC71",
    "occluded":  "#E67E22",
    "uncertain": "#E74C3C",
    "drifting":  "#F1C40F",
    "gripper":   "#9B59B6",
}

# ── bbox colour by status (BGR) ─────────────────────────────────────────────
_BBOX_BGR = {
    "ok":        (50,  205, 50),
    "occluded":  (0,   128, 255),
    "uncertain": (0,   0,   220),
    "drifting":  (0,   200, 255),
}

# ── frame border colour ─────────────────────────────────────────────────────
_BORDER_COLOR = {
    "flag":   "#E74C3C",
    "normal": "#3A7EBF",
    "final":  "#2ECC71",
}


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_frame(cap: cv2.VideoCapture, frame_id: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, bgr = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_id}")
    return bgr


def _draw_bboxes(bgr: np.ndarray, nodes: list[dict]) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, w / 2000)
    thick = max(1, w // 600)

    for n in nodes:
        oid = n.get("object_id", "")
        if oid == "gripper":
            continue
        bbox = n.get("bbox_xyxy", [])
        if not bbox or len(bbox) < 4:
            continue
        status = n.get("status", "ok")
        color = _BBOX_BGR.get(status, (180, 180, 180))
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick + 1)

        label = n.get("label", oid)
        conf = n.get("confidence", 0.0)
        depth = n.get("depth_used_m", 0.0)
        text = f"{label}  {conf:.2f}  {depth:.2f}m"
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - bl - 2), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(out, text, (x1 + 2, ty - bl), font, scale, (0, 0, 0), thick)

    return out


def _draw_sg_panel(
    ax: plt.Axes,
    nodes: list[dict],
    edges: list[dict],
    img_w: int,
    img_h: int,
) -> None:
    """Draw scene graph on ax with nodes at their image-space pixel positions."""
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)  # flipped to match image y-axis
    ax.set_aspect("equal")
    ax.axis("off")

    pos: dict[str, tuple[float, float]] = {}
    for n in nodes:
        oid = n.get("object_id", "")
        if n.get("pixel_center"):
            px, py = n["pixel_center"]
            pos[oid] = (float(px), float(py))

    # Edges (drawn first, behind nodes)
    for e in edges:
        fid = e.get("from_object_id", "")
        tid = e.get("to_object_id", "")
        if fid not in pos or tid not in pos:
            continue
        rel = e.get("relation", "")
        style = _REL_STYLE.get(rel, _DEFAULT_REL)
        x1, y1 = pos[fid]
        x2, y2 = pos[tid]
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->" if rel != "near" else "-",
                color=style["color"],
                lw=style["lw"],
                linestyle=style["style"],
            ),
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, rel.replace("_", " "), fontsize=5, color=style["color"],
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))

    # Nodes
    for n in nodes:
        oid = n.get("object_id", "")
        if oid not in pos:
            continue
        status = "gripper" if oid == "gripper" else n.get("status", "ok")
        color = _STATUS_COLOR.get(status, "#95A5A6")
        px, py = pos[oid]
        ax.scatter([px], [py], s=220, c=color, zorder=5, edgecolors="white", linewidths=1.5)
        label = n.get("label", oid)
        offset = -img_h * 0.045
        ax.text(px, py + offset, label, fontsize=6, ha="center", va="bottom", color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="#222233", ec="none", alpha=0.85))


def render_strip(
    sg_path: str | Path,
    video_path: str | Path,
    out_dir: str | Path,
    max_panels: int = 12,
) -> Path:
    """PNG strip: two rows (video | scene graph) for each selected keyframe."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(sg_path)
    cap = cv2.VideoCapture(str(video_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    flag_rows   = [r for r in rows if r.get("localization_flag", {}).get("failure_detected")]
    normal_rows = [r for r in rows if not r.get("localization_flag", {}).get("failure_detected")]

    selected: list[dict] = [rows[0]]

    # Up to half the budget from flag frames, evenly sampled
    budget_flags = max(1, max_panels // 2)
    if len(flag_rows) > budget_flags:
        step = len(flag_rows) // budget_flags
        flag_rows = flag_rows[::step][:budget_flags]
    selected.extend(flag_rows)

    remaining = max_panels - len(selected)
    if remaining > 0 and normal_rows:
        step = max(1, len(normal_rows) // remaining)
        selected.extend(normal_rows[::step][:remaining])

    if rows[-1] not in selected:
        selected.append(rows[-1])

    seen: set[int] = set()
    unique: list[dict] = []
    for r in sorted(selected, key=lambda x: x["frame_id"]):
        if r["frame_id"] not in seen:
            seen.add(r["frame_id"])
            unique.append(r)

    n = len(unique)
    print(f"Rendering {n}-panel strip...")

    panel_w_in = 5.0
    panel_h_in = panel_w_in * (img_h / img_w) + 0.4
    fig, axes = plt.subplots(2, n, figsize=(panel_w_in * n, panel_h_in * 2))
    fig.patch.set_facecolor("#1A1A2E")
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, row in enumerate(unique):
        frame_id = int(row["frame_id"])
        is_final = col == len(unique) - 1
        nodes    = row.get("nodes", [])
        edges    = row.get("edges", [])
        flag     = row.get("localization_flag", {})
        ts       = row.get("timestamp", 0.0)

        bgr = _read_frame(cap, frame_id)
        rgb = cv2.cvtColor(_draw_bboxes(bgr, nodes), cv2.COLOR_BGR2RGB)

        border = (_BORDER_COLOR["final"] if is_final
                  else _BORDER_COLOR["flag"] if flag.get("failure_detected")
                  else _BORDER_COLOR["normal"])

        flag_type = flag.get("type", "") if flag.get("failure_detected") else ""
        title = f"f{frame_id}  t={ts:.1f}s"
        if flag_type:
            title += f"\n[{flag_type}]"

        ax_img = axes[0, col]
        ax_img.imshow(rgb)
        ax_img.axis("off")
        ax_img.set_title(title, color="white", fontsize=6, pad=2)
        for spine in ax_img.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

        ax_sg = axes[1, col]
        ax_sg.set_facecolor("#12122A")
        ax_sg.imshow(rgb, alpha=0.15)
        _draw_sg_panel(ax_sg, nodes, edges, img_w, img_h)
        for spine in ax_sg.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

    legend_handles = [
        mpatches.Patch(color=_BORDER_COLOR["flag"],   label="failure flag"),
        mpatches.Patch(color=_BORDER_COLOR["normal"], label="normal"),
        mpatches.Patch(color=_BORDER_COLOR["final"],  label="final"),
    ] + [
        mpatches.Patch(color=s["color"], label=rel.replace("_", " "))
        for rel, s in _REL_STYLE.items()
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles),
               fontsize=6, framealpha=0.3, labelcolor="white",
               facecolor="#1A1A2E", edgecolor="none")

    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.3)
    out_path = out_dir / "keyframe_strip.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    cap.release()
    print(f"Saved → {out_path}")
    return out_path


def render_mp4(
    sg_path: str | Path,
    video_path: str | Path,
    out_dir: str | Path,
    fps: int = 10,
    keyframes_only: bool = False,
) -> Path:
    """MP4: left = bbox overlay, right = scene graph on faint frame ghost."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(sg_path)
    if keyframes_only:
        flag_ids   = {r["frame_id"] for r in rows if r.get("localization_flag", {}).get("failure_detected")}
        sample_ids = {r["frame_id"] for i, r in enumerate(rows) if i % 30 == 0}
        keep = flag_ids | sample_ids | {rows[0]["frame_id"], rows[-1]["frame_id"]}
        rows = [r for r in rows if r["frame_id"] in keep]

    cap = cv2.VideoCapture(str(video_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w, out_h = img_w * 2, img_h
    fname = "keyframes.mp4" if keyframes_only else "scene_graph.mp4"
    out_path = out_dir / fname
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    dpi = 100
    fig_w_in = img_w / dpi
    fig_h_in = img_h / dpi

    print(f"Rendering {len(rows)} frames → {out_path}")
    for i, row in enumerate(rows):
        frame_id = int(row["frame_id"])
        nodes    = row.get("nodes", [])
        edges    = row.get("edges", [])
        flag     = row.get("localization_flag", {})
        ts       = row.get("timestamp", 0.0)

        bgr = _read_frame(cap, frame_id)
        rgb = cv2.cvtColor(_draw_bboxes(bgr, nodes), cv2.COLOR_BGR2RGB)

        border    = _BORDER_COLOR["flag"] if flag.get("failure_detected") else _BORDER_COLOR["normal"]
        flag_type = flag.get("type", "") if flag.get("failure_detected") else ""

        fig, ax_sg = plt.subplots(1, 1, figsize=(fig_w_in, fig_h_in), dpi=dpi)
        fig.patch.set_facecolor("#12122A")
        ax_sg.set_facecolor("#12122A")
        ax_sg.imshow(rgb, alpha=0.15)
        _draw_sg_panel(ax_sg, nodes, edges, img_w, img_h)
        ax_sg.set_title(f"f{frame_id}  t={ts:.1f}s  {flag_type}", color="white", fontsize=7, pad=2)
        for spine in ax_sg.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(3)
            spine.set_visible(True)
        fig.tight_layout(pad=0)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, :3]  # drop alpha
        plt.close(fig)

        sg_bgr = cv2.resize(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR), (img_w, img_h))
        composite = np.concatenate([_draw_bboxes(bgr, nodes), sg_bgr], axis=1)

        bar = tuple(int(c * 255) for c in matplotlib.colors.to_rgb(border))[::-1]
        cv2.rectangle(composite, (0, 0), (out_w, 5), bar, -1)

        writer.write(composite)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}")

    writer.release()
    cap.release()
    print(f"Saved → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sg",    required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out",   required=True)
    parser.add_argument("--mp4",   action="store_true")
    parser.add_argument("--fps",   type=int, default=10)
    parser.add_argument("--keyframes-only", action="store_true")
    parser.add_argument("--panels", type=int, default=12)
    args = parser.parse_args()

    render_strip(args.sg, args.video, args.out, max_panels=args.panels)

    if args.mp4:
        render_mp4(args.sg, args.video, args.out, fps=args.fps,
                   keyframes_only=args.keyframes_only)


if __name__ == "__main__":
    main()
