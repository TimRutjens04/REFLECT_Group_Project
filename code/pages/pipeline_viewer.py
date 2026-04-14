"""
Pipeline Viewer — Streamlit page

Interactive frame-by-frame browser for the perception pipeline outputs.
Reuses the rendering helpers from visualize.py so the display matches
the exported video exactly.

Access via: just gui  → sidebar → "pipeline_viewer"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# ── make code/ importable so we can reuse visualize helpers ───────────────────
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from visualize import (  # noqa: E402
    _apply_masks,
    _draw_detections,
    _render_depth_panel,
    _render_footer,
    FOOTER_H,
)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT             = CODE_DIR.parent
ALIGNED_DIR      = ROOT / "aligned"
DETECT_DIR       = ROOT / "detect"
SEGMENT_DIR      = ROOT / "segment"
DEPTH_STATE_DIR  = ROOT / "depth_state"
TRACK_DIR        = ROOT / "track"
GRAPHS_DIR       = ROOT / "scene_graphs_pipeline"

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline Viewer",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Pipeline Viewer")
st.caption("Frame-by-frame inspection of Stage 1–5 perception outputs")


# ── episode list ───────────────────────────────────────────────────────────────
@st.cache_data
def list_pipeline_episodes() -> list[str]:
    return sorted(p.stem for p in GRAPHS_DIR.glob("*.json"))


episodes = list_pipeline_episodes()
if not episodes:
    st.error("No pipeline outputs found in scene_graphs_pipeline/. Run `just full-pipeline` first.")
    st.stop()


# ── sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    episode_id = st.selectbox("Episode", episodes)
    st.divider()
    autoplay   = st.toggle("Auto-play", value=False)
    play_speed = st.slider("Speed (fps)", 1, 10, 4) if autoplay else None
    st.divider()
    show_masks  = st.checkbox("Show SAM 2 masks", value=True)
    show_depth  = st.checkbox("Show depth panel", value=True)
    show_graph  = st.checkbox("Show scene graph", value=True)


# ── load episode data (cached per episode) ─────────────────────────────────────
@st.cache_data
def load_episode_data(ep: str) -> dict:
    aligned_path     = ALIGNED_DIR     / f"{ep}.npz"
    detect_path      = DETECT_DIR      / f"{ep}.npz"
    segment_path     = SEGMENT_DIR     / f"{ep}.npz"
    depth_state_path = DEPTH_STATE_DIR / f"{ep}.npz"
    track_path       = TRACK_DIR       / f"{ep}.npz"
    graph_path       = GRAPHS_DIR      / f"{ep}.json"

    if not aligned_path.exists() or not detect_path.exists():
        return {}

    aligned = np.load(aligned_path, allow_pickle=True)
    det     = np.load(detect_path,  allow_pickle=True)

    frames         = aligned["frames"]
    timestamps     = aligned["timestamps"]
    failure_labels = aligned["failure_labels"]
    all_boxes      = det["boxes"]
    all_n_dets     = det["n_dets"]
    label_vocab    = list(det["label_vocab"])
    all_label_ids  = det["label_ids"]

    N, H, W, _ = frames.shape

    if segment_path.exists():
        seg = np.load(segment_path, allow_pickle=True)
        masks_small = seg["masks_small"]
        mask_valid  = seg["mask_valid"]
    else:
        masks_small = np.zeros((N, 1, 1, 1), dtype=np.uint8)
        mask_valid  = np.zeros((N, 1), dtype=bool)

    if depth_state_path.exists():
        ds = np.load(depth_state_path, allow_pickle=True)
        depth_maps = ds["depth_maps"]
        obj_depth  = ds["obj_depth"]
        obj_state  = ds["obj_state"]
        state_vocab = list(ds["state_vocab"])
    else:
        depth_maps  = np.zeros((N, H // 2, W // 2), dtype=np.float32)
        obj_depth   = np.full((N, 1), np.nan, dtype=np.float32)
        obj_state   = np.full((N, 1), -1, dtype=np.int32)
        state_vocab = []

    if track_path.exists():
        trk = np.load(track_path, allow_pickle=True)
        track_ids   = trk["track_ids"]
        track_conf  = trk["tracking_confidence"]
        held        = trk["held_by_gripper"]
    else:
        track_ids   = np.zeros((N, 1), dtype=np.int32)
        track_conf  = np.ones((N, 1), dtype=np.float32)
        held        = np.zeros((N, 1), dtype=bool)

    frame_graphs: list[dict] = []
    if graph_path.exists():
        with open(graph_path) as f:
            frame_graphs = json.load(f).get("frames", [])

    return {
        "frames": frames, "timestamps": timestamps,
        "failure_labels": failure_labels,
        "all_boxes": all_boxes, "all_n_dets": all_n_dets,
        "label_vocab": label_vocab, "all_label_ids": all_label_ids,
        "masks_small": masks_small, "mask_valid": mask_valid,
        "depth_maps": depth_maps, "obj_depth": obj_depth,
        "obj_state": obj_state, "state_vocab": state_vocab,
        "track_ids": track_ids, "track_conf": track_conf, "held": held,
        "frame_graphs": frame_graphs,
        "N": N, "H": H, "W": W,
    }


data = load_episode_data(episode_id)
if not data:
    st.error(f"Could not load data for {episode_id}")
    st.stop()

N = data["N"]
H = data["H"]
W = data["W"]

# failure frame indices for quick navigation
fail_indices = [i for i in range(N) if data["failure_labels"][i]]


# ── frame selector ─────────────────────────────────────────────────────────────
col_slider, col_jump = st.columns([4, 1])

with col_slider:
    frame_idx = st.slider("Frame", 0, N - 1, value=fail_indices[0] if fail_indices else 0)

with col_jump:
    if fail_indices:
        if st.button("⚠ Jump to failure"):
            frame_idx = fail_indices[0]
            st.rerun()

# autoplay via fragment rerun
if autoplay and play_speed:
    import time
    time.sleep(1.0 / play_speed)
    next_frame = (frame_idx + 1) % N
    st.query_params["f"] = str(next_frame)
    st.rerun()


# ── render current frame ───────────────────────────────────────────────────────
i   = frame_idx
k   = int(data["all_n_dets"][i])
ts  = float(data["timestamps"][i])
fail = bool(data["failure_labels"][i])

import cv2  # noqa: E402 — imported here to avoid Streamlit watcher noise

# left panel
left = _apply_masks(
    data["frames"][i].copy(),
    data["masks_small"][i] if show_masks else np.zeros_like(data["masks_small"][i]),
    data["mask_valid"][i],
    data["track_ids"][i],
    k, H, W,
)
_draw_detections(
    left,
    data["all_boxes"][i], k,
    data["label_vocab"], data["all_label_ids"][i],
    data["track_ids"][i], data["track_conf"][i],
    data["held"][i], data["state_vocab"], data["obj_state"][i],
)
if fail:
    cv2.rectangle(left, (0, 0), (W - 1, H - 1), (0, 0, 220), 8)
cv2.putText(left, "Detection + Segmentation + Tracking",
            (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# right panel
right = _render_depth_panel(
    data["depth_maps"][i], H, W,
    data["all_boxes"][i], k,
    data["track_ids"][i], data["obj_depth"][i],
    data["state_vocab"], data["obj_state"][i],
)
if fail:
    cv2.rectangle(right, (0, 0), (W - 1, H - 1), (0, 0, 220), 8)
cv2.putText(right, "Depth Map (plasma) + Object States",
            (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# footer
fg = data["frame_graphs"][i] if i < len(data["frame_graphs"]) else {}
empty_flag = {"failure_detected": False, "type": None, "affected_object_ids": []}
flag       = fg.get("localization_flag", empty_flag)
relations  = fg.get("spatial_relations", [])

footer = _render_footer(i, ts, fail, flag, k, relations, W * 2)

# convert right panel BGR→RGB for st.image
right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)


# ── display ────────────────────────────────────────────────────────────────────
# failure banner
if fail:
    flag_type = flag.get("type") or "Unknown"
    affected  = flag.get("affected_object_ids", [])
    st.error(f"⚠ FAILURE FRAME — type: **{flag_type}** | affected IDs: {affected}")
else:
    st.success(f"Frame {i} — t={ts:.1f}s — normal")

# panels
if show_depth:
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(left, caption="Detection + Segmentation + Tracking", use_container_width=True)
    with img_col2:
        st.image(right_rgb, caption="Depth Map + Object States", use_container_width=True)
else:
    st.image(left, caption="Detection + Segmentation + Tracking", use_container_width=True)

# footer strip
footer_rgb = cv2.cvtColor(footer, cv2.COLOR_BGR2RGB)
st.image(footer_rgb, use_container_width=True)


# ── scene graph details ────────────────────────────────────────────────────────
if show_graph and fg:
    with st.expander("Scene graph — objects & relations", expanded=fail):
        obj_col, rel_col = st.columns(2)

        with obj_col:
            st.subheader("Objects")
            objs = fg.get("objects", [])
            if objs:
                rows = []
                for o in objs:
                    rows.append({
                        "ID":    o["id"],
                        "Label": o["label"],
                        "State": o["state"],
                        "Depth": f"{o['depth']:.3f}" if o["depth"] is not None else "—",
                        "Held":  "⚡" if o["held_by_gripper"] else "",
                        "Conf":  f"{o['tracking_confidence']:.2f}",
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.caption("No objects detected")

        with rel_col:
            st.subheader("Spatial Relations")
            if relations:
                rows = [{"Subject": r["subject"], "Relation": r["relation"],
                         "Object": r["object"]} for r in relations]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.caption("No relations")

# ── episode timeline ───────────────────────────────────────────────────────────
with st.expander("Episode timeline"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 1.2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    colors = ["#e74c3c" if data["failure_labels"][f] else "#3498db" for f in range(N)]
    ax.bar(range(N), [1] * N, color=colors, width=1.0)
    ax.axvline(frame_idx, color="#f0c040", linewidth=2.5, label=f"frame {frame_idx}")
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Frame", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=8)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
