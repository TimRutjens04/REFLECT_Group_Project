import time

import matplotlib.pyplot as plt
import streamlit as st

from .attention import compute_attention_maps, draw_object_box, overlay_attention
from .data import list_episodes, load_episode, load_task_metadata
from .localization import compute_scene_graph
from .logger import log
from .models import load_clip_model, load_owlvit_model
from .plots import plot_inline_scene_graph, plot_pca, plot_scene_graph, plot_signals


def _render_sidebar() -> tuple[str, float, bool, bool, float, bool]:
    """Render sidebar controls. Returns (episode_id, speed, playing, show_attention, attn_alpha, show_scene_graph)."""
    with st.sidebar:
        st.title("Controls")

        episodes = list_episodes()
        if not episodes:
            st.error("No episodes found in aligned/ and encoded/.")
            st.stop()

        episode_id = st.selectbox("Episode", episodes)
        st.divider()

        playing = st.session_state.get("playing", False)
        if st.button("⏸ Pause" if playing else "▶ Play", use_container_width=True):
            st.session_state["playing"] = not playing
            st.rerun()

        speed = st.select_slider(
            "Playback speed", options=[0.25, 0.5, 1.0, 2.0, 4.0],
            value=st.session_state.get("speed", 1.0),
        )
        st.session_state["speed"] = speed
        st.divider()

        show_attention = st.toggle("Show attention overlay", value=False)
        if show_attention:
            attn_alpha = st.slider("Heatmap opacity", 0.1, 0.8, 0.4, 0.05)
            show_scene_graph = st.toggle("Show scene graph", value=True)
            st.caption(
                "Red box = highest-attention patch mapped to image coords. "
                "Yellow-bordered nodes in graph = within top-3 attention patches."
            )
        else:
            attn_alpha, show_scene_graph = 0.4, False

        st.divider()
        st.caption("Frames sampled at 2fps")
        st.caption("Embeddings: CLIP + WAV2CLIP")

    return episode_id, speed, playing, show_attention, attn_alpha, show_scene_graph


def _load_perception_data(episode_id: str, show_attention: bool,
                           show_scene_graph: bool, object_list: list[str]):
    """Load attention maps (ViT-B/32) and scene graph (OWL-ViT v2), with error logging."""
    attn_maps, scene_graph = None, None
    if not show_attention:
        return attn_maps, scene_graph

    # Attention maps: ViT-B/32 (matches the stored embeddings)
    b32_model, b32_pre = load_clip_model()
    try:
        attn_maps = compute_attention_maps(episode_id, b32_model, b32_pre)
    except Exception:
        log.exception("Failed to compute attention maps for %s", episode_id)
        st.error("Attention map computation failed — see logs/gui.log for details.")

    # Object detection: OWL-ViT v2 (true zero-shot detection with bounding boxes)
    if show_scene_graph and object_list:
        owlvit_model, owlvit_processor = load_owlvit_model()
        try:
            scene_graph = compute_scene_graph(
                episode_id, tuple(object_list), owlvit_model, owlvit_processor
            )
        except Exception:
            log.exception("Failed to compute scene graph for %s", episode_id)
            st.error("Scene graph computation failed — see logs/gui.log for details.")

    return attn_maps, scene_graph


def _render_frame_col(data: dict, idx: int, n: int, timestamps, attn_maps,
                       attn_alpha: float, scene_graph):
    """Render left column: frame image + failure badge + scene graph + metrics."""
    is_failure = bool(data["failure_labels"][idx])
    if is_failure:
        st.error(f"⚠ FAILURE — t = {timestamps[idx]:.1f}s")
    else:
        st.success(f"✓ Normal — t = {timestamps[idx]:.1f}s")

    frame = data["frames"][idx]
    display_frame = overlay_attention(frame, attn_maps[idx], alpha=attn_alpha) \
        if attn_maps is not None else frame.copy()

    # White box for the selected object (detected only)
    selected_obj = st.session_state.get("selected_object")
    if selected_obj and scene_graph is not None:
        obj = next((o for o in scene_graph if o["name"] == selected_obj), None)
        if obj and obj["detected"] and obj["box"] is not None:
            display_frame = draw_object_box(display_frame, obj["box"], color=(255, 255, 255))

    caption = f"Frame {idx}/{n-1}"
    if attn_maps is not None:
        caption += " — red: attention peak"
    if selected_obj:
        caption += f", white: {selected_obj}"
    st.image(display_frame, use_container_width=True, caption=caption)

    # Scene graph — spatial relationships between objects vs. current attention
    if scene_graph is not None and attn_maps is not None:
        fig_sg = plot_inline_scene_graph(scene_graph, attn_maps[idx])
        st.pyplot(fig_sg, use_container_width=True)
        plt.close(fig_sg)
        st.caption(
            "Nodes = OWL-ViT detected objects. Edges = spatial relationships. "
            "**Yellow border** = in top-3 attended patches. **★** = attention peak. "
            "Grayed nodes were not detected above confidence threshold."
        )

    # Object highlight selector (detected objects only)
    if scene_graph is not None:
        detected_names = [o["name"] for o in scene_graph if o["detected"]]
        current = st.session_state.get("selected_object")
        options = ["(none)"] + detected_names
        default_idx = options.index(current) if current in options else 0
        chosen = st.selectbox("Highlight object in frame", options=options,
                              index=default_idx, key="obj_selector")
        st.session_state["selected_object"] = None if chosen == "(none)" else chosen

    # Compact per-frame metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Vis. sim.", f"{data['visual_norms'][idx]:.3f}")
    c2.metric("Aud. sim.", f"{data['audio_norms'][idx]:.3f}")
    c3.metric("Vis. Δ",    f"{data['frame_deltas'][idx]:.4f}",
              help="Cosine distance from previous frame")


def _render_object_localization_section(scene_graph, attn_maps, idx: int,
                                         first_frame, actions: list[str],
                                         show_scene_graph: bool,
                                         object_list: list[str]):
    """Render full-width object localization panel below the main layout."""
    if scene_graph is not None and attn_maps is not None:
        st.divider()
        st.markdown("#### Object localization — OWL-ViT v2 zero-shot detection")
        st.caption(
            "Object positions detected by OWL-ViT v2 on the first episode frame. "
            "**Yellow border** = object center within top-3 attended patches. "
            "Objects below the confidence threshold (0.10) are not shown."
        )
        fig_sg = plot_scene_graph(scene_graph, attn_maps[idx], first_frame, actions)
        st.pyplot(fig_sg, use_container_width=True)
        plt.close(fig_sg)
    elif show_scene_graph and not object_list:
        st.info("No object list found for this episode in tasks_real_world.json — "
                "scene graph is only available for real-world episodes.")


def main():
    st.set_page_config(
        page_title="Live Embedding Visualizer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    episode_id, speed, playing, show_attention, attn_alpha, show_scene_graph = _render_sidebar()

    # ── Reset state on episode change ────────────────────────────────────────
    if st.session_state.get("loaded_episode") != episode_id:
        log.info("Episode changed → %s", episode_id)
        st.session_state["current_frame"] = 0
        st.session_state["playing"] = False
        st.session_state["selected_object"] = None
        # Clear persistent object selector widget state to avoid stale selections
        st.session_state.pop("obj_selector", None)
        st.session_state["loaded_episode"] = episode_id

    data       = load_episode(episode_id)
    n          = len(data["frames"])
    timestamps = data["timestamps"]

    task_meta   = load_task_metadata().get(episode_id)
    object_list = task_meta.get("object_list", []) if task_meta else []
    actions     = task_meta.get("actions", [])     if task_meta else []

    attn_maps, scene_graph = _load_perception_data(
        episode_id, show_attention, show_scene_graph, object_list
    )

    # ── Frame scrubber ───────────────────────────────────────────────────────
    idx = st.slider(
        "Frame", min_value=0, max_value=n - 1,
        value=st.session_state.get("current_frame", 0),
        format="%d",
    )
    if idx != st.session_state.get("current_frame", 0):
        log.debug("Frame → %d  t=%.1fs  failure=%s", idx, float(timestamps[idx]),
                  bool(data["failure_labels"][idx]))
    st.session_state["current_frame"] = idx

    # ── Main layout ──────────────────────────────────────────────────────────
    col_frame, col_plots = st.columns([2, 3], gap="medium")

    with col_frame:
        _render_frame_col(data, idx, n, timestamps, attn_maps, attn_alpha, scene_graph)

    with col_plots:
        fig_sig = plot_signals(
            timestamps, data["visual_norms"], data["audio_norms"],
            data["frame_deltas"], idx,
        )
        st.pyplot(fig_sig, use_container_width=True)
        plt.close(fig_sig)

        fig_pca = plot_pca(data["pca_coords"], timestamps, idx)
        st.pyplot(fig_pca, use_container_width=True)
        plt.close(fig_pca)

    _render_object_localization_section(
        scene_graph, attn_maps, idx,
        data["frames"][0], actions,
        show_scene_graph, object_list,
    )

    # ── Playback loop ────────────────────────────────────────────────────────
    if st.session_state.get("playing", False):
        next_idx = idx + 1
        if next_idx >= n:
            st.session_state["playing"] = False
        else:
            time.sleep(0.5 / speed)
            st.session_state["current_frame"] = next_idx
            st.rerun()
