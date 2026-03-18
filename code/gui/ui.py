import time

import matplotlib.pyplot as plt
import streamlit as st

from .attention import compute_attention_maps, draw_object_box, overlay_attention
from .data import list_episodes, load_episode, load_task_metadata
from .localization import active_objects, compute_cur_objects
from .logger import log
from .models import load_clip_model, load_owlvit_model
from .plots import plot_inline_cur_objects, plot_pca, plot_cur_objects, plot_signals


def _render_sidebar() -> tuple[str, float, bool, float]:
    """Render sidebar controls. Returns (episode_id, speed, playing, attn_alpha)."""
    with st.sidebar:
        st.title("Controls")

        episodes = list_episodes()
        if not episodes:
            st.error("No episodes found in aligned/ and encoded/.")
            st.stop()

        episode_id = st.selectbox("Episode", episodes)
        st.divider()

        playing = st.session_state.get("playing", False)
        if st.button("⏸ Pause" if playing else "▶ Play",
                     use_container_width=True, key="play_pause_btn"):
            st.session_state["playing"] = not playing
            st.rerun()

        speed = st.select_slider(
            "Playback speed", options=[0.25, 0.5, 1.0, 2.0, 4.0],
            value=st.session_state.get("speed", 1.0), key="speed_slider",
        )
        st.session_state["speed"] = speed
        st.divider()

        attn_alpha = st.slider("Heatmap opacity", 0.1, 0.8, 0.4, 0.05,
                               key="attn_alpha")
        st.caption(
            "Red box = highest-attention patch mapped to image coords. "
            "Yellow-bordered nodes in graph = within top-3 attention patches."
        )
        st.divider()
        st.caption("Frames sampled at 2fps")
        st.caption("Embeddings: CLIP + WAV2CLIP")

    return episode_id, speed, playing, attn_alpha


def _load_perception_data(episode_id: str, object_list: list[str]):
    """Load attention maps (ViT-B/32) and scene graph (OWL-ViT v2), with error logging."""
    attn_maps, cur_objects = None, None

    # Attention maps: ViT-B/32 (matches the stored embeddings)
    b32_model, b32_pre = load_clip_model()
    try:
        attn_maps = compute_attention_maps(episode_id, b32_model, b32_pre)
    except Exception:
        log.exception("Failed to compute attention maps for %s", episode_id)
        st.error("Attention map computation failed — see logs/gui.log for details.")

    # Object detection: OWL-ViT v2 (true zero-shot detection with bounding boxes)
    if object_list:
        owlvit_model, owlvit_processor = load_owlvit_model()
        try:
            cur_objects = compute_cur_objects(
                episode_id, tuple(object_list), owlvit_model, owlvit_processor
            )
        except Exception:
            log.exception("Failed to compute scene graph for %s", episode_id)
            st.error("Scene graph computation failed — see logs/gui.log for details.")

    return attn_maps, cur_objects


def _render_frame_col(data: dict, idx: int, n: int, timestamps, attn_maps,
                       attn_alpha: float, cur_objects):
    """Render left column: frame image + failure badge + scene graph + metrics."""
    is_failure = bool(data["failure_labels"][idx])
    if is_failure:
        st.error(f"⚠ FAILURE — t = {timestamps[idx]:.1f}s")
    else:
        st.success(f"✓ Normal — t = {timestamps[idx]:.1f}s")

    # ── Failure summary (only on failure frames, only with attention loaded) ─
    if is_failure and attn_maps is not None:
        attn = attn_maps[idx]           # (7, 7)
        peak_r, peak_c = divmod(int(attn.argmax()), 7)

        # Which detected objects overlap the attention peak?
        peak_objects, top3_objects = [], []
        if cur_objects is not None:
            from .localization import _box_to_attn_patch
            flat_idx  = attn.ravel().argsort()[::-1][:3]
            top3_patches = {divmod(int(i), 7) for i in flat_idx}
            for obj in cur_objects:
                if not obj["detected"]:
                    continue
                pr, pc = _box_to_attn_patch(obj["cx_norm"], obj["cy_norm"])
                if (pr, pc) == (peak_r, peak_c):
                    peak_objects.append(obj["name"])
                if (pr, pc) in top3_patches:
                    top3_objects.append(obj["name"])

        with st.expander("Perception summary at this failure frame", expanded=True):
            c1, c2 = st.columns(2)
            c1.metric("Vis. sim.", f"{data['visual_norms'][idx]:.3f}",
                      delta=f"{data['visual_norms'][idx] - float(data['visual_norms'].mean()):.3f}",
                      help="vs. episode mean")
            c2.metric("Aud. sim.", f"{data['audio_norms'][idx]:.3f}",
                      delta=f"{data['audio_norms'][idx] - float(data['audio_norms'].mean()):.3f}",
                      help="vs. episode mean")
            st.metric("Frame Δ (cosine dist)", f"{data['frame_deltas'][idx]:.4f}",
                      help="Cosine distance from previous frame")
            st.markdown(f"**Attention peak:** patch ({peak_r}, {peak_c})")
            if peak_objects:
                st.markdown(f"**Object at attention peak:** {', '.join(peak_objects)}")
            else:
                st.markdown("**Object at attention peak:** _(none detected there)_")
            if top3_objects:
                st.markdown(f"**Objects in top-3 patches:** {', '.join(top3_objects)}")
            else:
                st.markdown("**Objects in top-3 patches:** _(none)_")

    frame = data["frames"][idx]
    display_frame = overlay_attention(frame, attn_maps[idx], alpha=attn_alpha) \
        if attn_maps is not None else frame.copy()

    # White box for the selected object (detected only)
    selected_obj = st.session_state.get("selected_object")
    if selected_obj and cur_objects is not None:
        obj = next((o for o in cur_objects if o["name"] == selected_obj), None)
        if obj and obj["detected"] and obj["box"] is not None:
            display_frame = draw_object_box(display_frame, obj["box"], color=(255, 255, 255))

    caption = f"Frame {idx}/{n-1}"
    if attn_maps is not None:
        caption += " — red: attention peak"
    if selected_obj:
        caption += f", white: {selected_obj}"
    st.image(display_frame, use_container_width=True, caption=caption)

    # Scene graph — spatial relationships between objects vs. current attention
    if cur_objects is not None and attn_maps is not None:
        fig_sg = plot_inline_cur_objects(cur_objects, attn_maps[idx])
        st.pyplot(fig_sg, use_container_width=True)
        plt.close(fig_sg)
        st.caption(
            "Nodes = OWL-ViT detected objects. Edges = spatial relationships. "
            "**Yellow border** = in top-3 attended patches. **★** = attention peak. "
            "Grayed nodes were not detected above confidence threshold."
        )

    # Object highlight selector (detected objects only)
    if cur_objects is not None:
        detected_names = [o["name"] for o in cur_objects if o["detected"]]
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


def _render_object_localization_section(cur_objects, attn_maps, idx: int,
                                         all_frames, actions: list[str],
                                         object_list: list[str]):
    """Render full-width object localization panel below the main layout."""
    if cur_objects is not None and attn_maps is not None:
        st.divider()
        st.markdown("#### Object localization — OWL-ViT v2 zero-shot detection")
        st.caption(
            "Object positions detected by OWL-ViT v2 at the most recent sampled frame. "
            "**Yellow border** = object center within top-3 attended patches. "
            "Objects below the confidence threshold (0.10) are not shown."
        )
        # Show the snapshot's own source frame alongside the boxes
        snapshot_frame = all_frames[idx]
        fig_sg = plot_cur_objects(cur_objects, attn_maps[idx], snapshot_frame, actions)
        st.pyplot(fig_sg, use_container_width=True)
        plt.close(fig_sg)
    elif attn_maps is not None and not object_list:
        st.info("No object list found for this episode in tasks_real_world.json — "
                "scene graph is only available for real-world episodes.")


def main():
    st.set_page_config(
        page_title="Live Embedding Visualizer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    episode_id, speed, playing, attn_alpha = _render_sidebar()

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

    attn_maps, cur_objects = _load_perception_data(episode_id, object_list)

    # ── Frame scrubber + failure navigation ─────────────────────────────────
    failure_indices = [i for i, f in enumerate(data["failure_labels"]) if f]

    idx = st.slider(
        "Frame", min_value=0, max_value=n - 1,
        value=st.session_state.get("current_frame", 0),
        format="%d",
    )
    if idx != st.session_state.get("current_frame", 0):
        log.debug("Frame → %d  t=%.1fs  failure=%s", idx, float(timestamps[idx]),
                  bool(data["failure_labels"][idx]))
    st.session_state["current_frame"] = idx

    if failure_indices:
        prev_fail = next((f for f in reversed(failure_indices) if f < idx), None)
        next_fail = next((f for f in failure_indices if f > idx), None)
        n_failures = len(failure_indices)
        label = (f"⚠ {n_failures} failure frame{'s' if n_failures > 1 else ''} "
                 f"— t = {', '.join(f'{timestamps[f]:.1f}s' for f in failure_indices)}")
        st.caption(label)
        col_prev, col_next = st.columns(2)
        if col_prev.button("⏮ Prev failure", disabled=prev_fail is None,
                           use_container_width=True, key="prev_fail_btn"):
            st.session_state["current_frame"] = prev_fail
            st.rerun()
        if col_next.button("Next failure ⏭", disabled=next_fail is None,
                           use_container_width=True, key="next_fail_btn"):
            st.session_state["current_frame"] = next_fail
            st.rerun()

    # ── Resolve active scene graph snapshot for current frame ────────────────
    # cur_objects is dict[frame_idx → object_list]; pick the most recent one ≤ idx
    cur_objects = active_objects(cur_objects, idx) if cur_objects else None

    # ── Main layout ──────────────────────────────────────────────────────────
    col_frame, col_plots = st.columns([2, 3], gap="medium")

    with col_frame:
        _render_frame_col(data, idx, n, timestamps, attn_maps, attn_alpha, cur_objects)

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
        cur_objects, attn_maps, idx,
        data["frames"], actions,
        object_list,
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
