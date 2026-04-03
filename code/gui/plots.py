import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

plt.switch_backend("Agg")  # avoid macOS AppKit crash in Streamlit subprocess

from .config import CURSOR, DARK_BG, GRID_COL, LINE_AUD, LINE_DELT, LINE_VIS
from .localization import _box_to_attn_patch, _jitter_positions, _rel_label


def _style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5)


def plot_signals(timestamps, visual_norms, audio_norms, frame_deltas, idx):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax1, ax2):
        _style_ax(ax)

    ax1.plot(timestamps, visual_norms, color=LINE_VIS, lw=1.5, label="Visual (CLIP)")
    ax1.plot(timestamps, audio_norms,  color=LINE_AUD, lw=1.5, label="Audio (WAV2CLIP)")
    ax1.axvline(timestamps[idx], color=CURSOR, lw=1.5, linestyle="--")
    ax1.set_ylabel("Sim. to mean", color="white", fontsize=8)
    ax1.set_title("Embedding activation over time", color="white", fontsize=9)
    ax1.legend(facecolor=DARK_BG, labelcolor="white", fontsize=7, loc="upper right", framealpha=0.6)

    ax2.fill_between(timestamps, frame_deltas, color=LINE_DELT, alpha=0.4)
    ax2.plot(timestamps, frame_deltas, color=LINE_DELT, lw=1.2)
    ax2.axvline(timestamps[idx], color=CURSOR, lw=1.5, linestyle="--")
    ax2.set_ylabel("Cosine dist", color="white", fontsize=8)
    ax2.set_xlabel("Time (s)", color="white", fontsize=8)
    ax2.set_title("Frame-to-frame visual change", color="white", fontsize=9)

    fig.tight_layout(pad=0.8)
    return fig


def plot_pca(pca_coords, timestamps, idx):
    n = len(timestamps)
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax)
    ax.set_title("PCA trajectory — visual embedding space", color="white", fontsize=9)
    ax.set_xlabel("PC 1", color="white", fontsize=7)
    ax.set_ylabel("PC 2", color="white", fontsize=7)

    colors = plt.cm.plasma(np.linspace(0, 1, n))
    for i in range(n - 1):
        alpha = 0.9 if i <= idx else 0.12
        lw    = 1.8  if i <= idx else 0.7
        ax.plot(pca_coords[i:i+2, 0], pca_coords[i:i+2, 1], color=colors[i], lw=lw, alpha=alpha)

    ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
               c=np.linspace(0, 1, n), cmap="plasma", s=12, alpha=0.25, zorder=2)
    ax.scatter(*pca_coords[0],  color="#aaaaaa", s=50, zorder=4, marker="o", label="start")
    ax.scatter(*pca_coords[-1], color="#ffffff", s=50, zorder=4, marker="s", label="end")
    ax.scatter(pca_coords[idx, 0], pca_coords[idx, 1],
               color=CURSOR, s=180, zorder=5, marker="*",
               edgecolors="white", linewidths=0.5, label=f"t={timestamps[idx]:.1f}s")

    ax.legend(facecolor=DARK_BG, labelcolor="white", fontsize=7, framealpha=0.6)
    fig.tight_layout(pad=0.8)
    return fig


def plot_audio_waveform(audio_windows: np.ndarray, audio_sr: int, idx: int) -> plt.Figure:
    """
    Waveform panel for the current frame's audio window.
    Top subplot: current-frame waveform.
    Bottom subplot: RMS energy over the full episode (one value per frame).
    """
    window = audio_windows[idx]
    n_samples = len(window)
    t_window  = np.linspace(-n_samples / (2 * audio_sr), n_samples / (2 * audio_sr), n_samples)

    # Per-frame RMS energy
    rms = np.sqrt((audio_windows ** 2).mean(axis=1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3.5),
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax1, ax2):
        _style_ax(ax)

    # Waveform
    ax1.plot(t_window, window, color=LINE_AUD, lw=0.8, alpha=0.9)
    ax1.axhline(0, color=GRID_COL, lw=0.6)
    ax1.set_ylabel("Amplitude", color="white", fontsize=8)
    ax1.set_xlabel("Time offset (s)", color="white", fontsize=8)
    ax1.set_title("Audio window — current frame", color="white", fontsize=9)

    # Episode-level RMS energy with cursor
    ax2.fill_between(range(len(rms)), rms, color=LINE_AUD, alpha=0.35)
    ax2.plot(rms, color=LINE_AUD, lw=1.2)
    ax2.axvline(idx, color=CURSOR, lw=1.5, linestyle="--")
    ax2.set_ylabel("RMS energy", color="white", fontsize=8)
    ax2.set_xlabel("Frame index", color="white", fontsize=8)
    ax2.set_title("Audio energy over episode", color="white", fontsize=9)

    fig.tight_layout(pad=0.8)
    return fig


def plot_gt_scene_graph(graphs: list[dict], idx: int) -> plt.Figure:
    """
    Render the ground-truth scene graph for a single step (frame index idx).
    Uses a fixed layout computed from all steps so positions are stable.
    Designed for the GUI left column (compact, dark theme).
    """
    from scene_graph import _compute_layout, _node_color  # type: ignore[import]

    # Colors
    C_HELD    = "#f0c040"
    C_SURFACE = "#2e2e3e"

    graph = graphs[idx]
    layout = _compute_layout(graphs)

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.08, 1.05)
    ax.axis("off")

    failure = graph["failure"]
    title_color = "#ff6666" if failure else "white"
    ax.set_title(
        f"GT scene graph — step {graph['step']}  {'⚠' if failure else ''}",
        color=title_color, fontsize=8, pad=4,
    )

    # Tier separator lines
    for y_sep in (0.35, 0.70):
        ax.axhline(y_sep, color=GRID_COL, lw=0.8, linestyle="--", zorder=0)

    # Build graph for this step
    G = nx.DiGraph()
    node_by_id = {n["id"]: n for n in graph["nodes"]}
    for n in graph["nodes"]:
        if n["id"] in layout:
            G.add_node(n["id"])
    for e in graph["edges"]:
        if e["src"] in layout and e["dst"] in layout:
            G.add_edge(e["src"], e["dst"], rel=e["rel"])

    pos = {nid: layout[nid] for nid in G.nodes if nid in layout}

    # Edges
    holds_edges = [(u, v) for u, v, d in G.edges(data=True) if d["rel"] == "holds"]
    onin_edges  = [(u, v) for u, v, d in G.edges(data=True) if d["rel"] == "on/in"]

    nx.draw_networkx_edges(G, pos, edgelist=holds_edges, ax=ax,
                           edge_color=C_HELD, style="dashed", width=1.8, alpha=0.85,
                           arrows=True, arrowsize=12,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G, pos, edgelist=onin_edges, ax=ax,
                           edge_color="#555566", style="solid", width=1.2, alpha=0.7,
                           arrows=True, arrowsize=10)

    edge_labels = {(u, v): d["rel"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=5, font_color="#888899",
                                 bbox=dict(facecolor=DARK_BG, edgecolor="none", pad=1))

    # Nodes
    for nid in G.nodes:
        n = node_by_id[nid]
        x, y = pos[nid]
        color = _node_color(n)
        ec = "#ff4444" if failure and n["tier"] == "object" else "white"
        lw = 1.8 if failure and n["tier"] == "object" else 0.6

        circle = plt.Circle((x, y), 0.055, color=color, ec=ec, lw=lw,
                             zorder=3, alpha=0.95)
        ax.add_patch(circle)

        short = n["name"] if len(n["name"]) <= 11 else n["name"][:9] + ".."
        ax.text(x, y, short, ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold", zorder=4)

        badges = []
        if n["is_picked_up"]:  badges.append("held")
        if n["is_toggled"]:    badges.append("on")
        if n["is_filled"]:     badges.append("filled")
        if n["is_broken"]:     badges.append("broken")
        if badges:
            ax.text(x, y - 0.082, " · ".join(badges), ha="center", va="top",
                    fontsize=5, color="#cccccc", zorder=4)

    # Compact legend (right margin)
    from scene_graph import C_AGENT, C_NORMAL, C_HELD as _HELD, C_TOGGLED, C_FILLED, C_BROKEN  # type: ignore[import]
    patches = [
        mpatches.Patch(color=C_AGENT,   label="Agent"),
        mpatches.Patch(color=C_NORMAL,  label="Normal"),
        mpatches.Patch(color=_HELD,     label="Held"),
        mpatches.Patch(color=C_TOGGLED, label="On"),
        mpatches.Patch(color=C_FILLED,  label="Filled"),
        mpatches.Patch(color=C_BROKEN,  label="Broken"),
        mpatches.Patch(color=C_SURFACE, label="Surface"),
    ]
    ax.legend(handles=patches, loc="lower left",
              facecolor=DARK_BG, labelcolor="white",
              fontsize=5.5, framealpha=0.85)

    ax.set_xlabel(f"Action: {graph['last_action']}", color="#888888", fontsize=6)
    fig.tight_layout(pad=0.4)
    return fig


def plot_inline_scene_graph(object_locs: list[dict], attn_7x7: np.ndarray) -> plt.Figure:
    """
    Compact scene graph for the left column.
    Objects are nodes at their detected positions (normalized 0–1 coords).
    Edges show pairwise spatial relationships between detected objects.
    Undetected objects appear grayed-out below the main plot area.
    Yellow border = object is in a top-3 attended patch.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.25, -0.05)   # row 0 (top of image) at top of plot
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Scene graph — spatial relationships", color="white", fontsize=8, pad=4)

    # Top-3 attention patches in 7×7 grid
    flat_idx = np.argsort(attn_7x7.ravel())[::-1][:3]
    top_patches = {divmod(int(i), 7) for i in flat_idx}

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(object_locs), 1)))
    positions = _jitter_positions(object_locs)

    # Edges — only between detected objects
    for i in range(len(object_locs)):
        for j in range(i + 1, len(object_locs)):
            a, b = object_locs[i], object_locs[j]
            if not a["detected"] or not b["detected"]:
                continue
            cx_a, cy_a = positions[i]
            cx_b, cy_b = positions[j]
            label = _rel_label(a, b)
            ax.plot([cx_a, cx_b], [cy_a, cy_b], color="#444455", lw=1.0, zorder=1)
            ax.text((cx_a + cx_b) / 2, (cy_a + cy_b) / 2, label,
                    ha="center", va="center", fontsize=4.5, color="#aaaaaa", zorder=2,
                    bbox=dict(facecolor=DARK_BG, edgecolor="none", pad=1))

    # Nodes
    for i, obj in enumerate(object_locs):
        cx, cy = positions[i]
        attended = False
        if obj["detected"]:
            pr, pc = _box_to_attn_patch(obj["cx_norm"], obj["cy_norm"])
            attended = (pr, pc) in top_patches

        node_color = colors[i] if obj["detected"] else "#444444"
        text_color = "white"    if obj["detected"] else "#777777"
        ec  = CURSOR if attended else ("white" if obj["detected"] else "#555555")
        lw  = 2.5    if attended else 1.0

        circle = plt.Circle((cx, cy), 0.04, color=node_color, zorder=3,
                             ec=ec, lw=lw, alpha=1.0 if obj["detected"] else 0.5)
        ax.add_patch(circle)

        short_name = " ".join(obj["name"].split()[:2])
        label_text = short_name if obj["detected"] else f"? {short_name}"
        ax.text(cx, cy, label_text,
                ha="center", va="center", fontsize=4.5,
                color=text_color, fontweight="bold" if obj["detected"] else "normal",
                zorder=4)

        if obj["detected"]:
            ax.text(cx, cy + 0.055, f"{obj['score']:.2f}",
                    ha="center", va="bottom", fontsize=4, color="#aaaaaa", zorder=4)

    # Attention peak marker
    pr, pc = np.unravel_index(attn_7x7.argmax(), (7, 7))
    # Convert 7×7 patch coords to normalized image coords (center of patch)
    ax.scatter((pc + 0.5) / 7, (pr + 0.5) / 7,
               marker="*", s=100, color=CURSOR, zorder=5,
               edgecolors="white", linewidths=0.4)

    fig.tight_layout(pad=0.3)
    return fig


def plot_scene_graph(object_locs: list[dict], attn_7x7: np.ndarray,
                     frame: np.ndarray, actions: list[str]) -> plt.Figure:
    """
    Full-width object localization panel.
    Shows object bounding boxes on the frame where the active snapshot was captured.

    Layout: left = frame with boxes | right = planned action list.
    """
    fig, (ax_frame, ax_actions) = plt.subplots(
        1, 2, figsize=(10, 4),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.patch.set_facecolor(DARK_BG)

    # ── Top-3 attention patches ──────────────────────────────────────────────
    flat_idx    = np.argsort(attn_7x7.ravel())[::-1][:3]
    top_patches = {divmod(int(i), 7) for i in flat_idx}

    colors    = plt.cm.Set2(np.linspace(0, 1, max(len(object_locs), 1)))
    obj_index = {o["name"]: i for i, o in enumerate(object_locs)}

    # ── Left: frame + bounding boxes ────────────────────────────────────────
    ax_frame.set_facecolor(DARK_BG)
    ax_frame.axis("off")
    ax_frame.set_title("OWL-ViT v2 — object localization", color="white", fontsize=9)
    ax_frame.imshow(frame)
    h, w = frame.shape[:2]

    not_detected_count = 0
    for obj in object_locs:
        i         = obj_index[obj["name"]]
        color_rgb = colors[i][:3]
        if obj["detected"] and obj["box"] is not None:
            x1, y1, x2, y2 = obj["box"]
            pr, pc     = _box_to_attn_patch(obj["cx_norm"], obj["cy_norm"])
            attended   = (pr, pc) in top_patches
            edge_color = CURSOR if attended else color_rgb
            lw         = 2.5    if attended else 1.5
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=edge_color, linewidth=lw, zorder=3,
            )
            ax_frame.add_patch(rect)
            short = " ".join(obj["name"].split()[:2])
            ax_frame.text(
                x1, max(y1 - 3, 0), f"{short} ({obj['score']:.2f})",
                color="white", fontsize=6, va="bottom", zorder=4,
                bbox=dict(facecolor=color_rgb, alpha=0.75, pad=1, edgecolor="none"),
            )
        else:
            ax_frame.text(
                8, h - 10 - not_detected_count * 14,
                f"✗ {obj['name']} (not detected)",
                color="#888888", fontsize=6, va="bottom",
            )
            not_detected_count += 1

    # ── Right: action list ────────────────────────────────────────────────
    ax_actions.set_facecolor(DARK_BG)
    ax_actions.axis("off")
    ax_actions.set_title("Planned actions", color="white", fontsize=8)

    skip_labels = {"Skip", "Ignore", "Terminate"}
    visible = [a for a in actions if a not in skip_labels]
    for j, action in enumerate(visible):
        ax_actions.text(
            0.05, 1.0 - j * (1.0 / max(len(visible), 1)) - 0.05,
            f"{j+1}. {action}",
            transform=ax_actions.transAxes,
            color="white", fontsize=7, va="top",
            wrap=True,
        )

    fig.tight_layout(pad=0.6)
    return fig
