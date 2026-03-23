"""Cross-episode summary table — Streamlit multipage app page."""
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Allow imports from code/ package when run as a page
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gui.config import ALIGNED_DIR, ENCODED_DIR, OWL_DIR, TASKS_JSON

DARK_BG  = "#0e1117"
GRID_COL = "#1e2130"
C_VIS    = "#4c9be8"   # blue  — visual sim
C_AUD    = "#7ec87e"   # green — audio sim
C_DELT   = "#e8754c"   # orange — frame delta
C_FAIL   = "#f0c040"   # yellow — failure marker
C_OWL    = "#cc77dd"   # purple — OWL-ViT score
LEADUP_WINDOW    = 5    # frames before F to examine
OWL_THRESHOLD    = 0.10


def _style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5)


def _norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


@st.cache_data
def _task_meta() -> dict[str, dict]:
    if not os.path.exists(TASKS_JSON):
        return {}
    with open(TASKS_JSON) as f:
        raw = json.load(f)
    return {v["general_folder_name"]: v for v in raw.values()}


def _load_owl_scores(episode_id: str) -> tuple[dict[int, np.ndarray] | None, list[str]]:
    """
    Load owl/<episode_id>.npz and return a frame-indexed dict of score rows.
    Returns ({frame_idx: scores_row (n_obj,)} or None, object_names list).
    """
    path = os.path.join(OWL_DIR, f"{episode_id}.npz")
    if not os.path.exists(path):
        return None, []
    obj_list = _task_meta().get(episode_id, {}).get("object_list", [])
    if not obj_list:
        return None, []
    d              = np.load(path, allow_pickle=False)
    sample_indices = d["sample_indices"].tolist()
    scores         = d["scores"].astype(np.float32)
    return {int(fi): scores[row] for row, fi in enumerate(sample_indices)}, obj_list


@st.cache_data(show_spinner="Computing episode summary...")
def _compute_summary() -> pd.DataFrame:
    aligned_eps = {f[:-4] for f in os.listdir(ALIGNED_DIR) if f.endswith(".npz")}
    encoded_eps = {f[:-4] for f in os.listdir(ENCODED_DIR) if f.endswith(".npz")}
    episodes = sorted(aligned_eps & encoded_eps)

    rows = []
    for ep in episodes:
        try:
            enc = np.load(os.path.join(ENCODED_DIR, f"{ep}.npz"), allow_pickle=False)
            vis = enc["visual_embeddings"].astype(np.float32)
            aud = enc["audio_embeddings"].astype(np.float32)
            labels = enc["failure_labels"]

            mean_vis = vis.mean(axis=0)
            mean_vis /= np.linalg.norm(mean_vis) + 1e-8
            vis_sim = (vis @ mean_vis).astype(np.float32)

            mean_aud = aud.mean(axis=0)
            mean_aud /= np.linalg.norm(mean_aud) + 1e-8
            aud_sim = (aud @ mean_aud).astype(np.float32)

            fail_mask   = labels.astype(bool)
            normal_mask = ~fail_mask
            n_failures  = int(fail_mask.sum())

            vis_fail   = float(vis_sim[fail_mask].mean())   if n_failures > 0 else float("nan")
            vis_normal = float(vis_sim[normal_mask].mean()) if normal_mask.any() else float("nan")
            vis_delta  = vis_fail - vis_normal               if n_failures > 0 else float("nan")

            aud_fail   = float(aud_sim[fail_mask].mean())   if n_failures > 0 else float("nan")
            aud_normal = float(aud_sim[normal_mask].mean()) if normal_mask.any() else float("nan")
            aud_delta  = aud_fail - aud_normal               if n_failures > 0 else float("nan")

            # Frame-to-frame visual change at failure
            n = len(vis)
            frame_deltas = np.zeros(n, dtype=np.float32)
            for i in range(1, n):
                frame_deltas[i] = 1.0 - float(np.dot(vis[i], vis[i - 1]))
            vis_delta_at_fail = float(frame_deltas[fail_mask].mean()) if n_failures > 0 else float("nan")

            rows.append({
                "episode":            ep,
                "n_frames":           n,
                "n_failure_frames":   n_failures,
                "vis_sim_normal":     round(vis_normal, 4),
                "vis_sim_failure":    round(vis_fail,   4),
                "vis_sim_delta":      round(vis_delta,  4),
                "aud_sim_normal":     round(aud_normal, 4),
                "aud_sim_failure":    round(aud_fail,   4),
                "aud_sim_delta":      round(aud_delta,  4),
                "mean_frame_delta_at_failure": round(vis_delta_at_fail, 4),
            })
        except Exception as exc:
            rows.append({"episode": ep, "error": str(exc)})

    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Computing lead-up windows...")
def _compute_leadup(window: int = LEADUP_WINDOW) -> list[dict]:
    """
    For each episode, find failure frame groups (consecutive runs).
    For each group, return signals over [F-window … F+1] where F = first frame
    of the run. Returns one dict per failure event.
    """
    aligned_eps = {f[:-4] for f in os.listdir(ALIGNED_DIR) if f.endswith(".npz")}
    encoded_eps = {f[:-4] for f in os.listdir(ENCODED_DIR) if f.endswith(".npz")}
    episodes = sorted(aligned_eps & encoded_eps)

    events = []
    for ep in episodes:
        try:
            enc = np.load(os.path.join(ENCODED_DIR, f"{ep}.npz"), allow_pickle=False)
            vis    = enc["visual_embeddings"].astype(np.float32)
            aud    = enc["audio_embeddings"].astype(np.float32)
            labels = enc["failure_labels"].astype(bool)
            ts     = enc["timestamps"].astype(np.float64)
            n      = len(vis)

            mean_vis = vis.mean(axis=0); mean_vis /= np.linalg.norm(mean_vis) + 1e-8
            vis_sim  = (vis @ mean_vis).astype(np.float32)

            mean_aud = aud.mean(axis=0); mean_aud /= np.linalg.norm(mean_aud) + 1e-8
            aud_sim  = (aud @ mean_aud).astype(np.float32)

            frame_delta = np.zeros(n, dtype=np.float32)
            for i in range(1, n):
                frame_delta[i] = 1.0 - float(np.dot(vis[i], vis[i - 1]))

            # Group consecutive failure frames; take first of each run
            fail_indices = np.where(labels)[0]
            if len(fail_indices) == 0:
                continue
            group_starts = [fail_indices[0]]
            for prev, cur in zip(fail_indices, fail_indices[1:]):
                if cur - prev > 1:
                    group_starts.append(cur)

            owl_scores, owl_names = _load_owl_scores(ep)

            for F in group_starts:
                start = max(0, F - window)
                sl    = slice(start, min(n, F + 1))

                pad        = window - (F - start)
                rel_full   = np.arange(-window, 1)
                vs_full    = np.full(window + 1, np.nan, dtype=np.float32)
                ad_full    = np.full(window + 1, np.nan, dtype=np.float32)
                fd_full    = np.full(window + 1, np.nan, dtype=np.float32)
                vs_full[pad:] = vis_sim[sl]
                ad_full[pad:] = aud_sim[sl]
                fd_full[pad:] = frame_delta[sl]

                # Compute drop onset: first frame where vis_sim < mean_normal - 1σ
                normal_mask = ~labels
                if normal_mask.any():
                    mu        = float(vis_sim[normal_mask].mean())
                    sigma     = float(vis_sim[normal_mask].std())
                    vis_thr   = mu - sigma
                    drop_onset = None
                    for offset_i, v in zip(rel_full, vs_full):
                        if not np.isnan(v) and v < vis_thr:
                            drop_onset = int(offset_i)
                            break
                else:
                    vis_thr    = float("nan")
                    drop_onset = None

                # Frame delta spike: argmax in window
                fd_valid     = np.where(~np.isnan(fd_full), fd_full, -np.inf)
                spike_offset = int(rel_full[np.argmax(fd_valid)])

                # OWL-ViT per-object scores in window (if pre-computed)
                owl_window: dict[str, np.ndarray] = {}
                owl_lost: dict[str, int | None] = {}   # offset where object dropped below threshold
                if owl_scores is not None:
                    for oi, oname in enumerate(owl_names):
                        raw = np.full(window + 1, np.nan, dtype=np.float32)
                        for slot, abs_frame in enumerate(range(start, min(n, F + 1))):
                            if abs_frame in owl_scores:
                                raw[pad + slot] = owl_scores[abs_frame][oi]
                        owl_window[oname] = raw
                        # first frame (in window) where score drops below threshold
                        lost = None
                        above_seen = False
                        for off, s in zip(rel_full, raw):
                            if np.isnan(s):
                                continue
                            if s >= OWL_THRESHOLD:
                                above_seen = True
                            elif above_seen and lost is None:
                                lost = int(off)
                        owl_lost[oname] = lost

                events.append({
                    "episode":       ep,
                    "failure_frame": int(F),
                    "failure_time":  float(ts[F]),
                    "rel":           rel_full,
                    "vis_sim":       vs_full,
                    "aud_sim":       ad_full,
                    "frame_delta":   fd_full,
                    "vis_threshold": vis_thr,
                    "drop_onset":    drop_onset,
                    "spike_offset":  spike_offset,
                    "owl_window":    owl_window,
                    "owl_lost":      owl_lost,
                })
        except Exception:
            pass

    return events


def _plot_leadup(events: list[dict]) -> plt.Figure:
    """Small-multiples figure: one column per failure event."""
    n_events = len(events)
    if n_events == 0:
        return None

    ncols = min(n_events, 4)
    nrows = (n_events + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 3.0),
        squeeze=False,
    )
    fig.patch.set_facecolor(DARK_BG)

    for ax_flat_i, event in enumerate(events):
        row, col = divmod(ax_flat_i, ncols)
        ax = axes[row][col]
        _style_ax(ax)

        rel  = event["rel"].astype(float)
        vs   = _norm01(np.nan_to_num(event["vis_sim"],   nan=0.5))
        ad   = _norm01(np.nan_to_num(event["aud_sim"],   nan=0.5))
        fd   = _norm01(np.nan_to_num(event["frame_delta"], nan=0.0))

        ax.plot(rel, vs, color=C_VIS,  lw=1.5, label="Vis sim",  zorder=3)
        ax.plot(rel, ad, color=C_AUD,  lw=1.5, label="Aud sim",  zorder=3)
        ax.fill_between(rel, fd, color=C_DELT, alpha=0.35,        zorder=2)
        ax.plot(rel, fd, color=C_DELT, lw=1.2, label="Frame Δ",  zorder=3)

        # OWL-ViT per-object scores (normalised, dashed)
        owl_colors = plt.cm.cool(np.linspace(0.2, 0.8, max(len(event["owl_window"]), 1)))
        for oi, (oname, raw) in enumerate(event["owl_window"].items()):
            owl_norm = _norm01(np.nan_to_num(raw, nan=0.0))
            short    = " ".join(oname.split()[:2])
            ax.plot(rel, owl_norm, color=owl_colors[oi], lw=0.9,
                    linestyle="--", alpha=0.8, label=f"OWL:{short}", zorder=3)
            lost = event["owl_lost"].get(oname)
            if lost is not None:
                ax.axvline(lost, color=owl_colors[oi], lw=0.8,
                           linestyle=":", alpha=0.6, zorder=4)
                ax.text(lost, 0.55 + oi * 0.1, f"✗{lost:+d}",
                        color=owl_colors[oi], fontsize=4.5,
                        va="bottom", ha="center", zorder=5)

        # Failure marker at offset 0
        ax.axvline(0, color=C_FAIL, lw=1.5, linestyle="--", zorder=4)

        # Vis sim drop onset
        do = event["drop_onset"]
        if do is not None and do < 0:
            ax.axvline(do, color=C_VIS, lw=1.0, linestyle=":", alpha=0.7, zorder=4)
            ax.text(do, 0.97, f"↓{do}", color=C_VIS,
                    fontsize=5.5, va="top", ha="center", zorder=5)

        # Frame delta spike
        so = event["spike_offset"]
        ax.text(so, 0.03, f"Δmax@{so:+d}", color=C_DELT,
                fontsize=5.5, va="bottom", ha="center", zorder=5)

        ep_short = event["episode"].replace("putFruitsBowl", "pFB")
        ax.set_title(
            f"{ep_short}\nF={event['failure_frame']} (t={event['failure_time']:.1f}s)",
            color="white", fontsize=6.5, pad=3,
        )
        ax.set_xlabel("Frames before failure", color="white", fontsize=6)
        ax.set_xlim(-LEADUP_WINDOW - 0.3, 0.5)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xticks(range(-LEADUP_WINDOW, 1))

        if col == 0:
            ax.set_ylabel("Normalized signal", color="white", fontsize=6)

        if ax_flat_i == 0:
            ax.legend(facecolor=DARK_BG, labelcolor="white", fontsize=5.5,
                      framealpha=0.7, loc="lower left")

    # Hide unused subplots
    for ax_flat_i in range(n_events, nrows * ncols):
        row, col = divmod(ax_flat_i, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Failure lead-up [F−5 … F]  |  blue=vis sim  green=aud sim  orange=frame Δ  "
        "purple dashed=OWL score (per object)  ✗=object lost  yellow=failure",
        color="white", fontsize=7.5, y=1.01,
    )
    fig.tight_layout(pad=0.6)
    return fig


st.set_page_config(page_title="Episode Summary", layout="wide")
st.title("Cross-episode summary")
st.caption(
    "All encoded episodes. Similarity = cosine similarity to episode mean embedding. "
    "Delta = failure mean − normal mean (negative = lower similarity at failure = more anomalous)."
)

df = _compute_summary()

# Highlight anomalous rows (vis_sim_delta < 0 means embedding drifted at failure)
if "vis_sim_delta" in df.columns:
    def _style(row):
        if row.get("n_failure_frames", 0) == 0:
            return ["color: #888888"] * len(row)
        if row.get("vis_sim_delta", 0) < -0.01:
            return ["background-color: #3a1a1a"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.dataframe(df, use_container_width=True, hide_index=True)

# Download as CSV
csv = df.to_csv(index=False).encode()
st.download_button(
    "⬇ Download summary CSV",
    data=csv,
    file_name="episode_summary.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Distribution plots")
if "vis_sim_delta" in df.columns and df["n_failure_frames"].sum() > 0:
    import matplotlib.pyplot as plt

    has_fail = df["n_failure_frames"] > 0
    df_fail = df[has_fail].dropna(subset=["vis_sim_delta", "aud_sim_delta"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2130")
        ax.grid(True, color="#1e2130", linewidth=0.5)

    axes[0].bar(df_fail["episode"], df_fail["vis_sim_delta"], color="#4c9be8", alpha=0.85)
    axes[0].axhline(0, color="#f0c040", lw=1.0, linestyle="--")
    axes[0].set_title("Visual sim Δ at failure (per episode)", color="white", fontsize=9)
    axes[0].set_xlabel("Episode", color="white", fontsize=7)
    axes[0].set_ylabel("Δ (failure − normal)", color="white", fontsize=7)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=5)

    axes[1].bar(df_fail["episode"], df_fail["aud_sim_delta"], color="#7ec87e", alpha=0.85)
    axes[1].axhline(0, color="#f0c040", lw=1.0, linestyle="--")
    axes[1].set_title("Audio sim Δ at failure (per episode)", color="white", fontsize=9)
    axes[1].set_xlabel("Episode", color="white", fontsize=7)
    axes[1].set_ylabel("Δ (failure − normal)", color="white", fontsize=7)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=5)

    fig.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)

    # Export
    buf = __import__("io").BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.download_button(
        "⬇ Export distribution plots",
        data=buf.getvalue(),
        file_name="summary_distributions.png",
        mime="image/png",
        use_container_width=True,
    )
    plt.close(fig)

st.divider()
st.subheader("Failure lead-up analysis")
st.caption(
    "For each failure event F, signals are shown over the [F−5 … F] window and normalized "
    "to [0, 1] per panel so shape is comparable regardless of episode scale. "
    "**Blue dotted** = first frame where visual sim drops below (normal mean − 1σ). "
    "Orange label = frame where frame-to-frame Δ peaks. "
    "Purple dashed lines = OWL-ViT confidence per object (requires `just owl` pre-computation). "
    "**✗** annotation = frame where object drops below detection threshold (perception lost it)."
)

events = _compute_leadup()
if events:
    # Episode filter
    ep_names = sorted({e["episode"] for e in events})
    selected = st.multiselect(
        "Filter episodes (empty = show all)", ep_names, default=[], key="leadup_filter"
    )
    visible = [e for e in events if not selected or e["episode"] in selected]

    if visible:
        fig_lu = _plot_leadup(visible)
        if fig_lu is not None:
            st.pyplot(fig_lu, use_container_width=True)
            buf = io.BytesIO()
            fig_lu.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                           facecolor=fig_lu.get_facecolor())
            buf.seek(0)
            st.download_button(
                "⬇ Export lead-up figure",
                data=buf.getvalue(),
                file_name="failure_leadup.png",
                mime="image/png",
                use_container_width=True,
            )
            plt.close(fig_lu)
    else:
        st.info("No failure events match the current filter.")
else:
    st.info("No failure events found across encoded episodes.")
