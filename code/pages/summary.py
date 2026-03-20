"""Cross-episode summary table — Streamlit multipage app page."""
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st

# Allow imports from code/ package when run as a page
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gui.config import ALIGNED_DIR, ENCODED_DIR


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
