"""
Analysis module — REFLECT / RoboFail dataset.

Reads encoded .npz files (from encode.py) and produces:

  1. UMAP visualization — all embeddings pooled across episodes
     - Colored by failure label (normal vs failure)
     - Colored by modality (visual vs audio)
     - Colored by task/episode
  2. Temporal trajectory — per-episode PCA state evolution over time
  3. Cross-modal similarity — cosine similarity between visual and audio at each timestep
  4. Normal vs failure separation — silhouette score + cosine distance distributions
  5. Fusion comparison — early fusion (concat) vs late fusion (per-modality PCA)

All outputs saved to analysis/ as PNG figures + a JSON metrics summary.

Usage:
    python analyze.py encoded/
"""

import json
import os
import sys
import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

ENCODED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "encoded"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))

PALETTE_NORMAL = "#4C8BE0"
PALETTE_FAILURE = "#E05C4C"
PALETTE_VISUAL = "#4CB8E0"
PALETTE_AUDIO = "#E0A84C"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_encoded(encoded_dir: str) -> list[dict]:
    """Load all encoded .npz files. Returns list of episode dicts."""
    episodes = []
    files = sorted([f for f in os.listdir(encoded_dir) if f.endswith(".npz")])
    for fname in files:
        path = os.path.join(encoded_dir, fname)
        d = np.load(path, allow_pickle=True)
        episodes.append({
            "id": fname.replace(".npz", ""),
            "task": fname.split("-")[0].replace(".npz", ""),
            "visual": d["visual_embeddings"].astype(np.float32),
            "audio": d["audio_embeddings"].astype(np.float32),
            "timestamps": d["timestamps"].astype(np.float64),
            "failure_labels": d["failure_labels"].astype(bool),
            "fps_base": float(d["fps_base"]),
        })
    return episodes


# ---------------------------------------------------------------------------
# 1. UMAP visualization
# ---------------------------------------------------------------------------

def run_umap(embeddings: np.ndarray, n_components: int = 2,
             n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    import umap
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeddings)


def plot_umap_failure(episodes: list[dict], output_dir: str):
    """UMAP of all visual embeddings, colored by normal/failure."""
    all_visual = np.concatenate([ep["visual"] for ep in episodes], axis=0)
    all_failure = np.concatenate([ep["failure_labels"] for ep in episodes])

    print("  Running UMAP on visual embeddings...")
    umap_2d = run_umap(all_visual)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(umap_2d[~all_failure, 0], umap_2d[~all_failure, 1],
               c=PALETTE_NORMAL, s=12, alpha=0.5, label="Normal")
    ax.scatter(umap_2d[all_failure, 0], umap_2d[all_failure, 1],
               c=PALETTE_FAILURE, s=50, alpha=0.9, zorder=5, label="Failure")
    ax.set_title("UMAP — Visual embeddings (CLIP)\nNormal vs Failure frames", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    out = os.path.join(output_dir, "umap_visual_failure.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")
    return umap_2d, all_failure


def plot_umap_modality(episodes: list[dict], output_dir: str):
    """UMAP of visual + audio embeddings pooled, colored by modality."""
    all_visual = np.concatenate([ep["visual"] for ep in episodes], axis=0)
    all_audio = np.concatenate([ep["audio"] for ep in episodes], axis=0)
    all_emb = np.concatenate([all_visual, all_audio], axis=0)
    modality = np.array(["visual"] * len(all_visual) + ["audio"] * len(all_audio))
    failure = np.concatenate([
        np.concatenate([ep["failure_labels"] for ep in episodes]),
        np.concatenate([ep["failure_labels"] for ep in episodes]),
    ])

    print("  Running UMAP on visual + audio embeddings combined...")
    umap_2d = run_umap(all_emb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: by modality
    ax = axes[0]
    vis_mask = modality == "visual"
    ax.scatter(umap_2d[vis_mask, 0], umap_2d[vis_mask, 1],
               c=PALETTE_VISUAL, s=8, alpha=0.4, label="Visual (CLIP)")
    ax.scatter(umap_2d[~vis_mask, 0], umap_2d[~vis_mask, 1],
               c=PALETTE_AUDIO, s=8, alpha=0.4, label="Audio (WAV2CLIP)")
    ax.set_title("UMAP — By modality", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # Right: by failure
    ax = axes[1]
    ax.scatter(umap_2d[~failure, 0], umap_2d[~failure, 1],
               c=PALETTE_NORMAL, s=8, alpha=0.4, label="Normal")
    ax.scatter(umap_2d[failure, 0], umap_2d[failure, 1],
               c=PALETTE_FAILURE, s=40, alpha=0.9, zorder=5, label="Failure")
    ax.set_title("UMAP — By failure label", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    fig.suptitle("UMAP — Visual + Audio in shared 512-dim space", fontsize=13)
    plt.tight_layout()
    out = os.path.join(output_dir, "umap_modality_failure.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


def plot_umap_task(episodes: list[dict], output_dir: str):
    """UMAP of visual embeddings colored by task/episode."""
    all_visual = np.concatenate([ep["visual"] for ep in episodes], axis=0)
    all_tasks = np.concatenate([
        np.full(len(ep["visual"]), ep["task"]) for ep in episodes
    ])

    print("  Running UMAP for task coloring...")
    umap_2d = run_umap(all_visual)

    unique_tasks = sorted(set(all_tasks))
    colors = cm.tab10(np.linspace(0, 1, len(unique_tasks)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for task, color in zip(unique_tasks, colors):
        mask = all_tasks == task
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                   c=[color], s=12, alpha=0.6, label=task)
    ax.set_title("UMAP — Visual embeddings by task", fontsize=12)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    out = os.path.join(output_dir, "umap_visual_task.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 2. Temporal trajectory (PCA per episode)
# ---------------------------------------------------------------------------

def plot_temporal_trajectory(episodes: list[dict], output_dir: str,
                               n_episodes_to_plot: int = 6):
    """
    For each episode: run PCA on visual embeddings → 2D trajectory over time.
    Color by time, mark failure frames with X.
    """
    eps_to_plot = episodes[:n_episodes_to_plot]
    ncols = 3
    nrows = (len(eps_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, ep in enumerate(eps_to_plot):
        ax = axes[i]
        emb = ep["visual"]
        ts = ep["timestamps"]
        labels = ep["failure_labels"]

        pca = PCA(n_components=2)
        pca_2d = pca.fit_transform(emb)

        # Color by time
        n = len(ts)
        colors = cm.viridis(np.linspace(0, 1, n))

        for j in range(n - 1):
            ax.plot(pca_2d[j:j+2, 0], pca_2d[j:j+2, 1],
                    color=colors[j], alpha=0.7, lw=1.5)
            ax.scatter(pca_2d[j, 0], pca_2d[j, 1],
                       c=[colors[j]], s=15, zorder=3)

        # Mark start and end
        ax.scatter(*pca_2d[0], marker="s", c="green", s=60, zorder=5, label="Start")
        ax.scatter(*pca_2d[-1], marker="^", c="purple", s=60, zorder=5, label="End")

        # Mark failure frames
        fail_idxs = np.where(labels)[0]
        if len(fail_idxs) > 0:
            ax.scatter(pca_2d[fail_idxs, 0], pca_2d[fail_idxs, 1],
                       marker="X", c=PALETTE_FAILURE, s=150, zorder=6,
                       edgecolors="black", lw=0.5, label="Failure")

        var_explained = pca.explained_variance_ratio_[:2].sum() * 100
        ax.set_title(f"{ep['id']}\nPCA var: {var_explained:.1f}%", fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes
    for j in range(len(eps_to_plot), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Temporal trajectory — Visual embeddings via PCA\n"
                 "(color = time: dark=early, bright=late; X = failure frame)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "temporal_trajectory_pca.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 3. Cross-modal similarity (visual vs audio cosine similarity over time)
# ---------------------------------------------------------------------------

def cosine_similarity_per_frame(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Cosine similarity between visual and audio embeddings per timestep. Both L2-normalized."""
    return (visual * audio).sum(axis=-1)  # dot product of unit vectors = cosine sim


def plot_cross_modal_similarity(episodes: list[dict], output_dir: str):
    """
    For each episode: plot cosine similarity between visual and audio over time.
    Mark failure frames. Show whether similarity drops at failure.
    """
    n = len(episodes)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    all_sims_normal = []
    all_sims_failure = []

    for i, ep in enumerate(episodes):
        ax = axes[i]
        sim = cosine_similarity_per_frame(ep["visual"], ep["audio"])
        ts = ep["timestamps"]
        labels = ep["failure_labels"]

        # Smooth for readability
        sim_smooth = uniform_filter1d(sim, size=3)

        ax.plot(ts, sim_smooth, color=PALETTE_VISUAL, lw=1.5, label="cos sim (smoothed)")
        ax.plot(ts, sim, color=PALETTE_VISUAL, lw=0.5, alpha=0.4)
        ax.axhline(sim.mean(), color="gray", lw=0.8, linestyle="--",
                   label=f"mean={sim.mean():.3f}")

        fail_idxs = np.where(labels)[0]
        if len(fail_idxs) > 0:
            ax.scatter(ts[fail_idxs], sim[fail_idxs],
                       c=PALETTE_FAILURE, s=80, zorder=5, label="Failure", marker="X")

        ax.set_title(ep["id"], fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cosine sim")
        ax.set_ylim(-0.1, 0.6)
        if i == 0:
            ax.legend(fontsize=7)

        all_sims_normal.extend(sim[~labels].tolist())
        all_sims_failure.extend(sim[labels].tolist())

    for j in range(len(episodes), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Cross-modal cosine similarity: visual ↔ audio per frame", fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "crossmodal_similarity.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")

    return all_sims_normal, all_sims_failure


# ---------------------------------------------------------------------------
# 4. Normal vs failure separation metrics
# ---------------------------------------------------------------------------

def compute_separation_metrics(episodes: list[dict]) -> dict:
    """
    Compute quantitative metrics for normal vs failure separability.
    Returns dict of metrics.
    """
    vis_normal, vis_failure = [], []
    aud_normal, aud_failure = [], []

    for ep in episodes:
        labels = ep["failure_labels"]
        vis_normal.append(ep["visual"][~labels])
        vis_failure.append(ep["visual"][labels])
        aud_normal.append(ep["audio"][~labels])
        aud_failure.append(ep["audio"][labels])

    vis_normal = np.concatenate(vis_normal)
    vis_failure = np.concatenate(vis_failure)
    aud_normal = np.concatenate(aud_normal)
    aud_failure = np.concatenate(aud_failure)

    metrics = {}

    # Mean cosine distance between failure centroid and normal centroid
    def mean_cos_dist(a, b):
        ca = a.mean(axis=0); ca /= np.linalg.norm(ca)
        cb = b.mean(axis=0); cb /= np.linalg.norm(cb)
        return float(1 - np.dot(ca, cb))

    metrics["visual_centroid_cos_dist"] = mean_cos_dist(vis_normal, vis_failure)
    metrics["audio_centroid_cos_dist"] = mean_cos_dist(aud_normal, aud_failure)

    # Silhouette score (requires at least 2 samples per class)
    if len(vis_failure) >= 2:
        n_sub = min(500, len(vis_normal))
        rng = np.random.default_rng(42)
        idx_n = rng.choice(len(vis_normal), n_sub, replace=False)
        X_vis = np.concatenate([vis_normal[idx_n], vis_failure])
        y_vis = np.array([0] * n_sub + [1] * len(vis_failure))
        metrics["visual_silhouette"] = float(silhouette_score(X_vis, y_vis, metric="cosine"))

        X_aud = np.concatenate([aud_normal[idx_n], aud_failure])
        metrics["audio_silhouette"] = float(silhouette_score(X_aud, y_vis, metric="cosine"))
    else:
        metrics["visual_silhouette"] = None
        metrics["audio_silhouette"] = None

    metrics["n_normal_frames"] = int(len(vis_normal))
    metrics["n_failure_frames"] = int(len(vis_failure))

    return metrics


def plot_separation_distributions(episodes: list[dict], output_dir: str) -> dict:
    """
    Plot cosine similarity distributions: normal-normal vs normal-failure.
    """
    metrics = compute_separation_metrics(episodes)

    # Cosine distance from each frame to the normal centroid
    all_vis = np.concatenate([ep["visual"] for ep in episodes])
    all_aud = np.concatenate([ep["audio"] for ep in episodes])
    all_labels = np.concatenate([ep["failure_labels"] for ep in episodes])

    vis_centroid = all_vis[~all_labels].mean(axis=0)
    vis_centroid /= np.linalg.norm(vis_centroid)
    aud_centroid = all_aud[~all_labels].mean(axis=0)
    aud_centroid /= np.linalg.norm(aud_centroid)

    vis_sim = (all_vis * vis_centroid).sum(axis=-1)
    aud_sim = (all_aud * aud_centroid).sum(axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, sim, label, title in [
        (axes[0], vis_sim, "Visual (CLIP)", "Cosine similarity to normal centroid"),
        (axes[1], aud_sim, "Audio (WAV2CLIP)", "Cosine similarity to normal centroid"),
    ]:
        bins = np.linspace(sim.min(), sim.max(), 40)
        ax.hist(sim[~all_labels], bins=bins, alpha=0.6, color=PALETTE_NORMAL,
                label=f"Normal (n={np.sum(~all_labels)})", density=True)
        ax.hist(sim[all_labels], bins=bins, alpha=0.6, color=PALETTE_FAILURE,
                label=f"Failure (n={np.sum(all_labels)})", density=True)
        ax.set_title(f"{label} — {title}", fontsize=10)
        ax.set_xlabel("Cosine similarity to normal centroid")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Embedding separation: normal vs failure frames", fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "separation_distributions.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")

    return metrics


# ---------------------------------------------------------------------------
# 5. Fusion comparison: early vs late
# ---------------------------------------------------------------------------

def plot_fusion_comparison(episodes: list[dict], output_dir: str):
    """
    Compare three fusion strategies on a single episode's temporal trajectory:
      - Visual only
      - Audio only
      - Early fusion: concat (512+512) → PCA 2D
      - Late fusion:  PCA(visual, 2D) + PCA(audio, 2D) = 4D → plot first 2 dims
    """
    # Use the episode with the most frames for best illustration
    ep = max(episodes, key=lambda e: len(e["visual"]))

    visual = ep["visual"]
    audio = ep["audio"]
    labels = ep["failure_labels"]
    ts = ep["timestamps"]

    strategies = {
        "Visual only (CLIP)": PCA(n_components=2).fit_transform(visual),
        "Audio only (WAV2CLIP)": PCA(n_components=2).fit_transform(audio),
        "Early fusion (concat → PCA)": PCA(n_components=2).fit_transform(
            np.concatenate([visual, audio], axis=-1)
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = cm.viridis(np.linspace(0, 1, len(ts)))

    for ax, (name, pca_2d) in zip(axes, strategies.items()):
        for j in range(len(ts) - 1):
            ax.plot(pca_2d[j:j+2, 0], pca_2d[j:j+2, 1],
                    color=colors[j], alpha=0.7, lw=1.5)
        ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=np.linspace(0, 1, len(ts)),
                   cmap="viridis", s=15, zorder=3)
        ax.scatter(*pca_2d[0], marker="s", c="green", s=80, zorder=6, label="Start")
        ax.scatter(*pca_2d[-1], marker="^", c="purple", s=80, zorder=6, label="End")

        fail_idxs = np.where(labels)[0]
        if len(fail_idxs) > 0:
            ax.scatter(pca_2d[fail_idxs, 0], pca_2d[fail_idxs, 1],
                       marker="X", c=PALETTE_FAILURE, s=200, zorder=7,
                       edgecolors="black", lw=0.5, label="Failure")

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle(f"Fusion comparison — {ep['id']} (color=time: dark→early, bright→late)",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "fusion_comparison.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(encoded_dir: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading encoded episodes...")
    episodes = load_all_encoded(encoded_dir)
    print(f"  Loaded {len(episodes)} episodes, "
          f"{sum(len(e['visual']) for e in episodes)} total frames")

    metrics = {}

    print("\n[1/5] UMAP — visual embeddings colored by failure...")
    plot_umap_failure(episodes, OUTPUT_DIR)

    print("\n[2/5] UMAP — visual + audio by modality and failure...")
    plot_umap_modality(episodes, OUTPUT_DIR)

    print("\n[3/5] UMAP — visual embeddings by task...")
    plot_umap_task(episodes, OUTPUT_DIR)

    print("\n[4/5] Temporal trajectory (PCA)...")
    plot_temporal_trajectory(episodes, OUTPUT_DIR)

    print("\n[5/5] Cross-modal similarity...")
    sims_normal, sims_failure = plot_cross_modal_similarity(episodes, OUTPUT_DIR)
    metrics["crossmodal_sim_normal_mean"] = float(np.mean(sims_normal))
    metrics["crossmodal_sim_failure_mean"] = float(np.mean(sims_failure))

    print("\n[+] Normal vs failure separation metrics...")
    sep_metrics = plot_separation_distributions(episodes, OUTPUT_DIR)
    metrics.update(sep_metrics)

    print("\n[+] Fusion comparison...")
    plot_fusion_comparison(episodes, OUTPUT_DIR)

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Metrics Summary ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else ENCODED_DIR
    main(target)
