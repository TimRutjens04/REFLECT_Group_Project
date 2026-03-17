"""
Live Embedding Visualizer — REFLECT / RoboFail dataset.

Simulates real-time perception by playing back aligned robot episodes with
CLIP visual and WAV2CLIP audio embeddings synchronized to the video frame.

Attention overlay: CLIP ViT-B/32 processes images as a 7×7 patch grid.
The attention weights from the [CLS] token to each patch (last layer) reveal
which spatial region drove the embedding at each timestep.

Scene graph: for each object in the task's object_list, CLIP text embeddings
are compared against patch-level visual embeddings to localize objects in the
7×7 grid without any bounding box annotations. The graph shows object positions
alongside the current attention map, making it visible when CLIP is attending
to an unexpected region.

Layout:
  Left  — current robot frame (with optional attention overlay + red box)
           + failure indicator + per-frame metrics
  Right — signal plots (embedding norms, frame delta) with moving cursor
          PCA trajectory with moving dot

Below (when attention on):
  Scene graph — object nodes localized via CLIP text-patch similarity,
                overlaid with current CLS attention, action sequence shown

Controls (sidebar):
  Episode selector, Play/Pause, speed, frame scrubber, attention toggle

Data sources:
  aligned/<episode>.npz  — frames (N, H, W, 3) uint8
  encoded/<episode>.npz  — visual_embeddings (N, 512), audio_embeddings (N, 512)
  data/tasks_real_world.json — object_list, actions per episode

Usage:
    poetry run streamlit run code/gui.py
"""

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler

import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from sklearn.decomposition import PCA

plt.switch_backend("Agg")  # avoid macOS AppKit crash in Streamlit subprocess


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALIGNED_DIR = os.path.join(ROOT, "aligned")
ENCODED_DIR = os.path.join(ROOT, "encoded")
TASKS_JSON  = os.path.join(ROOT, "data", "tasks_real_world.json")
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
LOG_PATH    = os.path.join(ROOT, "logs", "gui.log")


# ---------------------------------------------------------------------------
# Logging — file + stderr, initialized once per process
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logger = logging.getLogger("gui")
    if logger.handlers:
        return logger  # already configured (Streamlit re-runs the module)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Rotating file: 1 MB max, keep last 3 files
    fh = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Console (visible in the terminal where `just gui` was run)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = _setup_logger()


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

@st.cache_data
def load_task_metadata() -> dict[str, dict]:
    """Load tasks_real_world.json indexed by general_folder_name."""
    if not os.path.exists(TASKS_JSON):
        return {}
    with open(TASKS_JSON) as f:
        raw = json.load(f)
    return {v["general_folder_name"]: v for v in raw.values()}


# ---------------------------------------------------------------------------
# Model loading — cached once for the process lifetime
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CLIP ViT-B/32...")
def load_clip_model():
    """ViT-B/32 — used for attention maps (matches stored embeddings)."""
    log.info("Loading CLIP ViT-B/32 on device=%s", DEVICE)
    t0 = time.perf_counter()
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    log.info("CLIP ViT-B/32 loaded in %.2fs", time.perf_counter() - t0)
    return model, preprocess


@st.cache_resource(show_spinner="Loading CLIP ViT-L/14 for localization...")
def load_localization_model():
    """ViT-L/14 — used for patch-level object localization (16×16 = 256 patches)."""
    log.info("Loading CLIP ViT-L/14 on device=%s", DEVICE)
    t0 = time.perf_counter()
    model, preprocess = clip.load("ViT-L/14", device=DEVICE)
    model.eval()
    log.info("CLIP ViT-L/14 loaded in %.2fs", time.perf_counter() - t0)
    return model, preprocess


# ---------------------------------------------------------------------------
# Episode data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading episode...")
def load_episode(episode_id: str) -> dict:
    """Load frames from aligned/ and embeddings from encoded/."""
    log.info("Loading episode: %s", episode_id)
    t_load = time.perf_counter()
    aligned = np.load(os.path.join(ALIGNED_DIR, f"{episode_id}.npz"), allow_pickle=True)
    encoded = np.load(os.path.join(ENCODED_DIR, f"{episode_id}.npz"), allow_pickle=True)

    visual_emb = encoded["visual_embeddings"].astype(np.float32)
    audio_emb  = encoded["audio_embeddings"].astype(np.float32)

    mean_vis = visual_emb.mean(axis=0)
    mean_vis /= np.linalg.norm(mean_vis) + 1e-8
    visual_norms = (visual_emb @ mean_vis).astype(np.float32)

    mean_aud = audio_emb.mean(axis=0)
    mean_aud /= np.linalg.norm(mean_aud) + 1e-8
    audio_norms = (audio_emb @ mean_aud).astype(np.float32)

    n = len(visual_emb)
    frame_deltas = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        frame_deltas[i] = 1.0 - float(np.dot(visual_emb[i], visual_emb[i - 1]))

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(visual_emb).astype(np.float32)

    result = {
        "frames":         aligned["frames"],
        "timestamps":     encoded["timestamps"],
        "failure_labels": encoded["failure_labels"],
        "visual_norms":   visual_norms,
        "audio_norms":    audio_norms,
        "frame_deltas":   frame_deltas,
        "pca_coords":     pca_coords,
    }
    log.info(
        "Episode %s loaded: %d frames, %.1fs duration, %d failure frames  [%.2fs]",
        episode_id, n, float(encoded["timestamps"][-1]),
        int(encoded["failure_labels"].sum()),
        time.perf_counter() - t_load,
    )
    return result


def list_episodes() -> list[str]:
    aligned = {f[:-4] for f in os.listdir(ALIGNED_DIR) if f.endswith(".npz")}
    encoded = {f[:-4] for f in os.listdir(ENCODED_DIR) if f.endswith(".npz")}
    return sorted(aligned & encoded)


# ---------------------------------------------------------------------------
# CLIP attention maps
# ---------------------------------------------------------------------------

def _single_attention_map(frame: np.ndarray, model, preprocess) -> np.ndarray:
    """
    Extract 7×7 CLS→patch attention from CLIP ViT-B/32 last layer.
    Returns (7, 7) float32.
    """
    img = preprocess(Image.fromarray(frame)).unsqueeze(0).to(DEVICE)
    captured = {}
    guard = [False]

    def hook(module, input, output):
        if guard[0]:
            return
        guard[0] = True
        q, k, v = input[0], input[1], input[2]
        with torch.no_grad():
            _, attn = module(q, k, v, need_weights=True, average_attn_weights=True)
        captured["attn"] = attn.detach().cpu()
        guard[0] = False

    handle = model.visual.transformer.resblocks[-1].attn.register_forward_hook(hook)
    with torch.no_grad():
        model.encode_image(img)
    handle.remove()

    return captured["attn"][0, 0, 1:].reshape(7, 7).numpy().astype(np.float32)


@st.cache_data(show_spinner="Computing attention maps (one-time)...")
def compute_attention_maps(episode_id: str, _model, _preprocess) -> np.ndarray:
    """Compute (N, 7, 7) CLS attention maps for every frame. Cached per episode."""
    log.info("Computing attention maps for %s", episode_id)
    t0 = time.perf_counter()
    frames = load_episode(episode_id)["frames"]
    n = len(frames)
    maps = np.zeros((n, 7, 7), dtype=np.float32)
    progress = st.progress(0, text="Computing attention maps...")
    for i, frame in enumerate(frames):
        maps[i] = _single_attention_map(frame, _model, _preprocess)
        progress.progress((i + 1) / n, text=f"Attention maps: {i+1}/{n}")
    progress.empty()
    elapsed = time.perf_counter() - t0
    log.info("Attention maps done: %d frames in %.2fs (%.3fs/frame)", n, elapsed, elapsed / n)
    return maps


# ---------------------------------------------------------------------------
# Scene graph: object localization via CLIP text–patch similarity
# ---------------------------------------------------------------------------

def _get_patch_embeddings(frame: np.ndarray, model, preprocess) -> np.ndarray:
    """
    Extract 49 patch embeddings (7×7) from CLIP's last transformer layer,
    projected into the shared CLIP embedding space.

    CLIP's forward only keeps the CLS token after the transformer and applies
    ln_post + proj to it. Here we apply the same operations to all patch tokens
    so their embeddings are comparable to text embeddings for localization.

    Returns: (49, 512) float32, L2-normalized
    """
    img = preprocess(Image.fromarray(frame)).unsqueeze(0).to(DEVICE)
    captured = {}

    def hook(module, input, output):
        # output: (seq_len, batch, dim) from the full transformer stack
        captured["tokens"] = output.permute(1, 0, 2).detach()  # (B, 50, D)

    handle = model.visual.transformer.register_forward_hook(hook)
    with torch.no_grad():
        model.encode_image(img)
    handle.remove()

    # Patch tokens = indices 1..49 (index 0 is CLS)
    patch_tokens = captured["tokens"][0, 1:, :]  # (49, D) on DEVICE

    with torch.no_grad():
        # Apply ln_post (designed for CLS but valid for patches — same LayerNorm)
        patch_norm = model.visual.ln_post(patch_tokens)
        # Project into shared 512-dim CLIP space
        if model.visual.proj is not None:
            patch_emb = patch_norm @ model.visual.proj   # (49, 512)
        else:
            patch_emb = patch_norm
        patch_emb = patch_emb / patch_emb.norm(dim=-1, keepdim=True)

    return patch_emb.cpu().numpy().astype(np.float32)  # (49, 512)


@st.cache_data(show_spinner="Localizing objects in scene...")
def compute_scene_graph(episode_id: str, object_list: tuple[str, ...],  # noqa: C901
                         _model, _preprocess) -> list[dict]:
    """
    Localize each object in the 7×7 patch grid using CLIP text–patch similarity.
    Uses the first frame of the episode as the reference scene.

    Returns list of dicts:
        name      str      — object label
        r, c      int      — best-match patch row/col in [0, 6]
        score     float    — cosine similarity at best patch
        sim_map   (7,7)    — full similarity map for this object
    """
    log.info("Computing scene graph for %s  objects=%s", episode_id, list(object_list))
    t0 = time.perf_counter()
    frame = load_episode(episode_id)["frames"][0]
    patch_emb = _get_patch_embeddings(frame, _model, _preprocess)  # (49, 512)

    # Encode object names. Prefix with "a" for better CLIP alignment.
    prompts = [f"a {obj}" for obj in object_list]
    tokens = clip.tokenize(prompts, truncate=True).to(DEVICE)
    with torch.no_grad():
        text_emb = _model.encode_text(tokens).float()
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    sim = text_emb.cpu().numpy() @ patch_emb.T  # (n_objects, 49)

    n_patches_side = int(sim.shape[1] ** 0.5)  # 16 for ViT-L/14, 7 for ViT-B/32
    results = []
    for i, name in enumerate(object_list):
        best = int(sim[i].argmax())
        r, c = divmod(best, n_patches_side)
        score = float(sim[i, best])
        log.debug("  %-40s  → patch (%d,%d)  sim=%.3f", name, r, c, score)
        results.append({
            "name":    name,
            "r":       r,
            "c":       c,
            "score":   score,
            "sim_map": sim[i].reshape(n_patches_side, n_patches_side).astype(np.float32),
            "grid":    n_patches_side,
        })

    log.info("Scene graph done in %.2fs", time.perf_counter() - t0)
    return results


def _rel_label(a: dict, b: dict) -> str:
    """Infer primary spatial relationship from patch (r, c) coordinates."""
    dr = b["r"] - a["r"]
    dc = b["c"] - a["c"]
    if (dr**2 + dc**2) ** 0.5 < 1.5:
        return "near"
    if abs(dr) >= abs(dc):
        return "above" if dr > 0 else "below"
    return "right of" if dc > 0 else "left of"


def _jitter_positions(object_locs: list[dict]) -> list[tuple[float, float]]:
    """
    Compute display (cx, cy) for each object. When multiple objects share the
    same patch, spread them in a small circle so labels don't overlap.
    """
    from collections import defaultdict
    by_patch: dict = defaultdict(list)
    for i, obj in enumerate(object_locs):
        by_patch[(obj["r"], obj["c"])].append(i)

    positions: list[tuple[float, float]] = [(0.0, 0.0)] * len(object_locs)
    for (r, c), indices in by_patch.items():
        if len(indices) == 1:
            positions[indices[0]] = (float(c), float(r))
        else:
            radius = 0.55
            for k, idx in enumerate(indices):
                angle = 2 * np.pi * k / len(indices)
                positions[idx] = (c + radius * np.cos(angle), r + radius * np.sin(angle))
    return positions


def plot_inline_scene_graph(object_locs: list[dict], attn_7x7: np.ndarray) -> plt.Figure:
    """
    Compact scene graph for the left column — objects as nodes at their patch
    positions, pairwise spatial relationship edges, current attention highlighted.
    Co-located objects (same patch) are jittered apart so labels don't overlap.
    Supports any grid size (7×7 for ViT-B/32, 16×16 for ViT-L/14).
    """
    g = object_locs[0]["grid"] if object_locs else 7  # grid size (e.g. 16)
    lim = g - 0.5

    # Resize ViT-B/32 attention (7×7) to match the localization grid
    attn = cv2.resize(attn_7x7, (g, g), interpolation=cv2.INTER_LINEAR)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(lim, -0.5)  # row 0 at top
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Scene graph — spatial relationships", color="white", fontsize=8, pad=4)

    flat_idx = np.argsort(attn.ravel())[::-1][:3]
    top_patches = {divmod(int(i), g) for i in flat_idx}
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(object_locs), 1)))
    positions = _jitter_positions(object_locs)

    # Edges (pairwise relationships)
    for i in range(len(object_locs)):
        for j in range(i + 1, len(object_locs)):
            a, b = object_locs[i], object_locs[j]
            cx_a, cy_a = positions[i]
            cx_b, cy_b = positions[j]
            label = _rel_label(a, b)
            ax.plot([cx_a, cx_b], [cy_a, cy_b], color="#444455", lw=1.0, zorder=1)
            ax.text((cx_a + cx_b) / 2, (cy_a + cy_b) / 2, label,
                    ha="center", va="center", fontsize=4.5, color="#aaaaaa", zorder=2,
                    bbox=dict(facecolor=DARK_BG, edgecolor="none", pad=1))

    # Nodes — drawn at jittered positions
    for i, obj in enumerate(object_locs):
        cx, cy = positions[i]
        attended = (obj["r"], obj["c"]) in top_patches
        circle = plt.Circle((cx, cy), 0.38, color=colors[i], zorder=3,
                             ec=CURSOR if attended else "white",
                             lw=2.5 if attended else 1.0)
        ax.add_patch(circle)
        ax.text(cx, cy, " ".join(obj["name"].split()[:2]),
                ha="center", va="center", fontsize=5,
                color="white", fontweight="bold", zorder=4)

    # Attention peak marker (in resized grid coordinates)
    pr, pc = np.unravel_index(attn.argmax(), (g, g))
    ax.scatter(pc, pr, marker="*", s=120, color=CURSOR, zorder=5,
               edgecolors="white", linewidths=0.4)

    fig.tight_layout(pad=0.3)
    return fig


def plot_scene_graph(object_locs: list[dict], attn_7x7: np.ndarray,
                     actions: list[str]) -> plt.Figure:
    """
    Render the object localization panel:
    - Background: ViT-B/32 attention heatmap resized to the localization grid
    - Nodes: each object at its ViT-L/14 patch position (16×16 grid)
      * Yellow border  = within top-3 attention patches
      * White border   = elsewhere
    - Node label shows object name + similarity score
    - Action list shown on the right
    """
    g = object_locs[0]["grid"] if object_locs else 7
    lim = g - 0.5

    # Resize ViT-B/32 attention to localization grid resolution
    attn = cv2.resize(attn_7x7, (g, g), interpolation=cv2.INTER_LINEAR)

    fig, (ax_graph, ax_actions) = plt.subplots(
        1, 2, figsize=(10, 4),
        gridspec_kw={"width_ratios": [3, 1]}
    )
    fig.patch.set_facecolor(DARK_BG)

    # ── Left: object localization grid ─────────────────────────────────────
    ax_graph.set_facecolor(DARK_BG)
    ax_graph.set_xlim(-0.5, lim)
    ax_graph.set_ylim(lim, -0.5)   # row 0 at top
    ax_graph.set_aspect("equal")
    ax_graph.set_title(
        f"Object localization — CLIP ViT-L/14 text-patch similarity ({g}×{g} patch grid)",
        color="white", fontsize=9)
    ax_graph.set_xlabel("Patch column →", color="white", fontsize=7)
    ax_graph.set_ylabel("Patch row ↓", color="white", fontsize=7)
    ax_graph.tick_params(colors="white", labelsize=6)
    for spine in ax_graph.spines.values():
        spine.set_edgecolor(GRID_COL)

    # Attention heatmap as background (resized to localization grid)
    ax_graph.imshow(attn, extent=(-0.5, lim, lim, -0.5),
                    cmap="hot", alpha=0.45, vmin=0, vmax=attn.max(),
                    aspect="auto", origin="upper")

    # Grid lines
    for i in range(g + 1):
        ax_graph.axhline(i - 0.5, color=GRID_COL, lw=0.4)
        ax_graph.axvline(i - 0.5, color=GRID_COL, lw=0.4)

    # Top-3 attended patches (in resized grid)
    flat_idx = np.argsort(attn.ravel())[::-1][:3]
    top_patches = {divmod(int(i), g) for i in flat_idx}

    # Draw object nodes — jitter co-located objects to prevent label overlap
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(object_locs), 1)))
    positions = _jitter_positions(object_locs)
    for i, obj in enumerate(object_locs):
        r, c = obj["r"], obj["c"]
        cx, cy = positions[i]
        attended = (r, c) in top_patches
        edge_color = CURSOR if attended else "white"
        edge_lw    = 2.5   if attended else 1.0

        circle = plt.Circle(
            (cx, cy), 0.38,
            color=colors[i], zorder=3,
            ec=edge_color, lw=edge_lw
        )
        ax_graph.add_patch(circle)

        # Label: short name + score
        short_name = " ".join(obj["name"].split()[:2])
        ax_graph.text(
            cx, cy - 0.52, short_name,
            ha="center", va="bottom", fontsize=5.5, color="white",
            zorder=4, fontweight="bold" if attended else "normal"
        )
        ax_graph.text(
            cx, cy + 0.52, f"{obj['score']:.2f}",
            ha="center", va="top", fontsize=5, color="#aaaaaa", zorder=4
        )

    # Attention peak marker (★)
    peak_r, peak_c = np.unravel_index(attn_7x7.argmax(), (7, 7))
    ax_graph.scatter(peak_c, peak_r, marker="*", s=200, color=CURSOR,
                     zorder=5, edgecolors="white", linewidths=0.5)
    ax_graph.text(peak_c, peak_r - 0.6, "attn peak",
                  ha="center", va="bottom", fontsize=5, color=CURSOR, zorder=5)

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
            wrap=True
        )

    fig.tight_layout(pad=0.6)
    return fig


# ---------------------------------------------------------------------------
# Attention overlay rendering
# ---------------------------------------------------------------------------

def overlay_attention(frame: np.ndarray, attn_7x7: np.ndarray,
                       alpha: float = 0.45) -> np.ndarray:
    """
    Blend CLIP attention heatmap onto the frame and draw a red box around
    the highest-attention patch, mapped back to original image coordinates.
    Returns (H, W, 3) uint8 RGB.
    """
    h, w = frame.shape[:2]

    attn_resized = cv2.resize(attn_7x7, (w, h), interpolation=cv2.INTER_LINEAR)
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    heatmap_bgr = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    composite = (
        frame.astype(np.float32) * (1 - alpha)
        + heatmap_rgb.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)

    # Map top-attention patch back to original image coordinates.
    # CLIP center-crops to min(h,w)×min(h,w) before resizing to 224×224.
    crop_size   = min(h, w)
    top_offset  = (h - crop_size) // 2
    left_offset = (w - crop_size) // 2
    scale       = crop_size / 224.0

    r, c = np.unravel_index(attn_7x7.argmax(), (7, 7))
    y1 = max(0,     top_offset  + int(r * 32 * scale))
    x1 = max(0,     left_offset + int(c * 32 * scale))
    y2 = min(h - 1, top_offset  + int((r + 1) * 32 * scale))
    x2 = min(w - 1, left_offset + int((c + 1) * 32 * scale))

    cv2.rectangle(composite, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)
    return composite


# ---------------------------------------------------------------------------
# Signal + PCA plots
# ---------------------------------------------------------------------------

DARK_BG   = "#0e1117"
GRID_COL  = "#1e2130"
LINE_VIS  = "#4c9be8"
LINE_AUD  = "#7ec87e"
LINE_DELT = "#e8754c"
CURSOR    = "#f0c040"


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


# ---------------------------------------------------------------------------
# Main app — split into focused helpers to keep complexity manageable
# ---------------------------------------------------------------------------

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
    """Load attention maps (ViT-B/32) and scene graph (ViT-L/14), with error logging."""
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

    # Object localization: ViT-L/14 (16×16 patches = finer spatial resolution)
    if show_scene_graph and object_list:
        l14_model, l14_pre = load_localization_model()
        try:
            scene_graph = compute_scene_graph(
                episode_id, tuple(object_list), l14_model, l14_pre
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
    if attn_maps is not None:
        display_frame = overlay_attention(frame, attn_maps[idx], alpha=attn_alpha)
        st.image(display_frame, use_container_width=True,
                 caption=f"Frame {idx}/{n-1} — red box: highest CLIP attention patch")
    else:
        st.image(frame, use_container_width=True, caption=f"Frame {idx} / {n-1}")

    # Scene graph — spatial relationships between objects vs. current attention
    if scene_graph is not None and attn_maps is not None:
        fig_sg = plot_inline_scene_graph(scene_graph, attn_maps[idx])
        st.pyplot(fig_sg, use_container_width=True)
        plt.close(fig_sg)
        st.caption(
            "Nodes = CLIP-localized objects. Edges = spatial relationships. "
            "**Yellow border** = in top-3 attended patches. **★** = attention peak."
        )

    # Compact per-frame metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Vis. sim.", f"{data['visual_norms'][idx]:.3f}")
    c2.metric("Aud. sim.", f"{data['audio_norms'][idx]:.3f}")
    c3.metric("Vis. Δ",    f"{data['frame_deltas'][idx]:.4f}",
              help="Cosine distance from previous frame")


def _render_object_localization_section(scene_graph, attn_maps, idx: int,
                                         actions: list[str], show_scene_graph: bool,
                                         object_list: list[str]):
    """Render full-width object localization panel below the main layout."""
    if scene_graph is not None and attn_maps is not None:
        st.divider()
        st.markdown("#### Object localization — CLIP text-patch similarity")
        st.caption(
            "Object positions estimated by comparing CLIP text embeddings against "
            "patch-level visual embeddings from the first frame. "
            "**Yellow border** = object within top-3 attended patches. "
            "**★** = attention peak. Explains why CLIP may not re-attend to an object "
            "whose position is already encoded in the embedding."
        )
        fig_sg = plot_scene_graph(scene_graph, attn_maps[idx], actions)
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

    _render_object_localization_section(scene_graph, attn_maps, idx, actions,
                                         show_scene_graph, object_list)

    # ── Playback loop ────────────────────────────────────────────────────────
    if st.session_state.get("playing", False):
        next_idx = idx + 1
        if next_idx >= n:
            st.session_state["playing"] = False
        else:
            time.sleep(0.5 / speed)
            st.session_state["current_frame"] = next_idx
            st.rerun()


if __name__ == "__main__":
    main()
