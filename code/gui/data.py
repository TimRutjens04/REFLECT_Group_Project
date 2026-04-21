import json
import os
import re
import time

import numpy as np
import streamlit as st
from sklearn.decomposition import PCA

from .config import ALIGNED_DIR, DATA_DIR, ENCODED_DIR, TASKS_JSON
from .logger import log


@st.cache_data
def load_sim_task_meta(episode_id: str) -> dict | None:
    """
    Load actions and object_list from the per-episode task.json for sim episodes
    (boilWater-N, makeSalad-N). Returns None if the file does not exist.

    Path convention: data/sim_data/<task_name>/<episode_id>/task.json
    where <task_name> is the episode_id with the trailing -N stripped.
    """
    task_name = re.sub(r"-\d+$", "", episode_id)   # "boilWater-3" → "boilWater"
    path = os.path.join(DATA_DIR, "sim_data", task_name, episode_id, "task.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        meta = json.load(f)
    return {
        "actions":     meta.get("actions", []),
        "object_list": meta.get("object_list", []),
        "name":        meta.get("name", task_name),
        "gt_failure_reason": meta.get("gt_failure_reason", ""),
        "success_condition": meta.get("success_condition", ""),
    }


@st.cache_data
def load_task_metadata() -> dict[str, dict]:
    """Load tasks_real_world.json indexed by general_folder_name."""
    if not os.path.exists(TASKS_JSON):
        return {}
    with open(TASKS_JSON) as f:
        raw = json.load(f)
    return {v["general_folder_name"]: v for v in raw.values()}


@st.cache_data(show_spinner="Loading episode...")
def load_episode(episode_id: str) -> dict:
    """Load frames from aligned/ and embeddings from encoded/."""
    log.info("Loading episode: %s", episode_id)
    t_load = time.perf_counter()
    aligned = np.load(os.path.join(ALIGNED_DIR, f"{episode_id}.npz"), allow_pickle=False)
    encoded = np.load(os.path.join(ENCODED_DIR, f"{episode_id}.npz"), allow_pickle=False)

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
        "audio_windows":  aligned["audio_windows"],
        "audio_sr":       int(aligned["audio_sr"]),
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
