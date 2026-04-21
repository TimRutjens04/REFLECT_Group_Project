import json
import os
import time

import numpy as np
import streamlit as st
import torch
from PIL import Image

from .config import DEVICE, OWL_DIR, TASKS_JSON
from .data import load_episode
from .logger import log

OWLVIT_THRESHOLD  = 0.10  # detections below this are marked "not detected"
MAX_SPIKE_FRAMES  = 8     # max additional spike frames on top of failure windows
FAILURE_WINDOW    = 10    # frames on each side of a failure to sample densely
MAX_ALIASES       = 5     # max text queries per object (incl. original name)


def _wordnet_aliases(name: str) -> list[str]:
    """
    Expand an object name to related visual terms using WordNet synonyms and
    immediate hypernyms (broader categories). Returns deduplicated 'a <term>'
    strings, starting with the original. Falls back gracefully if WordNet data
    is unavailable.

    Multi-word names (e.g. 'green pear') are looked up both as-is and by the
    head noun ('pear') to maximise synset coverage.
    Scientific names and very long phrases are filtered out as they are not
    meaningful text queries for a vision-language model.
    """
    try:
        from nltk.corpus import wordnet as wn
    except LookupError:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.corpus import wordnet as wn

    aliases: list[str] = [f"a {name}"]
    seen: set[str] = {name}

    def _is_visual(term: str) -> bool:
        """Reject scientific names (CamelCase words) and very long phrases."""
        return len(term) < 30 and not any(c.isupper() for c in term[1:])

    # Look up the full name and the head noun for multi-word names
    lookup_terms = [name]
    words = name.split()
    if len(words) > 1:
        lookup_terms.append(words[-1])  # e.g. "pear" from "green pear"

    for term in lookup_terms:
        for syn in wn.synsets(term.replace(" ", "_"), pos=wn.NOUN)[:2]:
            for lemma in syn.lemmas()[:3]:
                alias = lemma.name().replace("_", " ")
                if alias not in seen and _is_visual(alias):
                    aliases.append(f"a {alias}")
                    seen.add(alias)
            for hyper in syn.hypernyms()[:1]:
                for lemma in hyper.lemmas()[:2]:
                    alias = lemma.name().replace("_", " ")
                    if alias not in seen and _is_visual(alias):
                        aliases.append(f"a {alias}")
                        seen.add(alias)
            if len(aliases) >= MAX_ALIASES:
                break
        if len(aliases) >= MAX_ALIASES:
            break

    return aliases[:MAX_ALIASES]


def _select_sample_frames(frame_deltas: np.ndarray,
                           failure_labels: np.ndarray) -> list[int]:
    """
    Choose which frames to run OWL-ViT detection on.

    Strategy:
      1. Always include frame 0 (initial scene layout).
      2. Always include every failure frame plus a ±FAILURE_WINDOW dense window
         around each — this gives per-frame detection scores for the full
         lead-up to each failure (no cap on these).
      3. Add up to MAX_SPIKE_FRAMES additional frames where frame_deltas >
         mean + 1.5*std, to capture object movement between failure windows.
    """
    n = len(frame_deltas)
    failure_indices = [int(i) for i, f in enumerate(failure_labels) if f]

    # Dense failure windows — uncapped
    must_include: set[int] = {0}
    for fi in failure_indices:
        for w in range(max(0, fi - FAILURE_WINDOW), min(n, fi + FAILURE_WINDOW + 1)):
            must_include.add(w)

    # Sparse spike frames — capped
    threshold    = frame_deltas.mean() + 1.5 * frame_deltas.std()
    spike_frames = [i for i in range(n)
                    if frame_deltas[i] > threshold and i not in must_include]
    spike_frames.sort(key=lambda i: -frame_deltas[i])

    selected = sorted(must_include | set(spike_frames[:MAX_SPIKE_FRAMES]))

    log.info("Frame sampling: failure_windows=%d  spikes=%d  total=%d  selected=%s",
             len(must_include), len(spike_frames), len(selected), selected)
    return selected


def _owl_npz_path(episode_id: str) -> str:
    return os.path.join(OWL_DIR, f"{episode_id}.npz")


def _object_list_for(episode_id: str) -> list[str]:
    """
    Look up the object_list for an episode.
    Real-world episodes: tasks_real_world.json (central file).
    Sim episodes (boilWater-N, makeSalad-N): per-episode task.json.
    """
    # Try real-world metadata first
    if os.path.exists(TASKS_JSON):
        with open(TASKS_JSON) as f:
            raw = json.load(f)
        meta = {v["general_folder_name"]: v for v in raw.values()}
        obj_list = meta.get(episode_id, {}).get("object_list", [])
        if obj_list:
            return obj_list

    # Fall back to per-episode task.json (sim episodes)
    import re
    task_name = re.sub(r"-\d+$", "", episode_id)
    sim_json = os.path.join(os.path.dirname(TASKS_JSON), task_name, episode_id, "task.json")
    if os.path.exists(sim_json):
        with open(sim_json) as f:
            return json.load(f).get("object_list", [])

    return []


@st.cache_data(show_spinner="Loading pre-computed OWL-ViT detections...")
def load_precomputed_owl(episode_id: str) -> dict[int, list[dict]] | None:
    """
    Load pre-computed OWL-ViT detections from owl/<episode_id>.npz.
    Returns a snapshots dict (same format as compute_scene_graph) covering
    ALL frames, or None if the file does not exist.
    """
    path = _owl_npz_path(episode_id)
    if not os.path.exists(path):
        return None

    object_list = _object_list_for(episode_id)
    if not object_list:
        return None

    d              = np.load(path, allow_pickle=False)
    sample_indices = d["sample_indices"].tolist()   # [frame_idx, ...]
    scores         = d["scores"]                    # (M, n_obj)
    detected       = d["detected"]                  # (M, n_obj)
    cx             = d["cx_norm"]                   # (M, n_obj)
    cy             = d["cy_norm"]                   # (M, n_obj)
    boxes_np       = d["boxes"]                     # (M, n_obj, 4)

    snapshots: dict[int, list[dict]] = {}
    for row, frame_idx in enumerate(sample_indices):
        objs = []
        for i, name in enumerate(object_list):
            s   = float(scores[row, i])
            det = bool(detected[row, i])
            b   = boxes_np[row, i]
            box = [int(b[0]), int(b[1]), int(b[2]), int(b[3])] \
                  if det and not np.any(np.isnan(b)) else None
            objs.append({
                "name":     name,
                "score":    s,
                "detected": det,
                "box":      box,
                "cx_norm":  float(cx[row, i]),
                "cy_norm":  float(cy[row, i]),
            })
        snapshots[frame_idx] = objs

    log.info("Loaded pre-computed OWL-ViT for %s: %d sampled frames, %d objects",
             episode_id, len(sample_indices), len(object_list))
    return snapshots


@st.cache_data(show_spinner="Detecting objects in scene (OWL-ViT)...")
def compute_scene_graph(episode_id: str, object_list: tuple[str, ...],
                         _model, _processor) -> dict[int, list[dict]]:
    """
    Detect each object using OWL-ViT v2 on each adaptively-selected frame.

    Returns a snapshot dict keyed by frame index:
        {frame_idx: [object_dict, ...]}

    Each object_dict:
        name      str        — object label
        score     float      — best OWL-ViT confidence in THIS frame
        detected  bool       — True if confidence >= OWLVIT_THRESHOLD
        box       list|None  — [x1, y1, x2, y2] pixel coords in this frame
        cx_norm   float      — box centre x, normalised 0–1
        cy_norm   float      — box centre y, normalised 0–1

    At display time, pick the snapshot with the largest key <= current_frame
    so the scene graph always reflects the most recent detection, not a future one.
    """
    log.info("Detecting objects for %s  objects=%s", episode_id, list(object_list))
    t0 = time.perf_counter()

    episode        = load_episode(episode_id)
    all_frames     = episode["frames"]
    frame_deltas   = episode["frame_deltas"]
    failure_labels = episode["failure_labels"]
    n = len(all_frames)

    sample_indices = _select_sample_frames(frame_deltas, failure_labels)

    # Build prompt list: WordNet expansion per object, track which object each
    # prompt belongs to so we can map OWL-ViT label indices back correctly.
    all_prompts: list[str] = []
    prompt_to_obj: list[int] = []
    for i, obj_name in enumerate(object_list):
        for alias in _wordnet_aliases(obj_name):
            all_prompts.append(alias)
            prompt_to_obj.append(i)
    texts = [all_prompts]
    log.info("Prompts (%d total): %s", len(all_prompts),
             {o: _wordnet_aliases(o) for o in object_list})

    snapshots: dict[int, list[dict]] = {}

    progress = st.progress(0, text="Detecting objects across frames...")
    for step, frame_idx in enumerate(sample_indices):
        frame = all_frames[frame_idx]
        h, w  = frame.shape[:2]
        target_sizes = torch.tensor([[h, w]], device=DEVICE)

        inputs = _processor(text=texts, images=Image.fromarray(frame),
                            return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)

        results = _processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=0.0,
            target_sizes=target_sizes,
            text_labels=texts,
        )[0]

        frame_boxes  = results["boxes"].cpu().numpy()
        frame_scores = results["scores"].cpu().numpy()
        frame_labels = results["labels"]

        # Best detection per object IN THIS FRAME ONLY
        frame_objects = []
        for i, name in enumerate(object_list):
            mask = np.array([prompt_to_obj[int(lbl)] == i for lbl in frame_labels])
            if mask.any():
                score   = float(frame_scores[mask].max())
                raw_box = frame_boxes[mask][frame_scores[mask].argmax()].tolist()
            else:
                score, raw_box = 0.0, None

            detected = score >= OWLVIT_THRESHOLD
            if detected and raw_box is not None:
                x1, y1, x2, y2 = raw_box
                box     = [int(x1), int(y1), int(x2), int(y2)]
                cx_norm = ((x1 + x2) / 2) / w
                cy_norm = ((y1 + y2) / 2) / h
            else:
                box     = None
                cx_norm = 0.5
                cy_norm = 0.5

            log.debug("  [fr %d] %-20s  detected=%s  score=%.3f  box=%s",
                      frame_idx, name, detected, score, box)
            frame_objects.append({
                "name":     name,
                "score":    score,
                "detected": detected,
                "box":      box,
                "cx_norm":  cx_norm,
                "cy_norm":  cy_norm,
            })

        snapshots[frame_idx] = frame_objects
        progress.progress((step + 1) / len(sample_indices),
                          text=f"Detecting objects... frame {frame_idx}/{n-1}")

    progress.empty()
    log.info("OWL-ViT done: %d frames in %.2fs", len(sample_indices),
             time.perf_counter() - t0)
    return snapshots


def active_objects(snapshots: dict[int, list[dict]], current_frame: int) -> list[dict]:
    """
    Return the object list from the most recent sampled frame <= current_frame.
    Falls back to the earliest snapshot if none qualifies (shouldn't happen since
    frame 0 is always sampled).
    """
    eligible = [k for k in snapshots if k <= current_frame]
    key = max(eligible) if eligible else min(snapshots)
    return snapshots[key]


def _box_to_attn_patch(cx_norm: float, cy_norm: float) -> tuple[int, int]:
    """Map normalised box centre (0–1) to 7×7 CLIP attention patch (r, c)."""
    r = min(int(cy_norm * 7), 6)
    c = min(int(cx_norm * 7), 6)
    return r, c


def _rel_label(a: dict, b: dict) -> str:
    """Infer primary spatial relationship from normalised (cx_norm, cy_norm)."""
    dx = b["cx_norm"] - a["cx_norm"]
    dy = b["cy_norm"] - a["cy_norm"]
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if dist < 0.1:
        return "near"
    if abs(dy) >= abs(dx):
        return "below" if dy > 0 else "above"
    return "right of" if dx > 0 else "left of"


def _jitter_positions(object_locs: list[dict]) -> list[tuple[float, float]]:
    """
    Compute display (cx, cy) in normalised 0–1 space for each object.
    Close detected objects are spread in a small circle; undetected objects
    are placed in a row below the main plot area (y = 1.12).
    """
    positions: list[tuple[float, float]] = [(0.0, 0.0)] * len(object_locs)

    detected_indices   = [i for i, o in enumerate(object_locs) if o["detected"]]
    undetected_indices = [i for i, o in enumerate(object_locs) if not o["detected"]]

    visited = {i: False for i in detected_indices}
    groups: list[list[int]] = []
    for i in detected_indices:
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        for j in detected_indices:
            if visited[j]:
                continue
            dx = object_locs[i]["cx_norm"] - object_locs[j]["cx_norm"]
            dy = object_locs[i]["cy_norm"] - object_locs[j]["cy_norm"]
            if (dx ** 2 + dy ** 2) ** 0.5 < 0.05:
                group.append(j)
                visited[j] = True
        groups.append(group)

    for group in groups:
        if len(group) == 1:
            idx = group[0]
            positions[idx] = (object_locs[idx]["cx_norm"], object_locs[idx]["cy_norm"])
        else:
            cx = float(np.mean([object_locs[i]["cx_norm"] for i in group]))
            cy = float(np.mean([object_locs[i]["cy_norm"] for i in group]))
            radius = 0.05
            for k, idx in enumerate(group):
                angle = 2 * np.pi * k / len(group)
                positions[idx] = (cx + radius * np.cos(angle),
                                  cy + radius * np.sin(angle))

    n_undet = len(undetected_indices)
    for k, idx in enumerate(undetected_indices):
        positions[idx] = ((k + 1) / (n_undet + 1), 1.12)

    return positions
