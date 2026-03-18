import time

import numpy as np
import streamlit as st
import torch
from PIL import Image

from .config import DEVICE
from .data import load_episode
from .logger import log

OWLVIT_THRESHOLD  = 0.10  # detections below this are marked "not detected"
MAX_SAMPLE_FRAMES = 8     # hard cap on OWL-ViT inference calls per episode
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
      2. Always include every failure frame (we care most about these).
      3. Include frames where frame_deltas > mean + 1.5*std — these are
         moments of significant scene change where object positions may
         have shifted, making fresh detection informative.
      4. Cap at MAX_SAMPLE_FRAMES total, prioritising spikes by magnitude.
    """
    n = len(frame_deltas)

    must_include: set[int] = {0}
    must_include.update(int(i) for i, f in enumerate(failure_labels) if f)

    threshold    = frame_deltas.mean() + 1.5 * frame_deltas.std()
    spike_frames = [i for i in range(n) if frame_deltas[i] > threshold]
    spike_frames.sort(key=lambda i: -frame_deltas[i])  # highest spikes first

    candidates = list(must_include) + [i for i in spike_frames if i not in must_include]
    selected   = sorted(candidates[:MAX_SAMPLE_FRAMES])

    log.info("Frame sampling: threshold=%.4f  spikes=%d  selected=%s",
             threshold, len(spike_frames), selected)
    return selected


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
