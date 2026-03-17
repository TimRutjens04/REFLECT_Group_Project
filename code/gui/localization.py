import time

import numpy as np
import streamlit as st
import torch
from PIL import Image

from .config import DEVICE
from .data import load_episode
from .logger import log

OWLVIT_THRESHOLD = 0.10  # detections below this are marked "not detected"


@st.cache_data(show_spinner="Detecting objects in scene (OWL-ViT)...")
def compute_scene_graph(episode_id: str, object_list: tuple[str, ...],
                         _model, _processor) -> list[dict]:
    """
    Detect each object in the first frame using OWL-ViT v2 zero-shot detection.

    Returns list of dicts:
        name      str        — object label
        score     float      — OWL-ViT confidence (0.0 if not detected)
        detected  bool       — True if confidence >= OWLVIT_THRESHOLD
        box       list|None  — [x1, y1, x2, y2] pixel coords in original frame
        cx_norm   float      — box center x, normalized 0–1 (0.5 if not detected)
        cy_norm   float      — box center y, normalized 0–1 (0.5 if not detected)
    """
    log.info("Detecting objects for %s  objects=%s", episode_id, list(object_list))
    t0 = time.perf_counter()
    frame = load_episode(episode_id)["frames"][0]
    h, w = frame.shape[:2]
    pil_image = Image.fromarray(frame)

    texts = [[f"a {obj}" for obj in object_list]]
    inputs = _processor(text=texts, images=pil_image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    target_sizes = torch.tensor([[h, w]], device=DEVICE)
    results = _processor.post_process_grounded_object_detection(
        outputs=outputs,
        threshold=0.0,
        target_sizes=target_sizes,
        text_labels=texts,
    )[0]

    boxes  = results["boxes"].cpu().numpy()  # (n_detections, 4)
    scores = results["scores"].cpu().numpy() # (n_detections,)
    labels = results["labels"]               # list of text strings e.g. "a carrot"

    log.info("OWL-ViT raw: %d detections, score range %.4f–%.4f",
             len(scores),
             scores.min() if len(scores) else 0,
             scores.max() if len(scores) else 0)

    object_results = []
    for i, name in enumerate(object_list):
        mask = np.array([int(lbl) == i for lbl in labels])
        log.debug("  query=%r  matches=%d", f"a {name}", mask.sum())
        if mask.any():
            best_score = float(scores[mask].max())
            best_box   = boxes[mask][scores[mask].argmax()].tolist()
        else:
            best_score = 0.0
            best_box   = None

        detected = best_score >= OWLVIT_THRESHOLD
        if detected and best_box is not None:
            x1, y1, x2, y2 = best_box
            box     = [int(x1), int(y1), int(x2), int(y2)]
            cx_norm = ((x1 + x2) / 2) / w
            cy_norm = ((y1 + y2) / 2) / h
        else:
            box     = None
            cx_norm = 0.5
            cy_norm = 0.5

        log.debug("  %-40s  detected=%s  score=%.3f  box=%s", name, detected, best_score, box)
        object_results.append({
            "name":     name,
            "score":    best_score,
            "detected": detected,
            "box":      box,
            "cx_norm":  cx_norm,
            "cy_norm":  cy_norm,
        })

    log.info("Object detection done in %.2fs", time.perf_counter() - t0)
    return object_results


def _box_to_attn_patch(cx_norm: float, cy_norm: float) -> tuple[int, int]:
    """Map normalized box center (0–1) to 7×7 CLIP attention patch (r, c)."""
    r = min(int(cy_norm * 7), 6)
    c = min(int(cx_norm * 7), 6)
    return r, c


def _rel_label(a: dict, b: dict) -> str:
    """Infer primary spatial relationship from normalized (cx_norm, cy_norm) coordinates."""
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
    Compute display (cx, cy) in normalized 0–1 space for each object.

    Detected objects that are close together (within 0.05) are spread in a
    small circle so labels don't overlap. Undetected objects are placed in a
    row below the main plot area (y > 1.05) so they don't obscure the scene.
    """
    positions: list[tuple[float, float]] = [(0.0, 0.0)] * len(object_locs)

    detected_indices   = [i for i, o in enumerate(object_locs) if o["detected"]]
    undetected_indices = [i for i, o in enumerate(object_locs) if not o["detected"]]

    # Group detected objects that are very close into clusters, then jitter.
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
                positions[idx] = (cx + radius * np.cos(angle), cy + radius * np.sin(angle))

    # Undetected objects: row below plot at y = 1.12, evenly spaced
    n_undet = len(undetected_indices)
    for k, idx in enumerate(undetected_indices):
        x = (k + 1) / (n_undet + 1)
        positions[idx] = (x, 1.12)

    return positions
