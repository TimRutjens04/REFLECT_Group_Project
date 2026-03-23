"""
OWL-ViT v2 object detection pipeline — REFLECT / RoboFail dataset.

Reads aligned .npz files and tasks_real_world.json, runs OWL-ViT v2
(google/owlv2-base-patch16) on EVERY frame of each real-world episode,
and writes per-frame detections to owl/<episode_id>.npz.

Only episodes that have an object_list in tasks_real_world.json are processed.
Simulated episodes (boilWater / makeSalad) are skipped — they have no object list.

=== OUTPUT FORMAT ===

  owl/<episode_id>.npz
    scores     (N, n_obj)    float32 — raw OWL-ViT confidence per frame per object
    detected   (N, n_obj)    bool    — True if score >= THRESHOLD
    cx_norm    (N, n_obj)    float32 — box centre x, normalised 0–1 (0.5 if undetected)
    cy_norm    (N, n_obj)    float32 — box centre y, normalised 0–1 (0.5 if undetected)
    boxes      (N, n_obj, 4) float32 — [x1,y1,x2,y2] pixel coords, NaN if undetected

  Object column order matches the object_list in tasks_real_world.json.
  Load object names from there (keyed by episode_id / general_folder_name).

Usage:
    python owl_detect.py                    # process all episodes
    python owl_detect.py appleInFridge1     # single episode by name
"""

import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALIGNED_DIR = os.path.join(ROOT, "aligned")
OWL_DIR     = os.path.join(ROOT, "owl")
TASKS_JSON  = os.path.join(ROOT, "data", "tasks_real_world.json")
MODEL_ID    = "google/owlv2-base-patch16"
THRESHOLD      = 0.10
MAX_ALIASES    = 5
MAX_SPIKE_FRAMES = 8
FAILURE_WINDOW   = 10   # must match gui/localization.py

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── WordNet alias expansion (same logic as gui/localization.py) ──────────────

def _wordnet_aliases(name: str) -> list[str]:
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
        return len(term) < 30 and not any(c.isupper() for c in term[1:])

    lookup_terms = [name]
    words = name.split()
    if len(words) > 1:
        lookup_terms.append(words[-1])

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


# ── Adaptive frame sampling (mirrors gui/localization.py) ────────────────────

def _select_sample_frames(frame_deltas: np.ndarray,
                           failure_labels: np.ndarray) -> list[int]:
    n = len(frame_deltas)
    failure_indices = [int(i) for i, f in enumerate(failure_labels) if f]

    must_include: set[int] = {0}
    for fi in failure_indices:
        for w in range(max(0, fi - FAILURE_WINDOW), min(n, fi + FAILURE_WINDOW + 1)):
            must_include.add(w)

    threshold    = frame_deltas.mean() + 1.5 * frame_deltas.std()
    spike_frames = [i for i in range(n)
                    if frame_deltas[i] > threshold and i not in must_include]
    spike_frames.sort(key=lambda i: -frame_deltas[i])

    return sorted(must_include | set(spike_frames[:MAX_SPIKE_FRAMES]))


# ── Core detection ────────────────────────────────────────────────────────────

def detect_episode(episode_id: str, object_list: list[str],
                   model, processor) -> None:
    """
    Run OWL-ViT on adaptively sampled frames and save to owl/<episode_id>.npz.
    Samples: dense ±FAILURE_WINDOW around each failure + up to MAX_SPIKE_FRAMES
    high-motion frames. Mirrors the sampling in gui/localization.py.
    """
    aligned_path = os.path.join(ALIGNED_DIR, f"{episode_id}.npz")
    encoded_path = os.path.join(aligned_path.replace("aligned/", "encoded/"))
    # frame_deltas need the visual embeddings; fall back to raw frame diff if absent
    encoded_path = os.path.join(os.path.dirname(ALIGNED_DIR), "encoded", f"{episode_id}.npz")
    out_path     = os.path.join(OWL_DIR, f"{episode_id}.npz")

    if not os.path.exists(aligned_path):
        print(f"  [skip] aligned file not found: {aligned_path}")
        return

    aligned        = np.load(aligned_path, allow_pickle=False)
    frames         = aligned["frames"]          # (N, H, W, 3) uint8
    failure_labels = aligned["failure_labels"]
    n              = len(frames)
    n_obj          = len(object_list)

    # Compute frame_deltas from encoded embeddings (same as GUI) or fall back to zeros
    if os.path.exists(encoded_path):
        enc = np.load(encoded_path, allow_pickle=False)
        vis = enc["visual_embeddings"].astype(np.float32)
        frame_deltas = np.zeros(n, dtype=np.float32)
        for i in range(1, n):
            frame_deltas[i] = 1.0 - float(np.dot(vis[i], vis[i - 1]))
    else:
        frame_deltas = np.zeros(n, dtype=np.float32)

    sample_indices = _select_sample_frames(frame_deltas, failure_labels)
    m = len(sample_indices)
    print(f"    sampling {m}/{n} frames  (failure window + spikes)")

    # Build text prompts with WordNet expansion
    all_prompts: list[str] = []
    prompt_to_obj: list[int] = []
    for i, obj_name in enumerate(object_list):
        for alias in _wordnet_aliases(obj_name):
            all_prompts.append(alias)
            prompt_to_obj.append(i)
    texts = [all_prompts]

    # Output arrays — one row per sampled frame (M, not N)
    scores   = np.zeros((m, n_obj),    dtype=np.float32)
    detected = np.zeros((m, n_obj),    dtype=bool)
    cx_norm  = np.full( (m, n_obj),    0.5, dtype=np.float32)
    cy_norm  = np.full( (m, n_obj),    0.5, dtype=np.float32)
    boxes    = np.full( (m, n_obj, 4), np.nan, dtype=np.float32)

    t0 = time.perf_counter()
    for row, frame_idx in enumerate(tqdm(sample_indices, desc=episode_id, leave=False)):
        frame  = frames[frame_idx]
        h, w   = frame.shape[:2]
        target = torch.tensor([[h, w]], device=DEVICE)

        inputs = processor(text=texts, images=Image.fromarray(frame),
                           return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=0.0,
            target_sizes=target,
            text_labels=texts,
        )[0]

        f_boxes  = results["boxes"].cpu().numpy()
        f_scores = results["scores"].cpu().numpy()
        f_labels = results["labels"]

        for i in range(n_obj):
            mask = np.array([prompt_to_obj[int(lbl)] == i for lbl in f_labels])
            if mask.any():
                best = int(f_scores[mask].argmax())
                s    = float(f_scores[mask][best])
                b    = f_boxes[mask][best]
                scores[row, i]   = s
                detected[row, i] = s >= THRESHOLD
                if s >= THRESHOLD:
                    x1, y1, x2, y2 = b
                    boxes[row, i]   = [x1, y1, x2, y2]
                    cx_norm[row, i] = ((x1 + x2) / 2) / w
                    cy_norm[row, i] = ((y1 + y2) / 2) / h

    np.savez_compressed(
        out_path,
        sample_indices=np.array(sample_indices, dtype=np.int32),
        scores=scores,
        detected=detected,
        cx_norm=cx_norm,
        cy_norm=cy_norm,
        boxes=boxes,
    )
    print(f"  saved {out_path}  ({m}/{n} frames sampled, {n_obj} objects, "
          f"{time.perf_counter()-t0:.1f}s)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(OWL_DIR, exist_ok=True)

    # Load task metadata — only real-world episodes have object lists
    if not os.path.exists(TASKS_JSON):
        print(f"ERROR: {TASKS_JSON} not found")
        sys.exit(1)
    with open(TASKS_JSON) as f:
        tasks_raw = json.load(f)
    meta = {v["general_folder_name"]: v for v in tasks_raw.values()}

    # Determine which episodes to process
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        episodes = [target]
    else:
        episodes = sorted(
            ep for ep in meta
            if meta[ep].get("object_list")
            and os.path.exists(os.path.join(ALIGNED_DIR, f"{ep}.npz"))
        )

    print(f"OWL-ViT detection: {len(episodes)} episode(s)  device={DEVICE}")
    print(f"Loading {MODEL_ID}...")
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model     = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    skipped = 0
    for ep in episodes:
        out_path = os.path.join(OWL_DIR, f"{ep}.npz")
        if os.path.exists(out_path):
            print(f"  [skip] {ep} — already computed")
            skipped += 1
            continue
        obj_list = meta.get(ep, {}).get("object_list", [])
        if not obj_list:
            print(f"  [skip] {ep} — no object_list in metadata")
            skipped += 1
            continue
        print(f"  {ep}  objects={obj_list}")
        detect_episode(ep, obj_list, model, processor)

    print(f"\nDone. {len(episodes) - skipped} computed, {skipped} skipped.")


if __name__ == "__main__":
    main()
