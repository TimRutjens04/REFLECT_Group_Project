"""
OWL-ViT v2 object detection pipeline — REFLECT / RoboFail dataset.

Reads aligned .npz files and tasks_real_world.json, runs OWL-ViT v2
(google/owlv2-base-patch16) on EVERY frame of each real-world episode,
and writes per-frame detections to owl/<episode_id>.npz.

Only episodes that resolve an object_list are processed.
Real-world episodes read from tasks_real_world.json. Sim episodes
(boilWater / makeSalad) read per-episode task.json metadata.

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
    python owl_detect.py --debug-sim        # print sim metadata lookup details
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor

ROOT = Path(__file__).resolve().parent.parent
ALIGNED_DIR = ROOT / "aligned"
OWL_DIR = ROOT / "owl"
TASKS_JSON = Path("/home/coder/datasets") / "tasks_real_world.json"
MODEL_ID = "google/owlv2-base-patch16"
THRESHOLD = 0.10
MAX_ALIASES = 5
MAX_SPIKE_FRAMES = 8
FAILURE_WINDOW = 10  # must match gui/localization.py
DEBUG_LOOKUP = False

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

## ── WordNet alias expansion (same logic as gui/localization.py) ──────────────


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


def _select_sample_frames(
    frame_deltas: np.ndarray, failure_labels: np.ndarray
) -> list[int]:
    n = len(frame_deltas)
    failure_indices = [int(i) for i, f in enumerate(failure_labels) if f]

    must_include: set[int] = {0}
    for fi in failure_indices:
        for w in range(max(0, fi - FAILURE_WINDOW), min(n, fi + FAILURE_WINDOW + 1)):
            must_include.add(w)

    threshold = frame_deltas.mean() + 1.5 * frame_deltas.std()
    spike_frames = [
        i for i in range(n) if frame_deltas[i] > threshold and i not in must_include
    ]
    spike_frames.sort(key=lambda i: -frame_deltas[i])

    return sorted(must_include | set(spike_frames[:MAX_SPIKE_FRAMES]))


# ── Core detection ────────────────────────────────────────────────────────────


def detect_episode(episode_id: str, object_list: list[str], model, processor) -> None:
    """
    Run OWL-ViT on adaptively sampled frames and save to owl/<episode_id>.npz.
    Samples: dense ±FAILURE_WINDOW around each failure + up to MAX_SPIKE_FRAMES
    high-motion frames. Mirrors the sampling in gui/localization.py.
    """
    aligned_path = ALIGNED_DIR / f"{episode_id}.npz"
    # frame_deltas need the visual embeddings; fall back to raw frame diff if absent
    encoded_path = ROOT / "encoded" / f"{episode_id}.npz"
    out_path = OWL_DIR / f"{episode_id}.npz"

    if not aligned_path.exists():
        print(f"  [skip] aligned file not found: {aligned_path}")
        return

    aligned = np.load(aligned_path, allow_pickle=False)
    frames = aligned["frames"]  # (N, H, W, 3) uint8
    failure_labels = aligned["failure_labels"]
    n = len(frames)
    n_obj = len(object_list)

    # Compute frame_deltas from encoded embeddings (same as GUI) or fall back to zeros
    if encoded_path.exists():
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
    scores = np.zeros((m, n_obj), dtype=np.float32)
    detected = np.zeros((m, n_obj), dtype=bool)
    cx_norm = np.full((m, n_obj), 0.5, dtype=np.float32)
    cy_norm = np.full((m, n_obj), 0.5, dtype=np.float32)
    boxes = np.full((m, n_obj, 4), np.nan, dtype=np.float32)

    t0 = time.perf_counter()
    for row, frame_idx in enumerate(tqdm(sample_indices, desc=episode_id, leave=False)):
        frame = frames[frame_idx]
        h, w = frame.shape[:2]
        target = torch.tensor([[h, w]], device=DEVICE)

        inputs = processor(
            text=texts, images=Image.fromarray(frame), return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=0.0,
            target_sizes=target,
            text_labels=texts,
        )[0]

        f_boxes = results["boxes"].cpu().numpy()
        f_scores = results["scores"].cpu().numpy()
        f_labels = results["labels"]

        for i in range(n_obj):
            mask = np.array([prompt_to_obj[int(lbl)] == i for lbl in f_labels])
            if mask.any():
                best = int(f_scores[mask].argmax())
                s = float(f_scores[mask][best])
                b = f_boxes[mask][best]
                scores[row, i] = s
                detected[row, i] = s >= THRESHOLD
                if s >= THRESHOLD:
                    x1, y1, x2, y2 = b
                    boxes[row, i] = [x1, y1, x2, y2]
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
    print(
        f"  saved {out_path}  ({m}/{n} frames sampled, {n_obj} objects, "
        f"{time.perf_counter() - t0:.1f}s)"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

import re as _re


def _is_sim_episode(episode_id: str) -> bool:
    return episode_id.startswith("boilWater") or episode_id.startswith("makeSalad")


def _candidate_sim_json_paths(episode_id: str) -> list[Path]:
    # Handle both boilWater-1 and boilWater1 naming variants.
    task_name = _re.sub(r"-?\d+$", "", episode_id)
    base = TASKS_JSON.parent / "sim_data"
    return [
        base / task_name / episode_id / "task.json",
        base
        / task_name
        / f"{task_name}-{episode_id[len(task_name) :].lstrip('-')}"
        / "task.json",
        base / episode_id / "task.json",
    ]


def _object_list_for(episode_id: str) -> list[str]:
    """
    Return the object_list for any episode.
    Real-world: tasks_real_world.json.
    Sim (boilWater-N, makeSalad-N): per-episode task.json.
    """
    # Real-world central metadata
    if TASKS_JSON.exists():
        with TASKS_JSON.open() as f:
            raw = json.load(f)
        obj_list = {v["general_folder_name"]: v for v in raw.values()}.get(
            episode_id, {}
        ).get("object_list", [])
        if obj_list:
            return obj_list
        if DEBUG_LOOKUP and _is_sim_episode(episode_id):
            print(
                f"    [debug] {episode_id}: not found in tasks_real_world.json (expected for sim)"
            )

    # Per-episode task.json for sim episodes
    for sim_json in _candidate_sim_json_paths(episode_id):
        if sim_json.exists():
            with sim_json.open() as f:
                obj_list = json.load(f).get("object_list", [])
            if DEBUG_LOOKUP and _is_sim_episode(episode_id):
                print(
                    f"    [debug] {episode_id}: found sim metadata at {sim_json} "
                    f"(object_list={len(obj_list)})"
                )
            return obj_list

    if DEBUG_LOOKUP and _is_sim_episode(episode_id):
        candidates = ", ".join(str(p) for p in _candidate_sim_json_paths(episode_id))
        print(f"    [debug] {episode_id}: no sim task.json found. tried: {candidates}")

    return []


def main():
    global DEBUG_LOOKUP
    OWL_DIR.mkdir(parents=True, exist_ok=True)

    # Parse args: optional episode name and/or --force flag
    args = sys.argv[1:]
    force = "--force" in args
    DEBUG_LOOKUP = "--debug-sim" in args
    args = [a for a in args if a not in {"--force", "--debug-sim"}]
    target = args[0] if args else None

    if target:
        episodes = [target]
    else:
        episodes = sorted(
            p.stem for p in ALIGNED_DIR.glob("*.npz") if _object_list_for(p.stem)
        )

    print(
        f"OWL-ViT detection: {len(episodes)} episode(s)  device={DEVICE}"
        + ("  [force]" if force else "")
    )
    if DEBUG_LOOKUP:
        print("Debug sim lookup enabled (--debug-sim)")

    run_jobs: list[tuple[str, list[str]]] = []
    skipped = 0
    for ep in episodes:
        out_path = OWL_DIR / f"{ep}.npz"
        if out_path.exists() and not force:
            print(f"  [skip] {ep} — already computed")
            skipped += 1
            continue
        obj_list = _object_list_for(ep)
        if not obj_list:
            print(f"  [skip] {ep} — no object_list in metadata")
            skipped += 1
            continue
        run_jobs.append((ep, obj_list))

    if not run_jobs:
        print(f"\nDone. {len(episodes) - skipped} computed, {skipped} skipped.")
        return

    print(f"Loading {MODEL_ID}...")
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    for ep, obj_list in run_jobs:
        print(f"  {ep}  objects={obj_list}")
        detect_episode(ep, obj_list, model, processor)

    print(f"\nDone. {len(run_jobs)} computed, {skipped} skipped.")


if __name__ == "__main__":
    main()
