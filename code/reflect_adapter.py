"""
Serialize our scene_graphs_pipeline/*.json format into the [OBSERVATION] text string
that REFLECT's reasoning-execution prompt template expects.

REFLECT's baseline format (from main/exp.py get_scene_text):
  "<obj1>, <obj2 (state)>. <subj> is <relation> <obj>. ..."

Integration:
  In a patched REFLECT main/exp.py, replace:
    scene_text = get_scene_text(get_scene_graph(event, ...))
  with:
    scene_text = scene_graph_to_observation(load_frame(episode, frame_idx))
"""

from __future__ import annotations

import json
from pathlib import Path

# Relations we emit — omit "near" (too noisy) and raw "above" when on_top_of covers it.
# Ordered by semantic specificity so the most informative sentence appears first.
_RELATION_ORDER = ["held_by_gripper", "on_top_of", "inside", "above"]

_RELATION_TEXT = {
    "held_by_gripper": "inside robot gripper",
    "on_top_of":       "on top of",
    "inside":          "inside",
    "above":           "above",
}

# Failure type → readable sentence appended after the scene description.
_FLAG_TEXT = {
    "Slip":         "The robot appears to have dropped an object (slip).",
    "Translation":  "An object has moved to an unexpected position.",
    "Wrong_object": "The robot picked up or interacted with the wrong object.",
    "No_Grasp":     "The robot failed to grasp the intended object.",
}

# States that add no information and should be omitted from object names.
_UNINFORMATIVE_STATES = {"empty", "intact", "whole", "raw", "clean", "off", "closed", "free"}


def scene_graph_to_observation(frame: dict) -> str:
    """
    Convert one frame dict from scene_graphs_pipeline/*.json to a REFLECT [OBSERVATION] string.

    Returns a string like:
      "Pot (full), Faucet (on), CounterTop. Pot is inside robot gripper. Note: slip detected."
    """
    objects: list[dict] = frame.get("objects", [])
    relations: list[dict] = frame.get("spatial_relations", [])
    flag: dict = frame.get("localization_flag", {})

    id_to_obj: dict[int, dict] = {o["id"]: o for o in objects}

    # ── 1. Object list ────────────────────────────────────────────────────────
    obj_parts: list[str] = []
    gripper_labels: list[str] = []
    for obj in objects:
        label = obj["label"]
        state = obj.get("state") or ""
        if obj.get("held_by_gripper"):
            gripper_labels.append(label)
        if state and state not in _UNINFORMATIVE_STATES:
            obj_parts.append(f"{label} ({state})")
        else:
            obj_parts.append(label)

    obj_str = ", ".join(obj_parts) + "." if obj_parts else ""

    # ── 2. Relation sentences ─────────────────────────────────────────────────
    # De-duplicate: "A is above B" and "B is above A" are both in our JSON;
    # emit only one direction (lower id first as tiebreaker).
    seen: set[tuple] = set()
    rel_sentences: list[str] = []

    # Gripper first (most informative for failure analysis)
    if gripper_labels:
        rel_sentences.append(f"{' and '.join(gripper_labels)} is inside robot gripper.")
    else:
        rel_sentences.append("nothing is inside robot gripper.")

    rel_by_type: dict[str, list[dict]] = {r: [] for r in _RELATION_ORDER[1:]}
    for rel in relations:
        rtype = rel["relation"]
        if rtype in rel_by_type:
            rel_by_type[rtype].append(rel)

    for rtype in _RELATION_ORDER[1:]:
        for rel in rel_by_type[rtype]:
            subj_id = rel["subject"]
            obj_id  = rel["object"]
            key = (min(subj_id, obj_id), max(subj_id, obj_id), rtype)
            if key in seen:
                continue
            seen.add(key)
            subj = id_to_obj.get(subj_id, {}).get("label", str(subj_id))
            obj  = id_to_obj.get(obj_id,  {}).get("label", str(obj_id))
            rel_sentences.append(f"{subj} is {_RELATION_TEXT[rtype]} {obj}.")

    rel_str = " ".join(rel_sentences)

    # ── 3. Localization flag note ─────────────────────────────────────────────
    note = ""
    if flag.get("failure_detected"):
        ftype = flag.get("type", "")
        note = " " + _FLAG_TEXT.get(ftype, f"Failure detected: {ftype}.")

    return f"{obj_str} {rel_str}{note}".strip()


# ── File I/O helpers for patching REFLECT exp.py ──────────────────────────────

def load_episode_graphs(json_path: str | Path) -> list[dict]:
    """Load all frames from a scene_graphs_pipeline/*.json file."""
    with open(json_path) as f:
        data = json.load(f)
    return data["frames"]


def get_frame_observation(json_path: str | Path, frame_idx: int) -> str:
    """
    Load the observation string for a single frame.

    Usage in patched REFLECT exp.py:
        from reflect_adapter import get_frame_observation
        obs = get_frame_observation(f"scene_graphs_pipeline/{episode}.json", step)
    """
    frames = load_episode_graphs(json_path)
    frame_map = {f["frame_idx"]: f for f in frames}
    if frame_idx not in frame_map:
        return f"No scene graph data for frame {frame_idx}."
    return scene_graph_to_observation(frame_map[frame_idx])


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "scene_graphs_pipeline/boilWater-1.json"
    frames = load_episode_graphs(path)
    fail_frames = [f for f in frames if f.get("failure_label")]
    target = fail_frames[0] if fail_frames else frames[-1]
    print(f"Frame {target['frame_idx']}:")
    print(scene_graph_to_observation(target))
