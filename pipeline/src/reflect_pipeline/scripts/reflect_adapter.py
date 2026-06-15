"""
Adapter: converts ARGUS pipeline's scene_graph.jsonl to REFLECT's SceneGraph format.

Drop-in replacement for scene_graphs_mem() in reflect.pipelines.fast_validation:

    # Before:
    local_sgs, global_sg, key_frames = scene_graphs_mem(
        events, object_list, nav_actions, interact_actions, with_audio, detected_sounds, task
    )

    # After:
    from results.reflect_adapter import load_argus_scene_graphs
    local_sgs, global_sg, key_frames = load_argus_scene_graphs(
        jsonl_path, task_detail,
        interact_actions=interact_actions,
        nav_actions=nav_actions,
    )

local_sgs  : dict {frame_id -> SceneGraph}  -- one per frame in our JSONL
global_sg  : SceneGraph                     -- last frame (final state for LLM)
key_frames : list[int]                      -- 1-indexed, same convention as REFLECT
"""

import json
from pathlib import Path

from reflect.perception.scene_graph import SceneGraph, Node, Edge

# Our relation vocab -> REFLECT edge_type strings (used by get_scene_text_util)
_RELATION_MAP = {
    "inside": "inside",
    "on_top_of": "on top of",
    "above": "above",
    "below": "below",
    "left_of": "on the left of",
    "near": "near",
}


def _label(object_id: str) -> str:
    """'apple_3' -> 'apple'  (strip the trailing _<track_id>)."""
    parts = object_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else object_id


def _row_to_sg(row: dict, task: dict) -> SceneGraph:
    """Convert one scene_graph.jsonl row -> reflect SceneGraph.

    Sets nodes and edges directly; does not call add_node/add_edge (those
    require AI2THOR event metadata we don't have).
    """
    sg = SceneGraph(event=None, task=task)
    sg.nodes = []
    sg.edges = {}

    for n in row.get("nodes", []):
        oid = n.get("object_id", "")
        if oid == "gripper":
            continue  # represented via edges only
        name = n.get("label") or _label(oid)
        node = Node(name=name, object_id=oid)
        status = n.get("status", "ok")
        if status not in ("ok",):
            node.set_state(f"{name} ({status})")
        sg.nodes.append(node)

    has_gripper_edge = False
    for e in row.get("edges", []):
        rel = e.get("relation", "")
        from_name = _label(e.get("from_object_id", ""))
        to_name = _label(e.get("to_object_id", ""))

        if rel == "held_by_gripper":
            sg.edges[(from_name, "robot gripper")] = Edge(
                Node(from_name), Node("robot gripper"), "inside"
            )
            has_gripper_edge = True
        else:
            edge_type = _RELATION_MAP.get(rel, rel)
            sg.edges[(from_name, to_name)] = Edge(
                Node(from_name), Node(to_name), edge_type
            )

    # REFLECT's summary_mem checks for "robot gripper" in edge keys;
    # always emit the "nothing in gripper" sentinel when not holding.
    if not has_gripper_edge:
        sg.edges[("nothing", "robot gripper")] = Edge(
            Node("nothing"), Node("robot gripper"), "inside"
        )

    return sg


def load_argus_scene_graphs(
    jsonl_path: str | Path,
    task: dict,
    interact_actions: dict | None = None,
    nav_actions: dict | None = None,
) -> tuple[dict, SceneGraph, list[int]]:
    """Read scene_graph.jsonl and return (local_sgs, global_sg, key_frames).

    Parameters
    ----------
    jsonl_path:
        Path to the scene_graph.jsonl produced by build_scene_graphs.py.
    task:
        Already-loaded task.json dict (same object passed to summary_mem /
        run_reasoning_mem).
    interact_actions:
        dict {step_idx -> action_str} from load_data() -- used to add action
        frames as key-frames (mirrors scene_graphs_mem logic).
    nav_actions:
        dict {(start, end) -> action_str} from load_data() -- end indices
        are added as key-frames.
    """
    rows = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No rows in {jsonl_path}")

    interact_keys: set[int] = set(interact_actions or {})
    nav_end_set: set[int] = {idx[1] for idx in (nav_actions or {}).keys()}

    local_sgs: dict[int, SceneGraph] = {}
    key_frames: list[int] = []
    prev_sg: SceneGraph | None = None

    for row in rows:
        frame_id = int(row["frame_id"])
        sg = _row_to_sg(row, task)
        local_sgs[frame_id] = sg

        kf = frame_id + 1  # 1-indexed (REFLECT convention)

        # Key-frame: scene graph changed since last frame.
        if prev_sg is None or sg != prev_sg:
            if kf not in key_frames:
                key_frames.append(kf)
            prev_sg = sg

        # Key-frame: action or nav-end frame.
        if kf in interact_keys or kf in nav_end_set:
            if kf not in key_frames:
                key_frames.append(kf)

        # Key-frame: our localization flag detected a failure.
        if row.get("localization_flag", {}).get("failure_detected"):
            if kf not in key_frames:
                key_frames.append(kf)

    # Global SG: last frame represents the final scene state for LLM reasoning.
    global_sg = local_sgs[rows[-1]["frame_id"]]

    return local_sgs, global_sg, key_frames
