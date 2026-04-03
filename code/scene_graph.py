"""
Ground-truth scene graph extraction and video rendering for simulated episodes.

Each step_N.pickle is an ai2thor.server.Event snapshot.  We pull task-relevant
object states and containment edges, render a networkx graph side-by-side with
the RGB frame, and write an MP4.

Usage:
    poetry run python3 code/scene_graph.py boilWater-1
    poetry run python3 code/scene_graph.py boilWater-1 --out scene_graphs/bw1.mp4
    poetry run python3 code/scene_graph.py --all        # all sim episodes

Output:
    scene_graphs/<episode_id>.mp4   — video (1 fps, matching aligned frames)
"""

import argparse
import io
import json
import os
import pickle
import re

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "scene_graphs")

# Visual style — matches dark GUI theme
DARK_BG = "#0e1117"
GRID_COL = "#1e2130"

C_AGENT = "#c87ee8"    # purple
C_SURFACE = "#2e2e3e"  # dark, surface objects (receptacles)
C_NORMAL = "#4c9be8"   # blue, default pickupable
C_HELD = "#f0c040"     # yellow, in Agent's hand
C_TOGGLED = "#7ec87e"  # green, toggled on
C_FILLED = "#5aade8"   # cyan, filled with liquid
C_BROKEN = "#e8754c"   # orange-red, broken

# Surface object types that act as receptacles
SURFACE_TYPES = frozenset({
    "CounterTop", "Sink", "SinkBasin", "StoveBurner",
    "Floor", "Shelf", "Cabinet", "Drawer", "Fridge", "Plate",
})

# Map objectType aliases that appear in parentReceptacles → canonical task type
_SINK_ALIAS = {"SinkBasin": "Sink"}


# ── Data extraction ──────────────────────────────────────────────────────────

def _load_task_json(episode_id: str) -> dict:
    task_name = re.sub(r"-\d+$", "", episode_id)
    path = os.path.join(DATA_DIR, task_name, episode_id, "task.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _build_name_map(all_objects: list[dict], unity_name_map: dict) -> dict[str, str]:
    """objectId → display name.  Multi-instance types use unity_name_map."""
    # Count instances per objectType
    by_type: dict[str, list[dict]] = {}
    for obj in all_objects:
        by_type.setdefault(obj["objectType"], []).append(obj)

    result: dict[str, str] = {}
    for otype, objs in by_type.items():
        if len(objs) == 1:
            result[objs[0]["objectId"]] = otype
        else:
            for obj in objs:
                raw = obj["name"]
                if raw in unity_name_map:
                    result[obj["objectId"]] = unity_name_map[raw]
                else:
                    # Fallback: truncate UUID portion
                    result[obj["objectId"]] = f"{otype}_{raw[-4:]}"
    return result


def _parent_display_name(parent_oid: str, name_map: dict[str, str]) -> str | None:
    """
    Resolve a parentReceptacles objectId string to a display name.
    objectId format: 'TypeName|x|y|z' or 'TypeName|x|y|z|SubType'.
    """
    if parent_oid in name_map:
        return name_map[parent_oid]
    otype = parent_oid.split("|")[0]
    # Alias resolution (SinkBasin → Sink)
    otype = _SINK_ALIAS.get(otype, otype)
    # Find the first object with matching type in name_map values
    for oid, dname in name_map.items():
        if dname == otype or dname.startswith(f"{otype}-") or dname.startswith(f"{otype}_"):
            return dname
    return otype  # best-effort


def extract_step_graphs(episode_id: str) -> list[dict]:
    """
    Load all step pickles for a sim episode and return a list of graph dicts
    (one per aligned frame, 0-indexed).

    Each dict:
        step          int
        last_action   str
        success       bool
        nodes         list[dict]   — id, name, tier, state flags
        edges         list[dict]   — src, dst, rel
        rgb_frame     np.ndarray   (H, W, 3) uint8
        failure       bool
    """
    task = _load_task_json(episode_id)
    object_list: list[str] = task.get("object_list", [])
    unity_name_map: dict[str, str] = task.get("unity_name_map", {})

    task_name = re.sub(r"-\d+$", "", episode_id)
    events_dir = os.path.join(DATA_DIR, task_name, episode_id, "events")
    if not os.path.isdir(events_dir):
        raise FileNotFoundError(f"No events dir: {events_dir}")

    step_files = sorted(
        [f for f in os.listdir(events_dir) if f.endswith(".pickle")],
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )

    # Also load failure labels from aligned npz for ground truth
    aligned_path = os.path.join(ROOT, "aligned", f"{episode_id}.npz")
    failure_labels: list[bool] = []
    if os.path.exists(aligned_path):
        d = np.load(aligned_path, allow_pickle=False)
        failure_labels = d["failure_labels"].tolist()

    graphs = []
    name_map: dict[str, str] | None = None  # built from first step, reused

    for i, fname in enumerate(step_files):
        step_num = int(fname.split("_")[1].split(".")[0])
        with open(os.path.join(events_dir, fname), "rb") as f:
            ev = pickle.load(f)

        meta = ev.metadata
        all_objects: list[dict] = meta.get("objects", [])

        # Build name_map once from the first step (object set is stable)
        if name_map is None:
            name_map = _build_name_map(all_objects, unity_name_map)

        # Filter to task-relevant objectTypes (+ always include Floor)
        task_types = set(object_list) | {"Floor"}
        relevant_oids = {
            obj["objectId"]: obj
            for obj in all_objects
            if obj["objectType"] in task_types
        }

        # ── Nodes ────────────────────────────────────────────────────────────
        nodes: list[dict] = []

        # Agent node
        agent_meta = meta.get("agent", {})
        nodes.append({
            "id": "Agent",
            "name": "Agent",
            "objectType": "Agent",
            "tier": "agent",
            "is_picked_up": False,
            "is_toggled": False,
            "is_filled": False,
            "is_broken": False,
            "visible": True,
            "position": agent_meta.get("position", {}),
        })

        for oid, obj in relevant_oids.items():
            otype = obj["objectType"]
            tier = "surface" if otype in SURFACE_TYPES else "object"
            nodes.append({
                "id": oid,
                "name": name_map.get(oid, otype),
                "objectType": otype,
                "tier": tier,
                "is_picked_up": bool(obj.get("isPickedUp", False)),
                "is_toggled": bool(obj.get("isToggled", False)),
                "is_filled": bool(obj.get("isFilledWithLiquid", False)),
                "is_broken": bool(obj.get("isBroken", False)),
                "visible": bool(obj.get("visible", False)),
                "position": obj.get("position", {}),
            })

        # ── Edges ────────────────────────────────────────────────────────────
        node_ids = {n["id"] for n in nodes}
        node_names = {n["name"] for n in nodes}
        edges: list[dict] = []

        for oid, obj in relevant_oids.items():
            if obj.get("isPickedUp"):
                edges.append({"src": "Agent", "dst": oid, "rel": "holds"})
                continue
            parents = obj.get("parentReceptacles") or []
            for parent_oid in parents:
                pname = _parent_display_name(parent_oid, name_map)
                # Find the node id for this parent
                parent_node_id = None
                if parent_oid in node_ids:
                    parent_node_id = parent_oid
                else:
                    # Match by display name
                    for n in nodes:
                        if n["name"] == pname:
                            parent_node_id = n["id"]
                            break
                if parent_node_id and parent_node_id != oid:
                    edges.append({"src": oid, "dst": parent_node_id, "rel": "on/in"})
                    break  # one primary parent edge is enough

        failure = failure_labels[i] if i < len(failure_labels) else False
        graphs.append({
            "step": step_num,
            "last_action": meta.get("lastAction", ""),
            "success": bool(meta.get("lastActionSuccess", True)),
            "nodes": nodes,
            "edges": edges,
            "rgb_frame": ev.frame,  # (H, W, 3) uint8
            "failure": failure,
        })

    return graphs


# ── Layout (fixed across all steps for smooth animation) ────────────────────

def _compute_layout(graphs: list[dict]) -> dict[str, tuple[float, float]]:
    """
    Assign fixed (x, y) positions to all node ids seen across all steps.
    Layout tiers:
      agent   → y = 0.88
      object  → y = 0.52
      surface → y = 0.15
    Nodes within each tier are spread evenly on x.
    """
    tier_order = {"agent": 2, "object": 1, "surface": 0}
    tier_y = {"agent": 0.88, "object": 0.52, "surface": 0.15}

    # Collect all unique nodes across steps (id, name, tier)
    seen: dict[str, dict] = {}
    for g in graphs:
        for n in g["nodes"]:
            if n["id"] not in seen:
                seen[n["id"]] = n

    # Group by tier, stable sort by name within tier
    by_tier: dict[str, list[dict]] = {"agent": [], "object": [], "surface": []}
    for n in seen.values():
        tier = n.get("tier", "surface")
        by_tier.setdefault(tier, []).append(n)
    for tier in by_tier:
        by_tier[tier].sort(key=lambda n: n["name"])

    positions: dict[str, tuple[float, float]] = {}
    for tier, nodes in by_tier.items():
        y = tier_y.get(tier, 0.5)
        count = len(nodes)
        for k, n in enumerate(nodes):
            x = (k + 1) / (count + 1)
            positions[n["id"]] = (x, y)
    return positions


# ── Rendering ────────────────────────────────────────────────────────────────

def _node_color(node: dict) -> str:
    if node["objectType"] == "Agent":
        return C_AGENT
    if node["tier"] == "surface":
        return C_SURFACE
    if node["is_picked_up"]:
        return C_HELD
    if node["is_broken"]:
        return C_BROKEN
    if node["is_filled"]:
        return C_FILLED
    if node["is_toggled"]:
        return C_TOGGLED
    return C_NORMAL


def render_graph_frame(
    graph: dict,
    layout: dict[str, tuple[float, float]],
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    """
    Render one step as a matplotlib figure and return as (H, W, 3) uint8 array.
    Layout: RGB frame (left 50%) | scene graph (right 50%).
    """
    fig, (ax_rgb, ax_graph) = plt.subplots(
        1, 2, figsize=(14, 7),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    fig.patch.set_facecolor(DARK_BG)

    # ── Left: RGB frame ──────────────────────────────────────────────────────
    ax_rgb.set_facecolor(DARK_BG)
    ax_rgb.axis("off")
    ax_rgb.imshow(graph["rgb_frame"])

    failure = graph["failure"]
    border_color = "#cc3333" if failure else "#2a7a2a"
    for spine in ax_rgb.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)

    status = "⚠ FAILURE" if failure else "✓ Normal"
    status_color = "#ff6666" if failure else "#66cc66"
    ax_rgb.set_title(
        f"Step {graph['step']}  |  {status}  |  t = {frame_idx}s",
        color=status_color, fontsize=11, pad=6,
    )
    action_str = graph["last_action"]
    success_str = "" if graph["success"] else " [FAILED]"
    ax_rgb.set_xlabel(
        f"Action: {action_str}{success_str}",
        color="#aaaaaa", fontsize=8,
    )

    # ── Right: scene graph ───────────────────────────────────────────────────
    ax_graph.set_facecolor(DARK_BG)
    ax_graph.set_xlim(-0.05, 1.05)
    ax_graph.set_ylim(-0.05, 1.05)
    ax_graph.axis("off")
    ax_graph.set_title("Ground-truth scene graph", color="white", fontsize=11, pad=6)

    # Tier separator lines
    for y_sep in (0.35, 0.70):
        ax_graph.axhline(y_sep, color=GRID_COL, lw=1.0, linestyle="--", zorder=0)

    # Tier labels
    for label, y_pos in [("surfaces / receptacles", 0.22), ("objects", 0.55), ("agent", 0.92)]:
        ax_graph.text(
            1.02, y_pos, label,
            color="#666677", fontsize=7, va="center", ha="left",
            transform=ax_graph.transAxes,
        )

    # Build networkx graph for this step
    G = nx.DiGraph()
    node_by_id = {n["id"]: n for n in graph["nodes"]}

    for n in graph["nodes"]:
        if n["id"] in layout:
            G.add_node(n["id"])

    for e in graph["edges"]:
        if e["src"] in layout and e["dst"] in layout:
            G.add_edge(e["src"], e["dst"], rel=e["rel"])

    pos = {nid: layout[nid] for nid in G.nodes if nid in layout}

    # Draw edges
    holds_edges = [(u, v) for u, v, d in G.edges(data=True) if d["rel"] == "holds"]
    onin_edges  = [(u, v) for u, v, d in G.edges(data=True) if d["rel"] == "on/in"]

    nx.draw_networkx_edges(
        G, pos, edgelist=holds_edges, ax=ax_graph,
        edge_color=C_HELD, style="dashed", width=2.0, alpha=0.85,
        arrows=True, arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=onin_edges, ax=ax_graph,
        edge_color="#555566", style="solid", width=1.5, alpha=0.7,
        arrows=True, arrowsize=12,
    )

    # Edge labels
    edge_labels = {(u, v): d["rel"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax_graph,
        font_size=6, font_color="#888899",
        bbox=dict(facecolor=DARK_BG, edgecolor="none", pad=1),
    )

    # Draw nodes
    for nid in G.nodes:
        n = node_by_id[nid]
        x, y = pos[nid]
        color = _node_color(n)
        ec = "#ff4444" if failure and n["tier"] == "object" else "white"
        lw = 2.0 if failure and n["tier"] == "object" else 0.8

        circle = plt.Circle(
            (x, y), 0.055,
            color=color, ec=ec, lw=lw, zorder=3, alpha=0.95,
        )
        ax_graph.add_patch(circle)

        # Short display name
        name = n["name"]
        short = name if len(name) <= 12 else name[:10] + ".."
        ax_graph.text(
            x, y, short,
            ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold",
            zorder=4,
        )

        # State badges below the node
        badges = []
        if n["is_picked_up"]:
            badges.append("held")
        if n["is_toggled"]:
            badges.append("on")
        if n["is_filled"]:
            badges.append("filled")
        if n["is_broken"]:
            badges.append("broken")
        if badges:
            ax_graph.text(
                x, y - 0.085, " · ".join(badges),
                ha="center", va="top", fontsize=5.5,
                color="#cccccc", zorder=4,
            )

    # Legend
    legend_patches = [
        mpatches.Patch(color=C_AGENT,   label="Agent"),
        mpatches.Patch(color=C_NORMAL,  label="Object (normal)"),
        mpatches.Patch(color=C_HELD,    label="Object (held)"),
        mpatches.Patch(color=C_TOGGLED, label="Object (on)"),
        mpatches.Patch(color=C_FILLED,  label="Object (filled)"),
        mpatches.Patch(color=C_BROKEN,  label="Object (broken)"),
        mpatches.Patch(color=C_SURFACE, label="Surface / receptacle"),
    ]
    ax_graph.legend(
        handles=legend_patches, loc="lower left",
        facecolor=DARK_BG, labelcolor="white",
        fontsize=6, framealpha=0.85,
    )

    # Progress bar
    progress = (frame_idx + 1) / total_frames
    ax_graph.axhline(-0.03, xmin=0, xmax=progress,
                     color="#4c9be8", lw=3, solid_capstyle="butt", zorder=5)

    fig.tight_layout(pad=0.6)

    # Render to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=DARK_BG)
    buf.seek(0)
    plt.close(fig)

    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR
    return frame


# ── Video export ─────────────────────────────────────────────────────────────

def save_video(episode_id: str, output_path: str | None = None, fps: int = 1) -> str:
    """
    Generate the scene graph video for a sim episode.
    Returns the path to the written MP4.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(OUT_DIR, f"{episode_id}.mp4")

    print(f"Extracting scene graphs for {episode_id} …")
    graphs = extract_step_graphs(episode_id)
    layout = _compute_layout(graphs)
    n = len(graphs)

    # Probe frame size from first rendered frame
    probe = render_graph_frame(graphs[0], layout, 0, n)
    h, w = probe.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    writer.write(probe)
    for i, graph in enumerate(tqdm(graphs[1:], desc="Rendering frames", unit="fr"), start=1):
        frame = render_graph_frame(graph, layout, i, n)
        writer.write(frame)

    writer.release()
    print(f"Saved → {output_path}  ({n} frames, {fps} fps)")
    return output_path


def sim_episode_ids() -> list[str]:
    """All sim episode IDs (boilWater-N, makeSalad-N) found under data/."""
    ids = []
    for task in os.listdir(DATA_DIR):
        task_dir = os.path.join(DATA_DIR, task)
        if not os.path.isdir(task_dir):
            continue
        for ep in os.listdir(task_dir):
            ep_dir = os.path.join(task_dir, ep)
            events_dir = os.path.join(ep_dir, "events")
            if os.path.isdir(events_dir):
                ids.append(ep)
    return sorted(ids)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", nargs="?", help="Episode ID, e.g. boilWater-1")
    parser.add_argument("--out", help="Output path (default: scene_graphs/<episode>.mp4)")
    parser.add_argument("--all", action="store_true", help="Process all sim episodes")
    parser.add_argument("--fps", type=int, default=1, help="Video FPS (default: 1)")
    args = parser.parse_args()

    if args.all:
        for ep in sim_episode_ids():
            save_video(ep, fps=args.fps)
    elif args.episode:
        save_video(args.episode, args.out, fps=args.fps)
    else:
        parser.print_help()
        sys.exit(1)
