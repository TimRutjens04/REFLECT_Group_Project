"""Interactive video and REFLECT-style scene-graph viewer.

Run from the repository root:

    $env:PYTHONPATH="pipeline/src"
    streamlit run pipeline/src/reflect_pipeline/scene_graph/scene_graph_viewer.py

Custom files can be passed after ``--``:

    streamlit run .../scene_graph_viewer.py -- \
        --scene-graph "example_data/scene_graph 1.jsonl" \
        --video "example_data/real_data/putAppleBowl1/videos/color.mp4"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PIPELINE_SRC = Path(__file__).resolve().parents[2]
if str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))

from reflect_pipeline.scene_graph.postprocess_scene_graph import (
    RELATION_STYLE,
    _draw_edge,
    _draw_node,
    _has_depth_warning,
    _node_label,
    add_robot_near_relations,
    graph_signature,
    load_jsonl,
    select_keyframes,
    smooth_relations,
    symbolic_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCENE_GRAPH = REPO_ROOT / "example_data" / "scene_graph 1.jsonl"
DEFAULT_VIDEO = (
    REPO_ROOT
    / "example_data"
    / "real_data"
    / "putAppleBowl1"
    / "videos"
    / "color.mp4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scene-graph", default=str(DEFAULT_SCENE_GRAPH))
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner="Loading and analysing scene graph...")
def prepare_scene_graph(
    scene_graph_path: str,
    min_stable_frames: int,
    robot_near_threshold_m: float,
    analysis_schema_version: str = "v3",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    del analysis_schema_version
    raw = load_jsonl(scene_graph_path)
    augmented = add_robot_near_relations(raw, threshold_m=robot_near_threshold_m)
    smoothed = smooth_relations(augmented, min_stable_frames=min_stable_frames)
    keyframes = select_keyframes(
        smoothed,
        max_keyframes=len(smoothed),
        max_gap_seconds=5.0,
        include_depth_triggers=True,
    )
    keyframe_reasons = {
        int(row["frame_id"]): list(row.get("keyframe_reasons", []))
        for row in keyframes
    }
    analysis = analyse_frames(smoothed, keyframe_reasons)
    return raw, smoothed, keyframes, analysis


def analyse_frames(
    rows: list[dict[str, Any]],
    keyframe_reasons: dict[int, list[str]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    previous_objects: set[str] = set()
    previous_graph: tuple[tuple[str, str, str], ...] | None = None
    previous_break_signature: tuple[Any, ...] = ()

    for index, row in enumerate(rows):
        nodes = row.get("nodes", [])
        edges = row.get("edges", [])
        object_ids = {
            str(node.get("object_id"))
            for node in nodes
            if str(node.get("object_id")) != "gripper"
        }
        missing = sorted(previous_objects - object_ids) if index else []
        appeared = sorted(object_ids - previous_objects) if index else sorted(object_ids)
        depth_warning_objects = sorted(
            str(node.get("object_id"))
            for node in nodes
            if str(node.get("object_id")) != "gripper" and _has_depth_warning(node)
        )
        occluded = sorted(
            str(node.get("object_id"))
            for node in nodes
            if str(node.get("status", "ok")) != "ok"
        )
        localization = row.get("localization_flag", {}) or {}
        localization_failure = bool(localization.get("failure_detected"))
        signature = graph_signature(row)
        relation_change = previous_graph is not None and signature != previous_graph

        break_reasons: list[str] = []
        if localization_failure:
            failure_type = localization.get("type") or "localization"
            break_reasons.append(f"localization: {failure_type}")
        if missing:
            break_reasons.append(f"missing: {', '.join(missing)}")
        if depth_warning_objects:
            break_reasons.append(f"depth: {', '.join(depth_warning_objects)}")
        if occluded:
            break_reasons.append(f"status: {', '.join(occluded)}")
        if not object_ids:
            break_reasons.append("no tracked scene objects available")
        break_signature = (
            localization.get("type") if localization_failure else None,
            tuple(missing),
            tuple(depth_warning_objects),
            tuple(occluded),
            not object_ids,
        )
        break_event = bool(break_reasons) and break_signature != previous_break_signature

        records.append(
            {
                "row_index": index,
                "frame_id": int(row.get("frame_id", index)),
                "timestamp": float(row.get("timestamp", 0.0) or 0.0),
                "object_count": len(object_ids),
                "edge_count": len(edges),
                "depth_warnings": len(depth_warning_objects),
                "localization_failure": localization_failure,
                "relation_change": relation_change,
                "is_break": bool(break_reasons),
                "break_event": break_event,
                "break_reasons": "; ".join(break_reasons),
                "missing_objects": ", ".join(missing),
                "appeared_objects": ", ".join(appeared),
                "keyframe": int(row.get("frame_id", index)) in keyframe_reasons,
                "keyframe_reasons": ", ".join(
                    keyframe_reasons.get(int(row.get("frame_id", index)), [])
                ),
            }
        )
        previous_objects = object_ids
        previous_graph = signature
        previous_break_signature = break_signature if break_reasons else ()

    return pd.DataFrame.from_records(records)


@st.cache_data
def video_metadata(video_path: str) -> dict[str, float | int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps if fps else 0.0,
        "width": width,
        "height": height,
    }


@st.cache_data(max_entries=96)
def load_video_frame(video_path: str, timestamp: float) -> np.ndarray:
    meta = video_metadata(video_path)
    frame_index = int(round(timestamp * float(meta["fps"])))
    frame_index = min(max(frame_index, 0), int(meta["frame_count"]) - 1)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read video frame {frame_index}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_video_overlay(frame_rgb: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    overlay = frame_rgb.copy()
    for node in row.get("nodes", []):
        if str(node.get("object_id")) == "gripper":
            continue
        bbox = node.get("bbox_xyxy") or []
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        warning = _has_depth_warning(node)
        color = (245, 158, 11) if warning else (34, 197, 94)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        label = _node_label(node)
        cv2.putText(
            overlay,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def render_graph_figure(
    row: dict[str, Any],
    layout: dict[str, tuple[float, float]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.axis("off")

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.035, 0.06),
            0.93,
            0.84,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            facecolor="#F0F0F0",
            edgecolor="#222222",
            linewidth=2.2,
            zorder=1,
        )
    )

    nodes = {
        str(node.get("object_id")): node
        for node in row.get("nodes", [])
        if str(node.get("object_id")) != "gripper"
    }
    visible_pos = {object_id: layout[object_id] for object_id in nodes if object_id in layout}
    robot_pos = (0.60, 0.23)
    nothing_pos = (0.22, 0.36)

    robot_shape = mpatches.RegularPolygon(
        robot_pos,
        numVertices=5,
        radius=0.055,
        orientation=np.pi / 5,
        facecolor="#CFCFCF",
        edgecolor="#222222",
        linewidth=1.9,
        zorder=7,
    )
    ax.add_patch(robot_shape)
    ax.text(
        robot_pos[0] + 0.07,
        robot_pos[1],
        "Robot",
        ha="left",
        va="center",
        fontsize=15,
        weight="bold",
    )

    held_objects: set[str] = set()
    for edge in row.get("edges", []):
        if edge.get("relation") != "held_by_gripper":
            continue
        source = str(edge.get("from_object_id", ""))
        target = str(edge.get("to_object_id", ""))
        held = source if source != "gripper" else target
        if held not in visible_pos:
            continue
        held_objects.add(held)
        _draw_edge(
            ax,
            robot_pos,
            visible_pos[held],
            "held_by_gripper",
            curve=-0.10,
            label="",
            style_key="held_by_gripper",
        )
        ax.text(
            (robot_pos[0] + visible_pos[held][0]) / 2,
            (robot_pos[1] + visible_pos[held][1]) / 2 - 0.14,
            "Holding",
            ha="center",
            va="center",
            fontsize=13,
            color=RELATION_STYLE["held_by_gripper"]["color"],
            zorder=9,
        )

    if not held_objects:
        _draw_node(ax, nothing_pos[0], nothing_pos[1], "nothing", "object", False)
        _draw_edge(
            ax,
            robot_pos,
            nothing_pos,
            "held_by_gripper",
            curve=0.08,
            label="",
            style_key="held_by_gripper",
        )
        ax.text(
            0.39,
            0.205,
            "Holding",
            ha="center",
            va="center",
            fontsize=13,
            color=RELATION_STYLE["held_by_gripper"]["color"],
            zorder=9,
        )

    for edge_index, edge in enumerate(row.get("edges", [])):
        source = str(edge.get("from_object_id", ""))
        target = str(edge.get("to_object_id", ""))
        relation = str(edge.get("relation", ""))
        if relation == "held_by_gripper":
            continue
        if relation == "near" and "gripper" in (source, target):
            object_id = target if source == "gripper" else source
            if object_id in visible_pos:
                _draw_edge(
                    ax,
                    robot_pos,
                    visible_pos[object_id],
                    "near",
                    curve=0.05,
                    label="Near",
                    style_key="near",
                    label_offset=(0.03, -0.05),
                )
            continue
        if source not in visible_pos or target not in visible_pos:
            continue

        curve = 0.12 if relation in {"inside", "on_top_of"} and edge_index % 2 == 0 else 0.0
        style_key = "inside" if relation == "near" else relation
        label_offset = (0.0, -0.055)
        if relation in {"inside", "on_top_of"}:
            label_offset = (0.03, -0.075)
        elif relation in {"above", "below"}:
            label_offset = (0.03, -0.065)
        elif relation in {"left_of", "right_of"}:
            label_offset = (0.0, -0.07)
        _draw_edge(
            ax,
            visible_pos[source],
            visible_pos[target],
            relation,
            curve=curve,
            label=relation,
            style_key=style_key,
            label_offset=label_offset,
        )

    for object_id, node in nodes.items():
        if object_id not in visible_pos:
            continue
        x, y = visible_pos[object_id]
        _draw_node(
            ax,
            x,
            y,
            _node_label(node),
            str(node.get("status", "object")),
            False,
        )
        if str(node.get("status", "ok")) != "ok":
            ax.text(
                x,
                y + 0.085,
                str(node.get("status")),
                ha="center",
                va="center",
                fontsize=10,
                color="#F2994A",
            )
        elif _has_depth_warning(node):
            ax.text(
                x + 0.08,
                y + 0.055,
                "!",
                ha="center",
                va="center",
                fontsize=11,
                weight="bold",
                color="#F2994A",
                bbox={
                    "boxstyle": "circle,pad=0.08",
                    "facecolor": "#F0F0F0",
                    "edgecolor": "#F2994A",
                },
                zorder=8,
            )

    ax.set_title(
        f"Frame {int(row.get('frame_id', 0))}  |  {float(row.get('timestamp', 0.0)):.2f}s",
        fontsize=13,
        pad=8,
    )
    fig.tight_layout(pad=0.4)
    return fig


def render_sidebar(default_scene_graph: str, default_video: str) -> tuple[str, str, int, float, str]:
    with st.sidebar:
        st.header("Scene Graph Viewer")
        scene_graph_path = st.text_input("Scene graph JSONL", value=default_scene_graph)
        video_path = st.text_input("Color video", value=default_video)
        st.divider()
        min_stable = st.slider("Relation smoothing frames", 1, 20, 4)
        st.caption(
            "A new relation must appear for this many consecutive JSON frames "
            "before replacing the currently accepted relation. At 30 FPS, 4 frames "
            "is about 0.13 seconds. This removes one-frame relation flicker."
        )
        robot_near = st.slider("Robot near threshold (m)", 0.05, 1.0, 0.40, 0.05)
        st.caption(
            "Adds Robot-Near-object only when the gripper and object are within "
            "this 3D distance. Visibility in the same image is not enough."
        )
        view_mode = st.radio(
            "Timeline filter",
            ["All frames", "Keyframes only", "Perception breaks only"],
        )
        st.caption(
            "Breaks include localization failures, missing objects, depth warnings, "
            "occlusion/uncertain status, or frames where no tracked scene objects "
            "such as the apple or bowl are available."
        )
        with st.expander("Perception and depth break rules"):
            st.markdown(
                """
The viewer marks a **perception break** when one or more of these conditions occurs:

1. `localization_flag.failure_detected` is true in the JSON.
2. A previously visible tracked object disappears.
3. An object status is not `ok`, for example `occluded` or `uncertain`.
4. No tracked scene objects are available in the frame. For example, neither
   the apple nor bowl is present in the graph. The robot/gripper alone does not
   describe the scene.
5. An object has a depth reliability warning.

A **depth warning** is raised when at least one object has:

- `any_depth_trigger = true`, or
- `depth_jump_flag = true`, or
- `depth_validity_flag = false`, or
- `depth_coherence_flag = false`.
                """
            )
    return scene_graph_path, video_path, min_stable, robot_near, view_mode


def shift_timeline(slider_key: str, delta: int, maximum: int) -> None:
    current = int(st.session_state.get(slider_key, 0))
    st.session_state[slider_key] = min(max(current + delta, 0), maximum)


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="RGB-D Scene Graph Explorer", layout="wide")
    st.title("RGB-D Video and REFLECT-Style Scene Graph Explorer")

    scene_graph_path, video_path, min_stable, robot_near, view_mode = render_sidebar(
        args.scene_graph,
        args.video,
    )

    if not Path(scene_graph_path).exists():
        st.error(f"Scene graph file not found: {scene_graph_path}")
        st.stop()
    if not Path(video_path).exists():
        st.error(f"Video file not found: {video_path}")
        st.stop()

    _, rows, _, analysis = prepare_scene_graph(
        scene_graph_path,
        min_stable,
        robot_near,
        "v3",
    )
    meta = video_metadata(video_path)
    layout = symbolic_layout(rows)

    if view_mode == "Keyframes only":
        candidate_indices = analysis.loc[analysis["keyframe"], "row_index"].astype(int).tolist()
    elif view_mode == "Perception breaks only":
        candidate_indices = analysis.loc[analysis["break_event"], "row_index"].astype(int).tolist()
    else:
        candidate_indices = analysis["row_index"].astype(int).tolist()

    if not candidate_indices:
        st.warning(f"No frames match the filter: {view_mode}")
        st.stop()

    slider_key = f"timeline_{view_mode}_{len(candidate_indices)}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = 0
    selected_position = st.slider(
        "Timeline",
        min_value=0,
        max_value=len(candidate_indices) - 1,
        key=slider_key,
        help="Drag through the full sequence or the currently filtered frame set.",
    )
    row_index = candidate_indices[selected_position]
    row = rows[row_index]
    current = analysis.iloc[row_index]

    navigation_left, navigation_mid, navigation_right = st.columns([1, 2, 1])
    with navigation_left:
        st.button(
            "Previous",
            disabled=selected_position == 0,
            use_container_width=True,
            on_click=shift_timeline,
            args=(slider_key, -1, len(candidate_indices) - 1),
        )
    with navigation_mid:
        st.markdown(
            f"<div style='text-align:center'><b>{view_mode}</b> &nbsp; "
            f"{selected_position + 1}/{len(candidate_indices)} &nbsp; | &nbsp; "
            f"frame {int(row['frame_id'])} &nbsp; | &nbsp; "
            f"{float(row['timestamp']):.2f}s</div>",
            unsafe_allow_html=True,
        )
    with navigation_right:
        st.button(
            "Next",
            disabled=selected_position == len(candidate_indices) - 1,
            use_container_width=True,
            on_click=shift_timeline,
            args=(slider_key, 1, len(candidate_indices) - 1),
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Objects", int(current["object_count"]))
    metric_cols[1].metric("Relations", int(current["edge_count"]))
    metric_cols[2].metric("Depth warnings", int(current["depth_warnings"]))
    metric_cols[3].metric("Video FPS", f"{float(meta['fps']):.1f}")

    if bool(current["is_break"]):
        st.error(f"Perception break: {current['break_reasons']}")
    elif bool(current["keyframe"]):
        st.info(f"Keyframe: {current['keyframe_reasons']}")
    else:
        st.success("No perception break flagged at this frame.")

    video_col, graph_col = st.columns(2, gap="large")
    with video_col:
        frame = load_video_frame(video_path, float(row["timestamp"]))
        frame = draw_video_overlay(frame, row)
        st.image(frame, caption="Video frame with tracked-object boxes", use_container_width=True)
    with graph_col:
        graph_figure = render_graph_figure(row, layout)
        st.pyplot(graph_figure, use_container_width=True)
        plt.close(graph_figure)

    breaks_tab, json_tab = st.tabs(["Where perception breaks", "Current JSON"])

    with breaks_tab:
        breaks = analysis.loc[
            analysis["break_event"],
            [
                "frame_id",
                "timestamp",
                "break_reasons",
                "missing_objects",
                "keyframe_reasons",
            ],
        ].copy()
        st.dataframe(breaks, use_container_width=True, hide_index=True)
        metric_left, metric_right = st.columns(2)
        metric_left.metric("Perception break events", len(breaks))
        metric_right.metric("Frames carrying a warning", int(analysis["is_break"].sum()))

    with json_tab:
        st.code(json.dumps(row, indent=2, ensure_ascii=False), language="json")


if __name__ == "__main__":
    main()
