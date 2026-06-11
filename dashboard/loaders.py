"""Parse JSONL logs and ground-truth JSON into pandas DataFrames.

All parsers are decorated with @st.cache_data; cache key = path + mtime.
"""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@st.cache_data(hash_funcs={str: lambda p: (p, _mtime(p))})
def load_detection(path: str) -> pd.DataFrame:
    """Returns one row per frame. Detections list stored as-is in 'detections' column."""
    rows = _read_jsonl(path)
    records = []
    for r in rows:
        records.append({
            "frame_id": r["frame_id"],
            "timestamp": r.get("timestamp"),
            "detector_ran": r.get("detector_ran", False),
            "trigger_reason": r.get("trigger_reason", "none"),
            "prompts_used": r.get("prompts_used", []),
            "detections": r.get("detections", []),
            "detection_count": len(r.get("detections", [])),
            "detection_success": r.get("detection_success", False),
            "failure_mode": r.get("failure_mode"),
            "runtime_ms": r.get("runtime_ms", 0.0),
        })
    return pd.DataFrame(records).sort_values("frame_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

@st.cache_data(hash_funcs={str: lambda p: (p, _mtime(p))})
def load_tracking(path: str) -> pd.DataFrame:
    """Returns one row per (frame_id, object_id)."""
    rows = _read_jsonl(path)
    records = []
    for r in rows:
        flags = r.get("flags", {})
        for obj in r.get("tracked_objects", []):
            records.append({
                "frame_id": r["frame_id"],
                "timestamp": r.get("timestamp"),
                "object_id": obj["object_id"],
                "bbox_xyxy": obj.get("bbox_xyxy"),
                "bbox_area_px": obj.get("bbox_area_px"),
                "bbox_area_ratio_to_init": obj.get("bbox_area_ratio_to_init"),
                "center_xy": obj.get("center_xy"),
                "displacement_px": obj.get("displacement_px"),
                "tracker_confidence": obj.get("tracker_confidence"),
                "tracker_status": obj.get("tracker_status", "ok"),
                "frames_since_redetect": obj.get("frames_since_redetect"),
                "bbox_size_change_flag": flags.get("bbox_size_change_flag", False),
                "drift_flag": flags.get("drift_flag", False),
                "frame_counter_K_flag": flags.get("frame_counter_K_flag", False),
                "any_recovery_trigger": flags.get("any_recovery_trigger", False),
            })
    return pd.DataFrame(records).sort_values(["frame_id", "object_id"]).reset_index(drop=True)


@st.cache_data(hash_funcs={str: lambda p: (p, _mtime(p))})
def load_tracking_frame_level(path: str) -> pd.DataFrame:
    """Returns one row per frame with tracked_object count and flags."""
    rows = _read_jsonl(path)
    records = []
    for r in rows:
        flags = r.get("flags", {})
        records.append({
            "frame_id": r["frame_id"],
            "timestamp": r.get("timestamp"),
            "tracked_count": len(r.get("tracked_objects", [])),
            "bbox_size_change_flag": flags.get("bbox_size_change_flag", False),
            "drift_flag": flags.get("drift_flag", False),
            "frame_counter_K_flag": flags.get("frame_counter_K_flag", False),
            "any_recovery_trigger": flags.get("any_recovery_trigger", False),
        })
    return pd.DataFrame(records).sort_values("frame_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@st.cache_data(hash_funcs={str: lambda p: (p, _mtime(p))})
def load_validation(path: str) -> pd.DataFrame:
    """One row per (frame_id, object_id) from {seq}__validation.jsonl.
    Returns empty DataFrame if the file does not exist or is empty."""
    if not Path(path).exists():
        return pd.DataFrame()
    rows = _read_jsonl(path)
    records = []
    for r in rows:
        for obj in r.get("tracked_objects", []):
            flags = obj.get("flags") or {}
            records.append({
                "frame_id":             r["frame_id"],
                "timestamp":            r.get("timestamp"),
                "object_id":            obj["object_id"],
                "label":                obj.get("label"),
                "bbox_xyxy":            obj.get("bbox_xyxy"),
                "tracker_confidence":   obj.get("tracker_confidence"),
                "tracker_status":       obj.get("tracker_status", "ok"),
                "last_detection_frame": obj.get("last_detection_frame"),
                "bbox_size_change_flag": bool(flags.get("bbox_size_change_flag", False)),
                "drift_flag":           bool(flags.get("drift_flag", False)),
                "recovery_trigger":     bool(flags.get("recovery_trigger", False)),
            })
    return (
        pd.DataFrame(records)
        .sort_values(["frame_id", "object_id"])
        .reset_index(drop=True)
        if records else pd.DataFrame()
    )


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------

@st.cache_data(hash_funcs={str: lambda p: (p, _mtime(p))})
def load_depth(path: str) -> pd.DataFrame:
    """Returns one row per (frame_id, object_id)."""
    rows = _read_jsonl(path)
    records = []
    for r in rows:
        for obj in r.get("per_object_depth", []):
            records.append({
                "frame_id":                r["frame_id"],
                "timestamp":               r.get("timestamp"),
                "object_id":               obj["object_id"],
                "label":                   obj.get("label"),
                "last_detection_frame":    obj.get("last_detection_frame"),
                "valid_depth_pixel_ratio": obj.get("valid_depth_pixel_ratio"),
                "depth_median_m":          obj.get("depth_median_m"),
                "depth_mean_m":            obj.get("depth_mean_m"),
                "depth_std_m":             obj.get("depth_std_m"),
                "depth_iqr_m":             obj.get("depth_iqr_m"),
                "depth_jump_flag":         bool(obj.get("depth_jump_flag", False)),
                "depth_coherence_flag":    bool(obj.get("depth_coherence_flag", False)),
                "depth_validity_flag":     bool(obj.get("depth_validity_flag", True)),
                "raw_depth_trigger":       bool(obj.get("raw_depth_trigger", False)),
                "any_depth_trigger":       bool(obj.get("any_depth_trigger", False)),
            })
    return pd.DataFrame(records).sort_values(["frame_id", "object_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scene Graph
# ---------------------------------------------------------------------------

@st.cache_data(hash_funcs={str: lambda p: (p, _mtime(p))})
def load_scene_graph(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (frame_level_df, nodes_df, edges_df)."""
    rows = _read_jsonl(path)
    frame_records = []
    node_records = []
    edge_records = []

    for r in rows:
        fid = r["frame_id"]
        ts = r.get("timestamp")
        frame_records.append({
            "frame_id": fid,
            "timestamp": ts,
            "node_count": len(r.get("nodes", [])),
            "edge_count": len(r.get("edges", [])),
            "near_distance_threshold_m": r.get("near_distance_threshold_m"),
            "localization_flag": r.get("localization_flag"),  # may be None
        })
        for node in r.get("nodes", []):
            pos = node.get("position_3d", {})
            node_records.append({
                "frame_id":    fid,
                "timestamp":   ts,
                "object_id":   node["object_id"],
                "label":       node.get("label"),
                # newer logs use "status" / "bbox_xyxy"; older ones "state" / "bbox"
                "state":       node.get("status", node.get("state")),
                "bbox":        node.get("bbox_xyxy", node.get("bbox")),
                "pixel_center": node.get("pixel_center"),
                "depth_used_m": node.get("depth_used_m"),
                "pos_x": pos.get("x"),
                "pos_y": pos.get("y"),
                "pos_z": pos.get("z"),
            })
        for edge in r.get("edges", []):
            edge_records.append({
                "frame_id":      fid,
                "timestamp":     ts,
                # newer logs use "from_object_id"/"to_object_id"; older ones "from"/"to"
                "from_id":       edge.get("from_object_id", edge.get("from")),
                "to_id":         edge.get("to_object_id", edge.get("to")),
                "relation":      edge.get("relation"),
                "distance_3d_m": edge.get("distance_3d_m"),
                "source":        edge.get("source"),
                "confidence":    edge.get("confidence"),
            })

    frames_df = pd.DataFrame(frame_records).sort_values("frame_id").reset_index(drop=True)
    nodes_df = pd.DataFrame(node_records).sort_values(["frame_id", "object_id"]).reset_index(drop=True) if node_records else pd.DataFrame()
    edges_df = pd.DataFrame(edge_records).sort_values(["frame_id"]).reset_index(drop=True) if edge_records else pd.DataFrame()
    return frames_df, nodes_df, edges_df


# ---------------------------------------------------------------------------
# Ground Truth
# ---------------------------------------------------------------------------

@st.cache_data(hash_funcs={str: lambda p: (p, _mtime(p))})
def load_ground_truth(path: str) -> dict:
    """Returns dict keyed by general_folder_name."""
    from utils import mmss_to_seconds
    if not Path(path).exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for entry in raw.values():
        gfn = entry.get("general_folder_name", "")
        raw_fs = entry.get("gt_failure_step")

        # Normalise: may be a list ["mm:ss","mm:ss"], a single string "mm:ss", or None
        if isinstance(raw_fs, list):
            fs = raw_fs  # already a list
        elif isinstance(raw_fs, str):
            fs = [raw_fs, raw_fs]  # single timestamp → treat as a point window
        else:
            fs = [None, None]

        def _safe_s(v):
            try:
                return mmss_to_seconds(v) if v else None
            except Exception:
                return None

        result[gfn] = {
            "task_name": entry.get("name", ""),
            "gt_failure_reason": entry.get("gt_failure_reason", ""),
            "gt_failure_window_raw": fs,
            "gt_failure_start_s": _safe_s(fs[0] if len(fs) > 0 else None),
            "gt_failure_end_s": _safe_s(fs[1] if len(fs) > 1 else None),
            "object_list": entry.get("object_list", []),
            "actions": entry.get("actions", []),
            "success_condition": entry.get("success_condition", ""),
        }
    return result


# ---------------------------------------------------------------------------
# Replay buffer stub (v1 — not yet implemented)
# ---------------------------------------------------------------------------

def load_replay_buffer(zarr_path: str) -> pd.DataFrame:
    # TODO: paste zarr-decoding code here when available.
    # The zarr layout and field names are TBD from the upstream team.
    return pd.DataFrame()
