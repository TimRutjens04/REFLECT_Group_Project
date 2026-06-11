"""LLM commentary module — calls local Ollama to produce per-sequence findings.

Prompt versioning rule: bump PROMPT_VERSION any time SYSTEM_PROMPT or the
input summary schema changes. The version string is part of the cache key so
stale cached results are automatically invalidated.

# PROMPT_VERSION history:
# v1.0 — initial release
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_HOST          = "http://localhost:11434"
OLLAMA_MODEL_PREFERRED = "llama3.1:8b"
PROMPT_VERSION       = "v1.2"
LLM_TIMEOUT_S        = 60
MAX_FINDINGS_VISIBLE = 4

_VALID_TABS = {"overview", "detection", "tracking", "frame_viewer"}

# Placeholder thresholds — update once Tim R and Georgi provide real values.
_THRESHOLDS = {
    "tracker_low_confidence": 0.5,
    "bbox_size_change":       0.4,
    "depth_jump_m":           0.15,
}

# TODO: populate with 2-3 (summary, commentary) pairs once Guray has hand-reviewed sequences.
FEW_SHOT_EXAMPLES: list[dict] = []


def list_installed_models() -> list[str]:
    """Returns the list of model names installed on the local Ollama server.
    Empty list if Ollama isn't reachable."""
    import requests
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
## Role and task

You are reviewing a single robot perception sequence. The pipeline detects objects \
using Grounding DINO, tracks them between detections, and validates observations using \
depth information. Your task is to read a structured summary of one sequence and produce \
findings that help a human reviewer identify the most important areas to inspect.

## Glossary

- tracker_confidence: score [0-1] produced by the tracker indicating how well the current \
bounding box matches the tracked object; low values suggest drift or occlusion.
- bbox_size_change_flag: fired when the tracked bounding-box area changes abruptly \
relative to its initial size.
- drift_flag: fired when the object's pixel displacement exceeds a threshold over several \
consecutive frames, suggesting the tracker is following the wrong region.
- recovery_trigger: event where the pipeline decides to re-run the detector because \
tracking quality fell below a threshold.
- frames_since_redetect: how many frames have elapsed since the last DINO detection run \
for a given object; large values mean the tracker is running unsupervised for a long time.
- outlier_frames: frames where a metric (e.g. tracker_confidence) is more than 2σ from \
the per-object mean.
- status_transitions: list of (frame_id, status) pairs where the tracker status changed \
(e.g. ok → drifting → recovered).
- gt_failure_window: the ground-truth annotated time range (in frames) where the task \
was classified as failing; findings inside this window are higher severity.
- depth_median_m: median depth in metres for an object's bounding-box pixels at a given frame.
- depth_jump_flag: fired when an object's depth changes abruptly between consecutive frames.
- depth_coherence_flag: True = depth is coherent; False = depth_coherence_violation \
(depth inconsistent with spatial neighbours).
- depth_validity_flag: True = enough valid depth pixels; False = depth_validity_violation \
(insufficient valid depth pixels in the bounding box).
- any_depth_trigger: True when any depth flag caused a re-detection to be requested.
- near_distance_threshold_m: sequence-wide 3-D distance threshold that defines the \
"near" relation between scene-graph nodes.
- object_flicker_rate: fraction of frames in which an object disappears then reappears; \
high values indicate unstable tracking or detection.
- edge_flicker_rate: number of on↔off transitions for a specific pair's "near" edge \
across the sequence; high values indicate unstable spatial relationships.
- neighbor_jaccard: Jaccard similarity between an object's near-neighbour set at \
consecutive frames; low values indicate an unstable neighbourhood.

## Output schema

Return ONLY valid JSON in this exact shape. Do not include any text outside the JSON. \
Do not invent frame_ids or object_ids that are not in the summary. \
If you cannot identify a finding for a category, leave it out rather than guessing. \
Maximum 6 findings.

{
  "summary": "One paragraph, 2-4 sentences, plain English overview of the sequence quality.",
  "findings": [
    {
      "id": "f1",
      "severity": "high",
      "claim": "Plain-English claim about what happened.",
      "evidence": {
        "tab": "tracking",
        "frame_id": 120,
        "object_id": "apple-1",
        "metric": "tracker_confidence",
        "value": 0.31
      }
    }
  ]
}

Severity rules:
- high = the claim references a frame_id inside the GT failure window, OR coincides with \
a pipeline flag the dashboard already raised.
- medium = the claim references a value past P95 or below P5 of the sequence's metric \
distribution.
- info = stable behavior worth noting for context.
"""

# ---------------------------------------------------------------------------
# Build structured summary
# ---------------------------------------------------------------------------

def _round3(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 3)


def _build_depth_block(depth_df: pd.DataFrame) -> dict:
    return {
        "n_objects": int(depth_df["object_id"].nunique()),
        "flag_counts": {
            "depth_jump":               int(depth_df["depth_jump_flag"].sum()),
            "depth_coherence_violation": int((~depth_df["depth_coherence_flag"]).sum()),
            "depth_validity_violation":  int((~depth_df["depth_validity_flag"]).sum()),
            "any_depth_trigger":         int(depth_df["any_depth_trigger"].sum()),
            "raw_depth_trigger":         int(depth_df.get("raw_depth_trigger", pd.Series(dtype=bool)).sum()),
        },
        "first_jump_frame": (
            int(depth_df.loc[depth_df["depth_jump_flag"], "frame_id"].min())
            if depth_df["depth_jump_flag"].any() else None
        ),
        "per_object_median_depth_m": (
            depth_df.groupby("object_id")["depth_median_m"]
            .median().round(3).to_dict()
        ),
    }


def _build_sg_block(sg_frame_df: pd.DataFrame, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict:
    from metrics import object_flicker_rate, mean_3d_displacement, edge_flicker  # local to avoid circular import

    if nodes_df is None or nodes_df.empty:
        return {"n_objects": 0}

    all_sg_frames   = sorted(nodes_df["frame_id"].unique().tolist())
    flicker_df_loc, _ = object_flicker_rate(nodes_df, all_sg_frames), None
    flicker_df_loc  = object_flicker_rate(nodes_df, all_sg_frames)
    _, disp_sum_loc = mean_3d_displacement(nodes_df)

    flicker_rates: dict = {}
    if not flicker_df_loc.empty:
        for _, row in flicker_df_loc.iterrows():
            flicker_rates[row["object_id"]] = _round3(row["flicker_rate"])

    mean_disp: dict = {}
    if not disp_sum_loc.empty:
        for _, row in disp_sum_loc.iterrows():
            mean_disp[row["object_id"]] = _round3(row["mean_disp_m"])

    near_ef_rate = None
    if edges_df is not None and not edges_df.empty:
        ef_df_loc, _ = edge_flicker(edges_df, all_sg_frames, relation_types=["near"])
        if not ef_df_loc.empty:
            near_ef_rate = _round3(float(ef_df_loc["flicker_count"].mean()))

    threshold = None
    if not sg_frame_df.empty and "near_distance_threshold_m" in sg_frame_df.columns:
        val = sg_frame_df["near_distance_threshold_m"].dropna()
        if not val.empty:
            threshold = _round3(float(val.iloc[0]))

    rel_counts: dict = {}
    if edges_df is not None and not edges_df.empty and "relation" in edges_df.columns:
        rel_counts = edges_df["relation"].value_counts().to_dict()

    return {
        "near_distance_threshold_m": threshold,
        "n_objects":                 int(nodes_df["object_id"].nunique()),
        "relation_type_counts":      rel_counts,
        "object_flicker_rates":      flicker_rates,
        "edge_flicker_rate_near":    near_ef_rate,
        "mean_3d_displacement_m":    mean_disp,
    }


def build_structured_summary(
    sid: str,
    det_df: pd.DataFrame,
    track_df: pd.DataFrame,
    gt: dict | None,
    depth_df: pd.DataFrame | None = None,
    sg_frame_df: pd.DataFrame | None = None,
    nodes_df: pd.DataFrame | None = None,
    edges_df: pd.DataFrame | None = None,
) -> dict:
    """
    Assemble the dict consumed by the LLM.  Target size: under 3 KB.
    Depth and scene_graph blocks are included when non-empty DataFrames are supplied.
    """
    n_frames = int(det_df["frame_id"].max() - det_df["frame_id"].min() + 1) if not det_df.empty else 0

    # --- Ground truth ---
    gt_block: dict | None = None
    if gt:
        gt_block = {
            "failure_reason":    gt.get("gt_failure_reason", ""),
            "failure_window_s":  [_round3(gt.get("gt_failure_start_s")),
                                  _round3(gt.get("gt_failure_end_s"))],
        }

    # --- Detection ---
    ran = det_df[det_df["detector_ran"]] if not det_df.empty else pd.DataFrame()
    trigger_counts: dict[str, int] = {}
    sel_confs: list[float] = []
    if not ran.empty:
        for _, row in ran.iterrows():
            reason = str(row.get("trigger_reason", "none"))
            trigger_counts[reason] = trigger_counts.get(reason, 0) + 1
            for det in (row["detections"] or []):
                if det.get("is_selected") and det.get("confidence") is not None:
                    sel_confs.append(float(det["confidence"]))

    sel_arr = np.array(sel_confs) if sel_confs else np.array([])
    conf_stats: dict | None = None
    if len(sel_arr) > 0:
        conf_stats = {
            "min":    _round3(sel_arr.min()),
            "median": _round3(np.median(sel_arr)),
            "max":    _round3(sel_arr.max()),
        }

    detection_block = {
        "n_dino_runs":              len(ran),
        "trigger_reasons":          trigger_counts,
        "selected_confidence_stats": conf_stats,
    }

    # --- Tracking ---
    tracking_block: dict = {}
    if not track_df.empty:
        per_obj: list[dict] = []
        for oid, grp in track_df.groupby("object_id"):
            conf = grp["tracker_confidence"].dropna().values.astype(float)
            # Status transitions
            sorted_grp = grp.sort_values("frame_id")
            transitions: list[dict] = []
            prev_s = None
            for _, r in sorted_grp.iterrows():
                s = r["tracker_status"]
                if s != prev_s:
                    transitions.append({"frame_id": int(r["frame_id"]), "status": s})
                    prev_s = s

            # Outlier frames (>2σ from per-object mean on tracker_confidence)
            outlier_frames: list[int] = []
            if len(conf) >= 3:
                mean_c, std_c = conf.mean(), conf.std()
                if std_c > 0:
                    conf_with_frames = grp[["frame_id", "tracker_confidence"]].dropna()
                    for _, r in conf_with_frames.iterrows():
                        if abs(float(r["tracker_confidence"]) - mean_c) > 2 * std_c:
                            outlier_frames.append(int(r["frame_id"]))

            per_obj.append({
                "object_id": oid,
                "confidence_stats": {
                    "min":    _round3(conf.min())    if len(conf) else None,
                    "p25":    _round3(np.percentile(conf, 25)) if len(conf) else None,
                    "median": _round3(np.median(conf)) if len(conf) else None,
                    "mean":   _round3(conf.mean())   if len(conf) else None,
                    "p75":    _round3(np.percentile(conf, 75)) if len(conf) else None,
                    "max":    _round3(conf.max())    if len(conf) else None,
                } if len(conf) > 0 else {},
                "status_transitions": transitions[:10],   # cap to keep JSON small
                "outlier_frames":     outlier_frames[:10],
            })

        # Recovery trigger counts
        total_rec = int(track_df["any_recovery_trigger"].sum())

        # Inside / outside GT window
        rec_in_gt = rec_out_gt = None
        gtsf = gtef = None
        if gt and gt.get("gt_failure_start_s") is not None:
            gt_s = gt["gt_failure_start_s"]
            gt_e = gt["gt_failure_end_s"]
            if not track_df.empty and "timestamp" in track_df.columns:
                gts_idx = (track_df["timestamp"] - gt_s).abs().idxmin()
                gte_idx = (track_df["timestamp"] - gt_e).abs().idxmin()
                gtsf = int(track_df.loc[gts_idx, "frame_id"])
                gtef = int(track_df.loc[gte_idx, "frame_id"])
                in_gt = track_df[
                    (track_df["frame_id"] >= gtsf) &
                    (track_df["frame_id"] <= gtef) &
                    (track_df["any_recovery_trigger"])
                ]
                rec_in_gt  = len(in_gt)
                rec_out_gt = total_rec - rec_in_gt

        # Object survival (no "lost" status inside GT window)
        survived: bool | None = None
        if gt and rec_in_gt is not None and gtsf is not None:
            win = track_df[
                (track_df["frame_id"] >= gtsf) &
                (track_df["frame_id"] <= gtef)
            ]
            survived = not bool((win["tracker_status"] == "lost").any()) if not win.empty else True

        tracking_block = {
            "objects":                       per_obj,
            "total_recovery_triggers":       total_rec,
            "recovery_triggers_in_gt_window":    rec_in_gt,
            "recovery_triggers_outside_gt_window": rec_out_gt,
        }

        alignment_block = {
            "any_flag_inside_gt_window": (rec_in_gt or 0) > 0,
            "object_survived_gt_window": survived,
        }
    else:
        alignment_block = {}

    modules: dict = {
        "detection": detection_block,
        "tracking":  tracking_block,
    }
    if depth_df is not None and not depth_df.empty:
        modules["depth"] = _build_depth_block(depth_df)
    if nodes_df is not None and not nodes_df.empty:
        modules["scene_graph"] = _build_sg_block(
            sg_frame_df if sg_frame_df is not None else pd.DataFrame(),
            nodes_df,
            edges_df if edges_df is not None else pd.DataFrame(),
        )

    return {
        "sequence_id": sid,
        "n_frames":    n_frames,
        "ground_truth": gt_block,
        "modules":     modules,
        "alignment":   alignment_block,
        "thresholds":  _THRESHOLDS,
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def generate_commentary(summary: dict, model: str) -> dict:
    """POST to Ollama and return parsed JSON, or error dict on any failure."""
    import requests  # local import so module loads without requests installed

    user_msg = json.dumps(summary, indent=2)
    payload  = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "stream":  False,
        "format":  "json",
        "options": {"temperature": 0.1},
    }
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=LLM_TIMEOUT_S,
        )
        if resp.status_code != 200:
            try:
                body_msg = resp.json().get("error", "")
            except Exception:
                body_msg = resp.text[:200]
            return {
                "summary": "",
                "findings": [],
                "error": f"Ollama returned HTTP {resp.status_code}: {body_msg}",
            }
        content = resp.json()["message"]["content"]
        return json.loads(content)
    except requests.exceptions.ConnectionError:
        return {"summary": "", "findings": [], "error": "Ollama is not running (connection refused)."}
    except requests.exceptions.Timeout:
        return {"summary": "", "findings": [], "error": f"Ollama timed out after {LLM_TIMEOUT_S}s."}
    except (KeyError, json.JSONDecodeError) as e:
        return {"summary": "", "findings": [], "error": f"Model returned invalid output: {e}"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_findings(
    parsed: dict,
    det_df: pd.DataFrame,
    track_df: pd.DataFrame,
    show_depth: bool = False,
    show_sg: bool = False,
) -> dict:
    """Drop findings that reference non-existent frame_ids or object_ids."""
    max_frame  = int(track_df["frame_id"].max()) if not track_df.empty else 0
    valid_oids = set(track_df["object_id"].unique()) if not track_df.empty else set()

    valid_tabs = {"overview", "detection", "tracking", "frame_viewer"}
    if show_depth:
        valid_tabs.add("depth")
    if show_sg:
        valid_tabs.add("scene_graph")

    clean: list[dict] = []
    for f in parsed.get("findings", []):
        ev  = f.get("evidence", {})
        fid = ev.get("frame_id")
        oid = ev.get("object_id")
        tab = ev.get("tab", "")

        if fid is not None and not (0 <= int(fid) <= max_frame):
            continue
        if oid is not None and oid not in valid_oids:
            continue
        if tab not in valid_tabs:
            f = {**f, "evidence": {**ev, "tab": "overview"}}

        clean.append(f)

    return {**parsed, "findings": clean}


# ---------------------------------------------------------------------------
# Hash + cached entry point
# ---------------------------------------------------------------------------

def summary_hash(summary: dict) -> str:
    return hashlib.sha1(
        json.dumps(summary, sort_keys=True).encode()
    ).hexdigest()[:12]


def get_commentary(
    summary: dict,
    det_df: pd.DataFrame,
    track_df: pd.DataFrame,
    model: str,
    force: bool = False,
    show_depth: bool = False,
    show_sg: bool = False,
) -> dict:
    """
    Public entry point.  Caches per (model, prompt_version, summary_hash).
    Pass force=True to bypass the cache (Regenerate button).
    """
    import streamlit as st

    if not model:
        return {
            "summary": "",
            "findings": [],
            "error": (
                "No LLM model selected — Ollama was not reachable when the "
                "sidebar loaded. Start it with `ollama serve`, then click Reload."
            ),
        }

    sh    = summary_hash(summary)
    sjson = json.dumps(summary, sort_keys=True)

    @st.cache_data
    def _cached(model: str, pv: str, h: str, sj: str) -> dict:
        raw = generate_commentary(json.loads(sj), model)
        return raw

    if force:
        _cached.clear()

    raw      = _cached(model, PROMPT_VERSION, sh, sjson)
    n_before = len(raw.get("findings", []))
    validated = validate_findings(raw, det_df, track_df, show_depth=show_depth, show_sg=show_sg)
    n_after   = len(validated.get("findings", []))

    validated["_meta"] = {
        "model":                       model,
        "prompt_version":              PROMPT_VERSION,
        "hash":                        sh,
        "generated_at":                datetime.now(timezone.utc).isoformat(),
        "n_findings_before_validation": n_before,
        "n_findings_after_validation":  n_after,
    }
    return validated
