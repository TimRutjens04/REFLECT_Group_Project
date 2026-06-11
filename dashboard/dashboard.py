"""Pipeline Analysis Dashboard — run with: streamlit run dashboard/dashboard.py"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
SHOW_DEPTH_TAB       = True
SHOW_SCENE_GRAPH_TAB = True

# ---------------------------------------------------------------------------
# Streamlit version guard
# ---------------------------------------------------------------------------
import streamlit as st
from packaging.version import Version

_ST_MIN = "1.35.0"
if Version(st.__version__) < Version(_ST_MIN):
    st.error(
        f"This dashboard requires Streamlit ≥ {_ST_MIN}. "
        f"You have {st.__version__}. Run: pip install --upgrade streamlit"
    )
    st.stop()

import numpy as np
import pandas as pd
import plotly.express as px

from loaders import (
    load_detection, load_tracking, load_tracking_frame_level,
    load_depth, load_scene_graph, load_ground_truth, load_validation,
)
from metrics import (
    object_flicker_rate, mean_3d_displacement,
    neighbor_jaccard, edge_flicker, auto_interpret,
)
from plots import (
    plot_object_count_consistency,
    plot_tracker_timeline,
    plot_detector_cadence,
    plot_detection_confidence,
    plot_depth_timeline,
    plot_sg_displacement,
    plot_sg_jaccard,
    plot_edge_flicker_heatmap,
    plot_flag_strip,
    plot_relation_mix_bar,
)
from interactive import (
    render_stats_table,
    render_frame_info_card,
    render_frame_panel,
    inline_frame_panel,
    absent_frames_viewer,
)
from llm import (
    build_structured_summary,
    get_commentary,
    list_installed_models,
    OLLAMA_MODEL_PREFERRED,
    MAX_FINDINGS_VISIBLE,
)
from utils import discover_sequences, find_frame_path, detect_object_mismatches
from frame_viewer import draw_bboxes, draw_scene_graph_overlay

st.set_page_config(layout="wide", page_title="Pipeline Analysis Dashboard")

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DASHBOARD_DIR   = Path(__file__).resolve().parent
REPO_ROOT       = DASHBOARD_DIR.parent
OUTPUTS_DEFAULT = str(REPO_ROOT / "pipeline" / "real_world" / "jsonl")
DATA_DEFAULT    = str(REPO_ROOT / "example_data" / "real_data")
GT_DEFAULT      = str(REPO_ROOT / "example_data" / "tasks_real_world.json")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Pipeline Analysis")

outputs_root = st.sidebar.text_input("Outputs root (jsonl dir)", value=OUTPUTS_DEFAULT)
data_root    = st.sidebar.text_input("Data root (real_data dir)", value=DATA_DEFAULT)
gt_path      = st.sidebar.text_input("tasks_real_world.json path", value=GT_DEFAULT)

if st.sidebar.button("Reload"):
    st.cache_data.clear()
    st.rerun()

# ── LLM model selector ─────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**LLM**")

installed_models = list_installed_models()
if installed_models:
    if OLLAMA_MODEL_PREFERRED in installed_models:
        default_idx = installed_models.index(OLLAMA_MODEL_PREFERRED)
    else:
        default_idx = 0
    llm_model = st.sidebar.selectbox(
        "Model",
        options=installed_models,
        index=default_idx,
        key="llm_model_selected",
        help=(
            f"For best output quality install `{OLLAMA_MODEL_PREFERRED}`: "
            f"run `ollama pull {OLLAMA_MODEL_PREFERRED}` in a terminal."
        ),
    )
    if llm_model != OLLAMA_MODEL_PREFERRED:
        st.sidebar.caption(
            f"⚠️ Using `{llm_model}`. `{OLLAMA_MODEL_PREFERRED}` is recommended "
            f"for better structured-output quality."
        )
else:
    st.sidebar.error(
        "Ollama is not reachable. Start it with: `ollama serve`"
    )
    llm_model = None

sequences = discover_sequences(outputs_root)
if not sequences:
    st.error(
        f"No sequences found under: {outputs_root}\n\n"
        "Each sequence needs `<id>__detection.jsonl` and `<id>__tracking.jsonl`."
    )
    st.stop()

sid = st.sidebar.selectbox("Sequence", sequences)

st.sidebar.markdown("---")
st.sidebar.markdown("**Chart options**")
show_per_object = st.sidebar.checkbox(
    "Show per-object overlays on tracker timeline", value=True,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
seq_dir = Path(outputs_root) / sid

det_df         = load_detection(str(seq_dir / f"{sid}__detection.jsonl"))
track_df       = load_tracking(str(seq_dir / f"{sid}__tracking.jsonl"))
track_frame_df = load_tracking_frame_level(str(seq_dir / f"{sid}__tracking.jsonl"))
validation_df  = load_validation(str(seq_dir / f"{sid}__validation.jsonl"))


def _try_load_depth():
    p = str(seq_dir / f"{sid}__depth.jsonl")
    if not Path(p).exists():
        return pd.DataFrame()
    try:
        return load_depth(p)
    except Exception as e:
        st.warning(f"Could not load depth log: {e}")
        return pd.DataFrame()


def _try_load_sg():
    p = str(seq_dir / f"{sid}__scene_graph.jsonl")
    if not Path(p).exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        return load_scene_graph(p)
    except Exception as e:
        st.warning(f"Could not load scene graph log: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


depth_df = _try_load_depth() if SHOW_DEPTH_TAB else pd.DataFrame()
sg_frame_df, nodes_df, edges_df = (
    _try_load_sg() if SHOW_SCENE_GRAPH_TAB
    else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
)

gt_dict = load_ground_truth(gt_path)
gt      = gt_dict.get(sid)

all_frames = sorted(set(
    det_df["frame_id"].tolist()
    + (track_frame_df["frame_id"].tolist() if not track_frame_df.empty else [])
    + (sg_frame_df["frame_id"].tolist()    if not sg_frame_df.empty    else [])
    + (depth_df["frame_id"].tolist()       if not depth_df.empty       else [])
))

gt_start_s = gt["gt_failure_start_s"] if gt else None
gt_end_s   = gt["gt_failure_end_s"]   if gt else None


def _ts_to_frame(ts_s, df):
    if ts_s is None or df.empty or "timestamp" not in df.columns:
        return None
    diff = (df["timestamp"] - ts_s).abs()
    return int(df.loc[diff.idxmin(), "frame_id"])


gt_start_frame = _ts_to_frame(gt_start_s, det_df)
gt_end_frame   = _ts_to_frame(gt_end_s,   det_df)

frames_dir       = str(Path(data_root) / sid / "frames")
frames_available = Path(frames_dir).exists()

total_frames = len(all_frames)
n_dino_runs  = int(det_df["detector_ran"].sum())
pct_dino     = n_dino_runs / total_frames * 100 if total_frames else 0
n_recovery   = int(track_frame_df["any_recovery_trigger"].sum()) if not track_frame_df.empty else 0
all_obj_ids  = set(
    [d.get("object_id") for dets in det_df["detections"] for d in dets if isinstance(d, dict)]
    + (track_df["object_id"].unique().tolist() if not track_df.empty else [])
    + (nodes_df["object_id"].unique().tolist() if not nodes_df.empty else [])
) - {None}

# LLM findings from previous run (used by Frame Viewer info card)
_llm_comm    = st.session_state.get(f"llm_{sid}", {})
llm_findings = _llm_comm.get("findings", []) if isinstance(_llm_comm, dict) else []

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SEVERITY_STYLE = {
    "high":   "background:#fde8e8;color:#c0392b;padding:2px 6px;border-radius:4px;font-size:0.82em;",
    "medium": "background:#fef3cd;color:#856404;padding:2px 6px;border-radius:4px;font-size:0.82em;",
    "info":   "background:#e8f0fe;color:#3d5a99;padding:2px 6px;border-radius:4px;font-size:0.82em;",
}


def _render_inspect_banner(tab_id: str) -> None:
    """Render the Inspect target banner at the top of a tab.
    `tab_id` must be unique per tab to avoid Streamlit key collisions."""
    it = st.session_state.get("inspect_target")
    if not it:
        return
    tab = it.get("tab", "")
    frm = it.get("frame")
    obj = it.get("object")
    msg = (
        f"Inspecting: **{tab}** · frame {frm} · object {obj}"
        f" — open the {tab.title()} tab and the Frame Viewer to see it."
    )
    c_msg, c_clear = st.columns([9, 1])
    with c_msg:
        st.info(msg)
    with c_clear:
        if st.button("Clear", key=f"clear_inspect_{tab_id}"):
            del st.session_state["inspect_target"]
            st.rerun()


def _render_finding_card(finding: dict, idx: int, suffix: str = "") -> None:
    sev   = finding.get("severity", "info")
    style = _SEVERITY_STYLE.get(sev, _SEVERITY_STYLE["info"])
    claim = finding.get("claim", "")
    ev    = finding.get("evidence", {})
    c_chip, c_text, c_btn = st.columns([1, 7, 1])
    with c_chip:
        st.markdown(f'<span style="{style}">{sev.upper()}</span>', unsafe_allow_html=True)
    with c_text:
        st.markdown(claim)
    with c_btn:
        if st.button("Inspect →", key=f"inspect_finding_{idx}{suffix}"):
            target_frame = ev.get("frame_id")
            st.session_state["inspect_target"] = {
                "tab":    ev.get("tab", "overview"),
                "frame":  target_frame,
                "object": ev.get("object_id"),
            }
            # Pre-position the Frame Viewer slider so opening that tab lands on the target frame
            if target_frame is not None:
                st.session_state["frame_viewer_frame"] = int(target_frame)
            st.rerun()


def _render_llm_commentary():
    with st.container(border=True):
        c_title, c_btn = st.columns([6, 1])
        with c_title:
            st.markdown("**LLM commentary**")
        with c_btn:
            if st.button("Regenerate", type="secondary", key="llm_regen"):
                st.session_state["llm_force_regenerate"] = True
                st.rerun()

        llm_key = f"llm_{sid}"
        force   = st.session_state.pop("llm_force_regenerate", False)

        if not force and llm_key not in st.session_state:
            st.caption("Click **Regenerate** to produce LLM commentary for this sequence.")
            return

        if force or llm_key not in st.session_state:
            if llm_model is None:
                st.info("Ollama is not reachable. Start it with: `ollama serve`, then click **Reload** in the sidebar.")
                return
            with st.spinner("Generating commentary…"):
                summary = build_structured_summary(
                    sid, det_df, track_df, gt,
                    depth_df=depth_df if SHOW_DEPTH_TAB else None,
                    sg_frame_df=sg_frame_df if SHOW_SCENE_GRAPH_TAB else None,
                    nodes_df=nodes_df if SHOW_SCENE_GRAPH_TAB else None,
                    edges_df=edges_df if SHOW_SCENE_GRAPH_TAB else None,
                )
                comm = get_commentary(
                    summary, det_df, track_df, model=llm_model, force=force,
                    show_depth=SHOW_DEPTH_TAB, show_sg=SHOW_SCENE_GRAPH_TAB,
                )
            st.session_state[llm_key] = comm
        else:
            comm = st.session_state[llm_key]

        if comm.get("error"):
            st.info(
                f"LLM commentary unavailable — {comm['error']}  \n"
                "Ollama may not be running, or the model returned invalid output. "
                "See the other tabs for the underlying signals."
            )
            if st.button("Retry", key="llm_retry"):
                if llm_key in st.session_state:
                    del st.session_state[llm_key]
                st.rerun()
            return

        if comm.get("summary"):
            st.markdown(f"_{comm['summary']}_")

        findings = comm.get("findings", [])
        visible  = findings[:MAX_FINDINGS_VISIBLE]
        hidden   = findings[MAX_FINDINGS_VISIBLE:]

        for i, f in enumerate(visible):
            _render_finding_card(f, i)

        if hidden:
            with st.expander(f"+{len(hidden)} more"):
                for i, f in enumerate(hidden, len(visible)):
                    _render_finding_card(f, i, suffix="_exp")

        meta = comm.get("_meta", {})
        if meta:
            h    = meta.get("hash", "")[:8]
            pv   = meta.get("prompt_version", "")
            mdl  = meta.get("model", "")
            gat  = meta.get("generated_at", "")[:16].replace("T", " ")
            st.caption(f"{mdl} · prompt {pv} · {h} · generated {gat}")


def _compute_alignment():
    if gt_start_frame is None:
        return None
    aligned = False
    if not track_frame_df.empty:
        win = track_frame_df[
            (track_frame_df["frame_id"] >= gt_start_frame) &
            (track_frame_df["frame_id"] <= gt_end_frame)
        ]
        aligned |= bool(win["any_recovery_trigger"].any())
    if not depth_df.empty:
        win = depth_df[
            (depth_df["frame_id"] >= gt_start_frame) &
            (depth_df["frame_id"] <= gt_end_frame)
        ]
        aligned |= bool(win["any_depth_trigger"].any())
    return aligned


@st.cache_data
def _build_summary_table(
    outputs_root: str,
    gt_json_path: str,
    show_depth: bool,
    show_sg: bool,
):
    from loaders import (
        load_detection, load_tracking, load_tracking_frame_level,
        load_depth, load_scene_graph, load_ground_truth,
    )
    gt_d = load_ground_truth(gt_json_path)
    seqs = discover_sequences(outputs_root)
    rows = []
    for seq in seqs:
        sdir = Path(outputs_root) / seq
        try:
            _det      = load_detection(str(sdir / f"{seq}__detection.jsonl"))
            _track    = load_tracking(str(sdir / f"{seq}__tracking.jsonl"))
            _track_fl = load_tracking_frame_level(str(sdir / f"{seq}__tracking.jsonl"))
            _depth    = load_depth(str(sdir / f"{seq}__depth.jsonl")) if (
                show_depth and (sdir / f"{seq}__depth.jsonl").exists()
            ) else pd.DataFrame()
            if show_sg and (sdir / f"{seq}__scene_graph.jsonl").exists():
                _, _nodes, _edges = load_scene_graph(str(sdir / f"{seq}__scene_graph.jsonl"))
            else:
                _nodes = _edges = pd.DataFrame()
        except Exception as e:
            rows.append({"sequence_id": seq, "error": str(e)})
            continue

        _gt   = gt_d.get(seq)
        _gt_ss = _gt["gt_failure_start_s"] if _gt else None
        _gt_es = _gt["gt_failure_end_s"]   if _gt else None

        def _tf(ts, df):
            if ts is None or df.empty or "timestamp" not in df.columns:
                return None
            diff = (df["timestamp"] - ts).abs()
            return int(df.loc[diff.idxmin(), "frame_id"])

        _gt_sf = _tf(_gt_ss, _det)
        _gt_ef = _tf(_gt_es, _det)

        n_dino = int(_det["detector_ran"].sum())
        n_rec  = int(_track_fl["any_recovery_trigger"].sum()) if not _track_fl.empty else 0

        survived = None
        if _gt_sf is not None and not _track.empty:
            win = _track[(_track["frame_id"] >= _gt_sf) & (_track["frame_id"] <= _gt_ef)]
            survived = not bool((win["tracker_status"] == "lost").any()) if not win.empty else True

        aligned = None
        if _gt_sf is not None:
            aligned = False
            if not _track_fl.empty:
                win = _track_fl[(_track_fl["frame_id"] >= _gt_sf) & (_track_fl["frame_id"] <= _gt_ef)]
                aligned |= bool(win["any_recovery_trigger"].any())

        gt_win_str = (
            f"{_gt['gt_failure_window_raw'][0]}–{_gt['gt_failure_window_raw'][1]}"
            if _gt else "N/A"
        )
        row = {
            "sequence_id":       seq,
            "task_name":         _gt["task_name"] if _gt else "N/A",
            "GT failure window": gt_win_str,
            "DINO calls":        n_dino,
            "recovery triggers": n_rec,
            "object survived":   survived,
            "aligned?":          aligned,
        }
        if show_depth and not _depth.empty:
            row["depth_flag_firings"] = int(_depth["any_depth_trigger"].sum())
        if show_sg and not _nodes.empty:
            row["sg_object_count"] = int(_nodes["object_id"].nunique())
            edges_changed = None
            if _gt_sf is not None and not _edges.empty and "relation" in _edges.columns:
                near_e = _edges[_edges["relation"] == "near"]
                if not near_e.empty:
                    win_e  = near_e[(near_e["frame_id"] >= _gt_sf) & (near_e["frame_id"] <= _gt_ef)]
                    prev_e = near_e[near_e["frame_id"] < _gt_sf]
                    if not prev_e.empty and not win_e.empty:
                        last_fid   = prev_e["frame_id"].max()
                        last_pairs = frozenset(
                            frozenset([r.from_id, r.to_id])
                            for _, r in prev_e[prev_e["frame_id"] == last_fid].iterrows()
                        )
                        edges_changed = False
                        for fid in sorted(win_e["frame_id"].unique()):
                            cur_pairs = frozenset(
                                frozenset([r.from_id, r.to_id])
                                for _, r in win_e[win_e["frame_id"] == fid].iterrows()
                            )
                            if cur_pairs != last_pairs:
                                edges_changed = True
                                break
            row["edges_changed_in_gt"] = edges_changed
        rows.append(row)
    return pd.DataFrame(rows)


def _render_expanded_frame_block():
    """If session state has an expanded frame, render it inline above other content."""
    ef = st.session_state.get("expanded_frame")
    if not ef:
        return

    frame_id = ef["frame_id"]
    caption  = ef["caption"]

    with st.container(border=True):
        col_a, col_b = st.columns([10, 1])
        with col_a:
            st.markdown(f"**Expanded frame {frame_id}** — {caption}")
        with col_b:
            if st.button("✕ Close", key="expanded_frame_close",
                         use_container_width=True):
                del st.session_state["expanded_frame"]
                st.rerun()

        if not frames_available:
            st.warning("Frames folder not available.")
            return
        fp = find_frame_path(frames_dir, frame_id)
        if not fp:
            st.warning(f"Image not found: frame {frame_id}")
            return

        det_row = det_df[det_df["frame_id"] == frame_id]
        dets    = det_row["detections"].values[0] if not det_row.empty else []
        t_rows  = track_df[track_df["frame_id"] == frame_id] if not track_df.empty else pd.DataFrame()
        tracked = [
            {"object_id": r.object_id, "bbox_xyxy": r.bbox_xyxy,
             "tracker_status": r.tracker_status, "tracker_confidence": r.tracker_confidence}
            for _, r in t_rows.iterrows()
        ] if not t_rows.empty else []
        img = draw_bboxes(fp, dets, tracked, set())
        st.image(img, caption=caption, use_container_width=True)


def _render_thumbnail(frame_id: int, caption: str, width: int = 200):
    if not frames_available:
        st.caption(caption)
        st.warning("No frames folder")
        return
    fp = find_frame_path(frames_dir, frame_id)
    if not fp:
        st.caption(caption)
        st.warning(f"Image not found: frame {frame_id}")
        return
    det_row = det_df[det_df["frame_id"] == frame_id]
    dets    = det_row["detections"].values[0] if not det_row.empty else []
    t_rows  = track_df[track_df["frame_id"] == frame_id] if not track_df.empty else pd.DataFrame()
    tracked = [
        {"object_id": r.object_id, "bbox_xyxy": r.bbox_xyxy,
         "tracker_status": r.tracker_status, "tracker_confidence": r.tracker_confidence}
        for _, r in t_rows.iterrows()
    ] if not t_rows.empty else []
    img = draw_bboxes(fp, dets, tracked, set())
    st.image(img, caption=caption, width=width)
    if st.button("🔍 Expand", key=f"thumb_expand_{frame_id}_{caption}",
                 use_container_width=True):
        st.session_state["expanded_frame"] = {
            "frame_id": frame_id,
            "caption":  caption,
        }
        st.rerun()


def _build_mismatch_df():
    det_ids, det_fl = set(), {}
    for _, row in det_df.iterrows():
        for d in row["detections"]:
            oid = d.get("object_id")
            if not oid:
                continue
            det_ids.add(oid)
            det_fl.setdefault(oid, {"det_first": None, "det_last": None})
            fid = row["frame_id"]
            if det_fl[oid]["det_first"] is None or fid < det_fl[oid]["det_first"]:
                det_fl[oid]["det_first"] = fid
            if det_fl[oid]["det_last"] is None or fid > det_fl[oid]["det_last"]:
                det_fl[oid]["det_last"] = fid

    track_ids = set(track_df["object_id"].unique()) if not track_df.empty else set()
    sg_ids    = set(nodes_df["object_id"].unique()) if not nodes_df.empty else set()
    mismatch  = detect_object_mismatches(det_ids, track_ids, sg_ids)
    rows = []
    for oid, presence in mismatch.items():
        fl    = det_fl.get(oid, {})
        t_sub = track_df[track_df["object_id"] == oid] if not track_df.empty else pd.DataFrame()
        rows.append({
            "object_id":  oid,
            "det_first":  fl.get("det_first"),
            "det_last":   fl.get("det_last"),
            "track_first": int(t_sub["frame_id"].min()) if not t_sub.empty else None,
            "track_last":  int(t_sub["frame_id"].max()) if not t_sub.empty else None,
            "in_detection": presence["in_detection"],
            "in_tracking":  presence["in_tracking"],
        })
    df = pd.DataFrame(rows)

    def _highlight(row):
        styles, cols = [""] * len(row), row.index.tolist()
        for flag_col, vcols in [("in_detection", ["det_first", "det_last"]),
                                 ("in_tracking",  ["track_first", "track_last"])]:
            if not row.get(flag_col, True):
                for vc in vcols:
                    if vc in cols:
                        styles[cols.index(vc)] = "background-color: #ffcccc"
        return styles

    return df, _highlight


# ===========================================================================
# TAB CONTENT
# ===========================================================================

FLAG_NAMES = [
    "bbox_size_change_flag",
    "drift_flag",
    "frame_counter_K_flag",
    "any_recovery_trigger",
]


def render_overview():
    _render_inspect_banner("ov")

    # If a thumbnail has been expanded, show only the full-size view and stop.
    if st.session_state.get("expanded_frame"):
        _render_expanded_frame_block()
        return

    st.header(f"Sequence: {sid}")

    if gt:
        raw = gt["gt_failure_window_raw"]
        st.markdown(f"**Task:** {gt['task_name']}")
        st.markdown(f"**GT failure reason:** {gt['gt_failure_reason']}")
        if gt_start_s is not None:
            st.markdown(
                f"**GT failure window:** {raw[0]} – {raw[1]}"
                f"  ({gt_start_s:.1f}s – {gt_end_s:.1f}s)"
            )
    else:
        st.info("No ground truth available for this sequence.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total frames",      total_frames)
    c2.metric("DINO runs",         n_dino_runs)
    c3.metric("% DINO ran",        f"{pct_dino:.1f}%")
    c4.metric("Recovery triggers", n_recovery)
    c5.metric("Objects ever seen", len(all_obj_ids))

    aligned = _compute_alignment()
    if aligned is None:
        st.info("GT-failure alignment: no ground truth for this sequence.")
    elif aligned:
        st.success("GT-failure alignment: ALIGNED — pipeline flag(s) fired inside the GT failure window")
    else:
        st.error("GT-failure alignment: NOT ALIGNED — no pipeline flags inside the GT failure window")

    st.divider()

    # LLM commentary block (between header and GT preview)
    _render_llm_commentary()

    st.divider()

    # GT-window frame preview
    if gt_start_frame is not None and frames_available:
        st.subheader("Ground-Truth Failure Window — Frame Preview")
        window_frames = [f for f in all_frames if gt_start_frame <= f <= gt_end_frame]

        # If GT window is too narrow (e.g. single-timestamp GT), expand to
        # ±15 frames around its center for preview only.
        MIN_PREVIEW_SPAN = 30  # frames; ~1 second at 30 fps
        window_expanded = False
        if len(window_frames) < 5 and all_frames:
            center = (gt_start_frame + gt_end_frame) // 2
            lo = max(min(all_frames), center - MIN_PREVIEW_SPAN // 2)
            hi = min(max(all_frames), center + MIN_PREVIEW_SPAN // 2)
            window_frames = [f for f in all_frames if lo <= f <= hi]
            window_expanded = True

        if window_frames:
            if len(window_frames) <= 5:
                sample = window_frames
            else:
                idxs = [0, len(window_frames)//4, len(window_frames)//2,
                        3*len(window_frames)//4, len(window_frames)-1]
                sample = [window_frames[i] for i in idxs]
            st.caption(
                "GT failure marked at a single timestamp — showing five frames around it for context."
                if window_expanded else
                "Five frames evenly spaced across the GT failure window."
            )
            cols = st.columns(len(sample))
            for col, fid in zip(cols, sample):
                ts = det_df[det_df["frame_id"] == fid]["timestamp"].values
                ts_str = f" t={ts[0]:.2f}s" if len(ts) else ""
                with col:
                    _render_thumbnail(fid, f"f{fid}{ts_str}", width=200)
        else:
            st.info("No frames fall inside the GT failure window.")
    elif not frames_available:
        st.info(
            f"Frames not yet extracted. Run:\n"
            f"`python dashboard/extract_frames.py --folder-name {sid}`"
        )

    st.divider()

    # Top-5 lowest-confidence DINO detections
    st.subheader("Top-5 Lowest-Confidence DINO Detections")
    ran = det_df[det_df["detector_ran"]]
    conf_records = []
    for _, row in ran.iterrows():
        for det in (row["detections"] or []):
            if det.get("is_selected") and det.get("confidence") is not None:
                conf_records.append({
                    "frame_id":  row["frame_id"],
                    "object_id": det["object_id"],
                    "confidence": float(det["confidence"]),
                })
    if conf_records:
        conf_df = (
            pd.DataFrame(conf_records)
            .sort_values("confidence")
            .drop_duplicates(subset=["frame_id", "object_id"])
            .head(5)
        )
        cols = st.columns(len(conf_df))
        for col, (_, row) in zip(cols, conf_df.iterrows()):
            cap = f"f{row.frame_id} {row.object_id}\nconf={row.confidence:.3f}"
            with col:
                _render_thumbnail(int(row.frame_id), cap, width=200)
        st.session_state["frame_viewer_frame"] = int(conf_df.iloc[0]["frame_id"])
    else:
        st.info("No DINO frames with is_selected detections found.")

    st.divider()

    # Cross-sequence summary
    st.subheader("Failure-Alignment Summary (all sequences)")
    summary_df = _build_summary_table(outputs_root, gt_path, SHOW_DEPTH_TAB, SHOW_SCENE_GRAPH_TAB)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download CSV", summary_df.to_csv(index=False),
        file_name="failure_alignment_summary.csv", mime="text/csv",
    )


# ---------------------------------------------------------------------------

def render_detection():
    _render_inspect_banner("det")
    st.header("Detection")

    # ── Filters ─────────────────────────────────────────────────────────────
    ran_df    = det_df[det_df["detector_ran"]] if not det_df.empty else pd.DataFrame()
    det_obj_ids = sorted({
        d.get("object_id") for _, row in ran_df.iterrows()
        for d in (row["detections"] or [])
        if d.get("is_selected") and d.get("object_id")
    })
    trigger_reasons = sorted(ran_df["trigger_reason"].dropna().unique().tolist()) if not ran_df.empty else []

    def _det_sel_all_obj():
        for oid in det_obj_ids:
            st.session_state[f"det_obj_{oid}"] = True

    def _det_desel_all_obj():
        for oid in det_obj_ids:
            st.session_state[f"det_obj_{oid}"] = False

    def _det_sel_all_trg():
        for tr in trigger_reasons:
            st.session_state[f"det_trg_{tr}"] = True

    def _det_desel_all_trg():
        for tr in trigger_reasons:
            st.session_state[f"det_trg_{tr}"] = False

    with st.expander("Filters", expanded=False):
        c_obj, c_trg = st.columns(2)

        with c_obj:
            st.markdown("**Object IDs**")
            b1, b2 = st.columns(2)
            with b1:
                st.button("Select all", key="det_sel_all_obj",
                          on_click=_det_sel_all_obj, use_container_width=True)
            with b2:
                st.button("Deselect all", key="det_desel_all_obj",
                          on_click=_det_desel_all_obj, use_container_width=True)
            for oid in det_obj_ids:
                if f"det_obj_{oid}" not in st.session_state:
                    st.session_state[f"det_obj_{oid}"] = False
                st.checkbox(oid, key=f"det_obj_{oid}")

        with c_trg:
            st.markdown("**Trigger reasons**")
            b3, b4 = st.columns(2)
            with b3:
                st.button("Select all", key="det_sel_all_trg",
                          on_click=_det_sel_all_trg, use_container_width=True)
            with b4:
                st.button("Deselect all", key="det_desel_all_trg",
                          on_click=_det_desel_all_trg, use_container_width=True)
            for tr in trigger_reasons:
                if f"det_trg_{tr}" not in st.session_state:
                    st.session_state[f"det_trg_{tr}"] = False
                st.checkbox(tr, key=f"det_trg_{tr}")

    selected_det_objects  = [oid for oid in det_obj_ids  if st.session_state.get(f"det_obj_{oid}", False)]
    selected_det_triggers = [tr  for tr  in trigger_reasons if st.session_state.get(f"det_trg_{tr}", False)]

    st.subheader("Detection Confidence Timeline")
    fig_dc, _ = plot_detection_confidence(
        det_df, gt_start_frame, gt_end_frame,
        selected_objects=selected_det_objects,
        selected_triggers=selected_det_triggers or None,
    )
    event_dc = st.plotly_chart(
        fig_dc, on_select="rerun", key="detection_conf_click",
        use_container_width=True,
    )
    inline_frame_panel(
        event_dc, "detection_conf_click", all_frames,
        frames_dir if frames_available else None,
        det_df, track_df, depth_df, nodes_df, edges_df,
    )
    if fig_dc.data:
        st.markdown(
            "**Failure mode legend:** `no_object` — nothing found; "
            "`low_confidence` — below threshold; "
            "`multiple_ambiguous` — couldn't pick one; "
            "`wrong_category` — wrong class. "
            "Red markers = non-null `failure_mode`."
        )

    st.divider()

    st.subheader("Detector Cadence")
    col_plot, col_table = st.columns([3, 1])
    with col_plot:
        st.plotly_chart(plot_detector_cadence(track_df, det_df), use_container_width=True)
    with col_table:
        if not ran_df.empty:
            rc = ran_df["trigger_reason"].value_counts().reset_index()
            rc.columns = ["trigger_reason", "count"]
            st.caption("Trigger reason counts")
            st.dataframe(rc, use_container_width=True, hide_index=True)
            mean_rt = ran_df["runtime_ms"].dropna().mean()
            st.caption(f"DINO mean runtime: {mean_rt:.1f} ms")


# ---------------------------------------------------------------------------

def render_tracking():
    _render_inspect_banner("trk")
    st.header("Tracking")

    # ── Filters ─────────────────────────────────────────────────────────────
    all_obj_ids_track = sorted(track_df["object_id"].unique().tolist()) if not track_df.empty else []

    def _trk_sel_all_obj():
        for oid in all_obj_ids_track:
            st.session_state[f"trk_obj_{oid}"] = True

    def _trk_desel_all_obj():
        for oid in all_obj_ids_track:
            st.session_state[f"trk_obj_{oid}"] = False

    def _trk_sel_all_flg():
        for f in FLAG_NAMES:
            st.session_state[f"trk_flg_{f}"] = True

    def _trk_desel_all_flg():
        for f in FLAG_NAMES:
            st.session_state[f"trk_flg_{f}"] = False

    with st.expander("Filters", expanded=False):
        c_obj, c_flg = st.columns(2)

        with c_obj:
            st.markdown("**Objects**")
            b1, b2 = st.columns(2)
            with b1:
                st.button("Select all", key="trk_sel_all_obj",
                          on_click=_trk_sel_all_obj, use_container_width=True)
            with b2:
                st.button("Deselect all", key="trk_desel_all_obj",
                          on_click=_trk_desel_all_obj, use_container_width=True)
            for oid in all_obj_ids_track:
                if f"trk_obj_{oid}" not in st.session_state:
                    st.session_state[f"trk_obj_{oid}"] = False
                st.checkbox(oid, key=f"trk_obj_{oid}")

        with c_flg:
            st.markdown("**Flags**")
            b3, b4 = st.columns(2)
            with b3:
                st.button("Select all", key="trk_sel_all_flg",
                          on_click=_trk_sel_all_flg, use_container_width=True)
            with b4:
                st.button("Deselect all", key="trk_desel_all_flg",
                          on_click=_trk_desel_all_flg, use_container_width=True)
            for f in FLAG_NAMES:
                if f"trk_flg_{f}" not in st.session_state:
                    st.session_state[f"trk_flg_{f}"] = False
                st.checkbox(f, key=f"trk_flg_{f}")

    selected_track_objects = [oid for oid in all_obj_ids_track if st.session_state.get(f"trk_obj_{oid}", False)]
    selected_track_flags   = [f   for f   in FLAG_NAMES        if st.session_state.get(f"trk_flg_{f}",   False)]

    st.subheader("Tracker Timeline")
    fig_tt, _ = plot_tracker_timeline(
        track_df, det_df, gt_start_frame, gt_end_frame,
        show_per_object=show_per_object,
        selected_objects=selected_track_objects,
        selected_flags=selected_track_flags or None,
    )
    event_tt = st.plotly_chart(
        fig_tt, on_select="rerun", key="tracker_conf_click",
        use_container_width=True,
    )
    inline_frame_panel(
        event_tt, "tracker_conf_click", all_frames,
        frames_dir if frames_available else None,
        det_df, track_df, depth_df, nodes_df, edges_df,
    )

    st.divider()

    st.subheader("Object Count Consistency")
    st.plotly_chart(
        plot_object_count_consistency(det_df, track_frame_df, sg_frame_df, all_frames),
        use_container_width=True,
    )

    st.subheader("Object Presence by Module")
    mismatch_df, _highlight = _build_mismatch_df()
    if not mismatch_df.empty:
        st.dataframe(
            mismatch_df[["object_id","det_first","det_last","track_first","track_last",
                          "in_detection","in_tracking"]].style.apply(_highlight, axis=1),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------

_DEPTH_FLAG_NAMES = [
    "depth_jump_flag", "depth_coherence_flag", "depth_validity_flag",
    "raw_depth_trigger", "any_depth_trigger",
]


def render_depth():
    _render_inspect_banner("dep")
    if not SHOW_DEPTH_TAB:
        st.info(
            "Depth not integrated yet — coming soon.  \n"
            "Set `SHOW_DEPTH_TAB = True` at the top of `dashboard.py` once the "
            "upstream depth module produces aligned files."
        )
        return
    st.header("Depth")
    if depth_df.empty:
        st.info("No depth data available for this sequence.")
        return

    all_depth_obj_ids = sorted(depth_df["object_id"].unique().tolist())

    # ── Filters ─────────────────────────────────────────────────────────────
    def _dep_sel_all_obj():
        for oid in all_depth_obj_ids:
            st.session_state[f"dep_obj_{oid}"] = True

    def _dep_desel_all_obj():
        for oid in all_depth_obj_ids:
            st.session_state[f"dep_obj_{oid}"] = False

    def _dep_sel_all_flg():
        for fn in _DEPTH_FLAG_NAMES:
            st.session_state[f"dep_flg_{fn}"] = True

    def _dep_desel_all_flg():
        for fn in _DEPTH_FLAG_NAMES:
            st.session_state[f"dep_flg_{fn}"] = False

    with st.expander("Filters", expanded=False):
        c_obj, c_flg = st.columns(2)
        with c_obj:
            st.markdown("**Object IDs**")
            b1, b2 = st.columns(2)
            with b1:
                st.button("Select all", key="dep_sel_all_obj",
                          on_click=_dep_sel_all_obj, use_container_width=True)
            with b2:
                st.button("Deselect all", key="dep_desel_all_obj",
                          on_click=_dep_desel_all_obj, use_container_width=True)
            for oid in all_depth_obj_ids:
                if f"dep_obj_{oid}" not in st.session_state:
                    st.session_state[f"dep_obj_{oid}"] = False
                st.checkbox(oid, key=f"dep_obj_{oid}")
        with c_flg:
            st.markdown("**Depth flags**")
            b3, b4 = st.columns(2)
            with b3:
                st.button("Select all", key="dep_sel_all_flg",
                          on_click=_dep_sel_all_flg, use_container_width=True)
            with b4:
                st.button("Deselect all", key="dep_desel_all_flg",
                          on_click=_dep_desel_all_flg, use_container_width=True)
            for fn in _DEPTH_FLAG_NAMES:
                if f"dep_flg_{fn}" not in st.session_state:
                    st.session_state[f"dep_flg_{fn}"] = False
                st.checkbox(fn, key=f"dep_flg_{fn}")

    selected_depth_objects = [
        oid for oid in all_depth_obj_ids if st.session_state.get(f"dep_obj_{oid}", False)
    ]
    selected_depth_flags = [
        fn for fn in _DEPTH_FLAG_NAMES if st.session_state.get(f"dep_flg_{fn}", False)
    ]

    # ── Depth timeline ───────────────────────────────────────────────────────
    fig_depth, _ = plot_depth_timeline(
        depth_df, gt_start_frame, gt_end_frame,
        selected_objects=selected_depth_objects,
        selected_flags=selected_depth_flags or None,
    )
    event_depth = st.plotly_chart(
        fig_depth, on_select="rerun", key="depth_median_click",
        use_container_width=True,
    )
    inline_frame_panel(
        event_depth, "depth_median_click", all_frames,
        frames_dir if frames_available else None,
        det_df, track_df, depth_df, nodes_df, edges_df,
    )


# ---------------------------------------------------------------------------

def render_scene_graph():
    _render_inspect_banner("sg")
    if not SHOW_SCENE_GRAPH_TAB:
        st.info(
            "Scene Graph not integrated yet — coming soon.  \n"
            "Set `SHOW_SCENE_GRAPH_TAB = True` at the top of `dashboard.py` once the "
            "upstream scene-graph module produces aligned files."
        )
        return
    st.header("Scene Graph")
    if nodes_df.empty:
        st.info("No scene graph data available for this sequence.")
        return

    all_sg_ids       = sorted(nodes_df["object_id"].unique())
    unique_relations = (
        sorted(edges_df["relation"].dropna().unique().tolist())
        if not edges_df.empty else []
    )
    unique_sources = (
        sorted(edges_df["source"].dropna().unique().tolist())
        if not edges_df.empty and "source" in edges_df.columns else []
    )

    # ── Filters ─────────────────────────────────────────────────────────────
    def _sg_sel_all_obj():
        for oid in all_sg_ids:
            st.session_state[f"sg_obj_{oid}"] = True

    def _sg_desel_all_obj():
        for oid in all_sg_ids:
            st.session_state[f"sg_obj_{oid}"] = False

    def _sg_sel_all_rel():
        for rel in unique_relations:
            st.session_state[f"sg_rel_{rel}"] = True

    def _sg_desel_all_rel():
        for rel in unique_relations:
            st.session_state[f"sg_rel_{rel}"] = False

    with st.expander("Filters", expanded=False):
        c_obj, c_rel, c_src = st.columns(3)
        with c_obj:
            st.markdown("**Object IDs**")
            b1, b2 = st.columns(2)
            with b1:
                st.button("Select all", key="sg_sel_all_obj",
                          on_click=_sg_sel_all_obj, use_container_width=True)
            with b2:
                st.button("Deselect all", key="sg_desel_all_obj",
                          on_click=_sg_desel_all_obj, use_container_width=True)
            for oid in all_sg_ids:
                if f"sg_obj_{oid}" not in st.session_state:
                    st.session_state[f"sg_obj_{oid}"] = False
                st.checkbox(oid, key=f"sg_obj_{oid}")
        with c_rel:
            st.markdown("**Relation types**")
            b3, b4 = st.columns(2)
            with b3:
                st.button("Select all", key="sg_sel_all_rel",
                          on_click=_sg_sel_all_rel, use_container_width=True)
            with b4:
                st.button("Deselect all", key="sg_desel_all_rel",
                          on_click=_sg_desel_all_rel, use_container_width=True)
            for rel in unique_relations:
                if f"sg_rel_{rel}" not in st.session_state:
                    st.session_state[f"sg_rel_{rel}"] = (rel == "near")
                st.checkbox(rel, key=f"sg_rel_{rel}")
        with c_src:
            st.markdown("**Edge source**")
            if len(unique_sources) > 1:
                st.selectbox(
                    "Source", options=["all sources"] + unique_sources,
                    key="sg_source_filter",
                )
            else:
                st.caption("Only one source type." if unique_sources else "No edges.")

    selected_sg_objects   = [
        oid for oid in all_sg_ids      if st.session_state.get(f"sg_obj_{oid}", False)
    ]
    selected_sg_relations = [
        rel for rel in unique_relations if st.session_state.get(f"sg_rel_{rel}", rel == "near")
    ]
    effective_relations = selected_sg_relations if selected_sg_relations else ["near"]
    selected_sg_source  = st.session_state.get("sg_source_filter", "all sources")

    display_nodes = (
        nodes_df[nodes_df["object_id"].isin(selected_sg_objects)]
        if selected_sg_objects else nodes_df
    )
    filtered_edges = edges_df.copy() if not edges_df.empty else pd.DataFrame()
    if not filtered_edges.empty and selected_sg_source not in ("all sources", None):
        filtered_edges = filtered_edges[filtered_edges["source"] == selected_sg_source]

    # ── Relation-mix bar ─────────────────────────────────────────────────────
    if not edges_df.empty:
        st.subheader("Relation Mix")
        st.plotly_chart(
            plot_relation_mix_bar(filtered_edges if not filtered_edges.empty else edges_df),
            use_container_width=True,
            config={"displayModeBar": False},
            key="sg_relation_mix",
        )

    # ── Metrics ──────────────────────────────────────────────────────────────
    flicker_df_sg = object_flicker_rate(display_nodes, all_frames)
    disp_df, disp_summary = mean_3d_displacement(display_nodes)
    jac_df, jac_summary   = neighbor_jaccard(
        filtered_edges, display_nodes, relation_types=effective_relations,
    )
    ef_df, ef_mat = edge_flicker(
        filtered_edges, all_frames, relation_types=effective_relations,
    )

    st.subheader("Object Flicker Rate")
    c_a, c_b = st.columns([1, 2])
    with c_a:
        if not flicker_df_sg.empty:
            st.dataframe(flicker_df_sg, use_container_width=True, hide_index=True)
    with c_b:
        if not flicker_df_sg.empty:
            fig_fl = px.bar(flicker_df_sg, x="object_id", y="flicker_rate",
                            title="Flicker Rate per Object", height=250)
            st.plotly_chart(fig_fl, use_container_width=True)

    st.subheader("Inspect Absent Frames")
    sel_oid_opts = (
        sorted(display_nodes["object_id"].unique().tolist())
        if not display_nodes.empty else all_sg_ids
    )
    sel_oid = st.selectbox("Inspect absent frames for:", options=sel_oid_opts,
                           key="absent_oid_select")
    if sel_oid:
        absent_frames_viewer(
            sel_oid, all_frames, nodes_df,
            frames_dir if frames_available else None,
            det_df, track_df, depth_df, edges_df,
        )
    st.divider()

    st.subheader("3D Displacement")
    fig_disp, stats_disp = plot_sg_displacement(disp_df, all_sg_ids)
    c_chart, c_stats = st.columns([3, 1])
    with c_chart:
        event_disp = st.plotly_chart(
            fig_disp, on_select="rerun", key="sg_disp_click", use_container_width=True,
        )
    with c_stats:
        if fig_disp.data:
            render_stats_table(stats_disp)
    inline_frame_panel(event_disp, "sg_disp_click", all_frames,
                       frames_dir if frames_available else None,
                       det_df, track_df, depth_df, nodes_df, edges_df)
    if not disp_summary.empty:
        st.dataframe(disp_summary, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader(f"Neighbor Jaccard — {', '.join(effective_relations)}")
    fig_jac, stats_jac = plot_sg_jaccard(jac_df, all_sg_ids)
    c_chart, c_stats = st.columns([3, 1])
    with c_chart:
        event_jac = st.plotly_chart(
            fig_jac, on_select="rerun", key="sg_jac_click", use_container_width=True,
        )
    with c_stats:
        if fig_jac.data:
            render_stats_table(stats_jac)
    inline_frame_panel(event_jac, "sg_jac_click", all_frames,
                       frames_dir if frames_available else None,
                       det_df, track_df, depth_df, nodes_df, edges_df)
    if not jac_summary.empty:
        st.dataframe(jac_summary, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Edge Flicker Rate")
    c_c, c_d = st.columns([1, 2])
    with c_c:
        if not ef_df.empty:
            st.dataframe(ef_df, use_container_width=True, hide_index=True)
    with c_d:
        if not ef_mat.empty and len(ef_mat) <= 10:
            st.plotly_chart(plot_edge_flicker_heatmap(ef_mat), use_container_width=True)

    st.divider()
    st.subheader("Auto-interpretation")
    st.markdown(auto_interpret(
        flicker_df_sg, disp_summary, jac_summary, ef_df,
        track_df, gt_start_s, gt_end_s,
        relation_types=effective_relations,
    ))


# ---------------------------------------------------------------------------

def _render_sg_legend(sg_nodes: list[dict]) -> None:
    if not sg_nodes:
        st.caption("No scene graph nodes on this frame.")
        return
    from frame_viewer import _color_for_label
    seen = {}
    for n in sg_nodes:
        lbl = n.get("label") or n.get("object_id", "")
        if lbl not in seen:
            seen[lbl] = _color_for_label(lbl)
    chips = "&nbsp;&nbsp;".join(
        f'<span style="color:{c};font-size:1.1em;">●</span> {lbl}'
        for lbl, c in seen.items()
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_frame_viewer():
    _render_inspect_banner("fv")
    st.header("Frame Viewer")

    if not frames_available:
        st.warning(
            f"Frames folder not found: `{frames_dir}`  \n"
            f"Extract frames first:  \n"
            f"```\npython dashboard/extract_frames.py --folder-name {sid}\n```"
        )
        return

    # ── View mode toggle ─────────────────────────────────────────────────────
    c_view, c_cap = st.columns([1, 4])
    with c_view:
        view_mode = st.radio(
            "View",
            options=["Bboxes", "Scene graph"],
            index=0,
            horizontal=True,
            key="frame_viewer_view_mode",
            label_visibility="collapsed",
        )
    with c_cap:
        st.caption("Switch between bbox overlays and scene graph overlays on the same frame.")

    frame_min = min(all_frames)
    frame_max = max(all_frames)
    default   = st.session_state.get("frame_viewer_frame", frame_min)
    default   = max(frame_min, min(int(default), frame_max))

    # ── Flag strip above slider (Change 4) ──────────────────────────────────
    strip_fig = plot_flag_strip(track_frame_df, frame_max)
    st.plotly_chart(
        strip_fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="flag_strip",
    )

    selected = st.slider("Frame ID", frame_min, frame_max, default)
    st.session_state["frame_viewer_frame"] = selected

    # ── Collect frame data ───────────────────────────────────────────────────
    det_row  = det_df[det_df["frame_id"] == selected] if not det_df.empty else pd.DataFrame()
    dets     = det_row["detections"].values[0] if not det_row.empty else []
    t_rows   = track_df[track_df["frame_id"] == selected] if not track_df.empty else pd.DataFrame()
    tracked  = [
        {"object_id": r.object_id, "bbox_xyxy": r.bbox_xyxy,
         "tracker_status": r.tracker_status, "tracker_confidence": r.tracker_confidence}
        for _, r in t_rows.iterrows()
    ] if not t_rows.empty else []
    d_rows        = depth_df[depth_df["frame_id"] == selected] if not depth_df.empty else pd.DataFrame()
    depth_flagged: set[str] = set()
    if not d_rows.empty:
        for _, r in d_rows.iterrows():
            if (r.get("any_depth_trigger") or r.get("depth_jump_flag") or
                    not r.get("depth_validity_flag", True) or
                    not r.get("depth_coherence_flag", True)):
                depth_flagged.add(r["object_id"])

    # ── Two-column layout: image + info card ────────────────────────────────
    col_img, col_info = st.columns([2, 1])

    with col_img:
        fp = find_frame_path(frames_dir, selected)
        if fp:
            if view_mode == "Bboxes":
                img = draw_bboxes(fp, dets, tracked, depth_flagged)
                st.image(img, use_container_width=True)
                # Legend strip (Change 3)
                st.markdown(
                    '<span style="color:#2ca02c;font-size:1.1em;">■</span> DET — Grounding DINO&nbsp;&nbsp;'
                    '<span style="color:#1f77b4;font-size:1.1em;">■</span> TRK — Tracker&nbsp;&nbsp;'
                    '<span style="color:#d62728;font-size:1.1em;">■</span> Depth-flagged',
                    unsafe_allow_html=True,
                )
            elif not SHOW_SCENE_GRAPH_TAB:
                st.image(fp, use_container_width=True)
                st.caption("Scene graph not available (feature flag off).")
            elif nodes_df.empty:
                st.image(fp, use_container_width=True)
                st.caption("No scene graph nodes for this sequence.")
            else:  # Scene graph
                n_rows = nodes_df[nodes_df["frame_id"] == selected]
                e_rows = edges_df[edges_df["frame_id"] == selected] if not edges_df.empty else pd.DataFrame()
                sg_nodes = n_rows.to_dict(orient="records") if not n_rows.empty else []
                sg_edges = (
                    e_rows.rename(columns={"from_id": "from", "to_id": "to"})
                          .to_dict(orient="records")
                    if not e_rows.empty else []
                )
                img = draw_scene_graph_overlay(fp, sg_nodes, sg_edges)
                st.image(img, use_container_width=True)
                # Scene graph legend (per-label colors actually present this frame)
                _render_sg_legend(sg_nodes)
        else:
            st.warning(f"Image not found for frame {selected}.")

    with col_info:
        render_frame_info_card(
            selected, det_df, track_df, depth_df, nodes_df, edges_df, llm_findings,
        )


# ===========================================================================
# MAIN — Six tabs
# ===========================================================================

tab_ov, tab_det, tab_trk, tab_dep, tab_sg, tab_fv = st.tabs([
    "📊 Overview",
    "🔍 Detection",
    "🎯 Tracking",
    "📏 Depth",
    "🕸 Scene Graph",
    "🖼 Frame Viewer",
])

with tab_ov:  render_overview()
with tab_det: render_detection()
with tab_trk: render_tracking()
with tab_dep: render_depth()
with tab_sg:  render_scene_graph()
with tab_fv:  render_frame_viewer()


# cd /Users/guray/Desktop/REFLECT_Group_Project-1

# streamlit run dashboard/dashboard.py