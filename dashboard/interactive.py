"""
interactive.py — clickable timeline panels, frame info card, absent-frames viewer.

Public API:
  render_stats_table(stats)            – compact stats table (deprecated for timelines;
                                         boxplot subplot is the new reference)
  render_frame_info_card(...)          – compact human-readable card for Frame Viewer
  render_frame_panel(frame_id, ...)    – image + JSON dump (used by inline_frame_panel)
  inline_frame_panel(event, key, ...) – handle click state + render panel below a chart
  absent_frames_viewer(object_id, ...) – iterate over absent-frame viewer
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from frame_viewer import draw_bboxes
from utils import find_frame_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_points(event) -> list:
    """Extract selected points from st.plotly_chart on_select return value."""
    if event is None:
        return []
    try:
        pts = event.selection.points
        return pts if pts else []
    except AttributeError:
        pass
    try:
        pts = event["selection"]["points"]
        return pts if pts else []
    except (KeyError, TypeError):
        return []


def _get_x(point) -> int | None:
    try:
        x = point.get("x") if isinstance(point, dict) else getattr(point, "x", None)
        if x is None:
            return None
        x_int = int(round(float(x)))
        return x_int if x_int >= 0 else None
    except (ValueError, TypeError):
        return None


def _nearest_frame(x_val: int, frame_ids: list[int]) -> int:
    if not frame_ids:
        return 0
    arr = np.array(frame_ids)
    return int(np.argmin(np.abs(arr - x_val)))


def _extract_label(oid: str) -> str:
    """'apple-1' → 'apple'; falls back to full oid if no trailing digit."""
    parts = oid.rsplit("-", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else oid


def _label_counts_str(label_counts: dict[str, int]) -> str:
    if not label_counts:
        return "none"
    return ", ".join(f"{v} {k}" for k, v in sorted(label_counts.items()))


# ---------------------------------------------------------------------------
# Stats table  (Deprecated for tracker/detection timelines; boxplot subplot
#               is the new reference — kept for scene graph and other panels)
# ---------------------------------------------------------------------------

def render_stats_table(stats: dict) -> None:
    """
    Render a compact stats table.
    Accepts:
      - single-metric dict: {"mean": 0.12, "p5": …, …}
      - multi-metric dict:  {"tracker_confidence": {stats}, "displacement_px": {stats}, …}

    Deprecated for tracker/detection timelines; boxplot subplot is the new reference.
    """
    if not stats:
        return
    if all(isinstance(v, dict) for v in stats.values()):
        rows = []
        for metric, s in stats.items():
            row = {"metric": metric}
            row.update({
                k: f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "—"
                for k, v in s.items()
            })
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame([
            {"stat": k,
             "value": f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "—"}
            for k, v in stats.items()
        ])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Frame info card (Change 2) — compact human-readable right panel
# ---------------------------------------------------------------------------

_CHIP = (
    'background:{bg};color:{fg};padding:2px 6px;'
    'border-radius:4px;margin-right:4px;font-size:0.82em;'
)

_FLAG_STYLES: dict[str, tuple[str, str]] = {
    "bbox_size_change_flag": ("#fef3cd", "#856404"),
    "drift_flag":            ("#fde8e8", "#c0392b"),
    "frame_counter_K_flag":  ("#f0f0f0", "#555555"),
    "any_recovery_trigger":  ("#f3e8fe", "#6b3fa0"),
}

# ---------------------------------------------------------------------------
# HTML card helpers (Bug 4)
# ---------------------------------------------------------------------------

_TBL_CSS = """\
<style>
.fc-table{border-collapse:collapse;width:100%;font-size:12px;margin:2px 0 8px 0}
.fc-table th{background:#f5f5f5;padding:3px 6px;text-align:left;border-bottom:1px solid #ddd;font-weight:500;color:#444}
.fc-table td{padding:3px 6px;border-bottom:1px solid #f0f0f0;vertical-align:top}
.fc-table tr:last-child td{border-bottom:none}
.fc-chip{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;font-weight:500}
.fc-sec{font-size:12px;font-weight:500;color:#666;margin:8px 0 2px 0;text-transform:uppercase;letter-spacing:0.5px}
</style>"""


def _sec(title: str) -> str:
    return f'<div class="fc-sec">{title}</div>'


def _chip(text: str, bg: str, fg: str) -> str:
    return f'<span class="fc-chip" style="background:{bg};color:{fg}">{text}</span>'


def _tbl(headers: list[str], rows: list[list[str]]) -> str:
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f'<table class="fc-table"><tr>{ths}</tr>{trs}</table>'


def render_frame_info_card(
    frame_id: int,
    det_df: pd.DataFrame,
    track_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    sg_nodes_df: pd.DataFrame,
    sg_edges_df: pd.DataFrame,
    llm_findings: list | None = None,
) -> None:
    det_row = det_df[det_df["frame_id"] == frame_id] if not det_df.empty else pd.DataFrame()
    t_rows  = track_df[track_df["frame_id"] == frame_id] if not track_df.empty else pd.DataFrame()
    d_rows  = depth_df[depth_df["frame_id"] == frame_id] if not depth_df.empty else pd.DataFrame()
    n_rows  = sg_nodes_df[sg_nodes_df["frame_id"] == frame_id] if not sg_nodes_df.empty else pd.DataFrame()
    e_rows  = sg_edges_df[sg_edges_df["frame_id"] == frame_id] if not sg_edges_df.empty else pd.DataFrame()

    ts      = float(det_row["timestamp"].values[0]) if not det_row.empty else None
    ts_str  = f"{ts:.3f}s" if ts is not None else "—"
    det_ran = (not det_row.empty) and bool(det_row["detector_ran"].values[0])

    with st.container(border=True):
        st.markdown(_TBL_CSS, unsafe_allow_html=True)
        st.markdown(f"**Frame {frame_id}** · _{ts_str}_")

        parts: list[str] = []

        # ── Detected ─────────────────────────────────────────────────────────
        if det_ran:
            dets    = det_row["detections"].values[0] or []
            prompts = det_row["prompts_used"].values[0] or []
            rows_det = []
            for d in dets:
                oid  = d.get("object_id", "?")
                lbl  = d.get("label") or _extract_label(oid)
                conf = d.get("confidence")
                conf_str = f"{float(conf):.2f}" if conf is not None else "—"
                sel  = "✓" if d.get("is_selected") else "·"
                prompt_match = "—"
                for p in prompts:
                    if isinstance(p, str) and lbl.lower() in p.lower():
                        prompt_match = p[:30] + ("…" if len(p) > 30 else "")
                        break
                rows_det.append([oid, lbl, prompt_match, conf_str, sel])
            parts.append(_sec(f"Detected · {len(dets)} objects · {_chip('DINO ran', '#e0f3e0', '#1a6b1a')}"))
            if rows_det:
                parts.append(_tbl(["object_id", "label", "prompt", "conf", "sel"], rows_det))
        elif not det_row.empty:
            ran_before = det_df[(det_df["detector_ran"]) & (det_df["frame_id"] < frame_id)]
            ago_str = ""
            if not ran_before.empty:
                last = int(ran_before["frame_id"].max())
                ago_str = f" · last at frame {last} ({frame_id - last} ago)"
            parts.append(_sec(f"Detected · DINO did not run{ago_str}"))

        # ── Tracked ──────────────────────────────────────────────────────────
        if not t_rows.empty:
            _S_COLORS = {
                "ok":        ("#e0f3e0", "#1a6b1a"),
                "recovered": ("#e0f3e0", "#1a6b1a"),
                "drifting":  ("#fef3cd", "#856404"),
                "occluded":  ("#fef3cd", "#856404"),
                "lost":      ("#fde8e8", "#c0392b"),
            }
            rows_trk = []
            for _, r in t_rows.iterrows():
                oid    = r.get("object_id", "?")
                lbl    = _extract_label(str(oid))
                status = str(r.get("tracker_status", "ok"))
                s_bg, s_fg = _S_COLORS.get(status, ("#f0f0f0", "#555"))
                conf  = r.get("tracker_confidence")
                conf_str = (
                    f"{float(conf):.2f}"
                    if conf is not None and not (isinstance(conf, float) and np.isnan(conf))
                    else "—"
                )
                ratio = r.get("bbox_area_ratio_to_init")
                ratio_str = (
                    f"{float(ratio):.2f}"
                    if ratio is not None and not (isinstance(ratio, float) and np.isnan(ratio))
                    else "—"
                )
                fsr = r.get("frames_since_redetect")
                fsr_str = (
                    str(int(fsr))
                    if fsr is not None and not (isinstance(fsr, float) and np.isnan(fsr))
                    else "—"
                )
                rows_trk.append([oid, lbl, _chip(status, s_bg, s_fg), conf_str, ratio_str, fsr_str])
            parts.append(_sec(f"Tracked · {len(t_rows)} objects"))
            parts.append(_tbl(["object_id", "label", "status", "conf", "bbox ratio", "redetect"],
                               rows_trk))

        # ── SG nodes ─────────────────────────────────────────────────────────
        if not n_rows.empty:
            rows_sg = []
            for _, r in n_rows.iterrows():
                oid   = r.get("object_id", "?")
                lbl   = r.get("label") or _extract_label(str(oid))
                state = str(r.get("state") or "—")
                dm    = r.get("depth_used_m")
                dm_str = (
                    f"{float(dm):.2f} m"
                    if dm is not None and not (isinstance(dm, float) and np.isnan(dm))
                    else "—"
                )
                px, py, pz = r.get("pos_x"), r.get("pos_y"), r.get("pos_z")
                pos_str = (
                    f"({float(px):.2f}, {float(py):.2f}, {float(pz):.2f})"
                    if all(v is not None and not (isinstance(v, float) and np.isnan(v))
                           for v in [px, py, pz])
                    else "—"
                )
                rows_sg.append([oid, lbl, state, dm_str, pos_str])
            parts.append(_sec(f"Scene graph nodes · {len(n_rows)}"))
            parts.append(_tbl(["object_id", "label", "state", "depth", "3D pos"], rows_sg))

        # ── Relations ─────────────────────────────────────────────────────────
        if not e_rows.empty:
            MAX_REL = 6
            rows_rel = []
            for _, er in e_rows.head(MAX_REL).iterrows():
                from_id = er.get("from_id", "?")
                to_id   = er.get("to_id",   "?")
                rel     = str(er.get("relation") or "?")
                dist    = er.get("distance_3d_m")
                dist_str = (
                    f"{float(dist):.2f} m"
                    if dist is not None and not (isinstance(dist, float) and np.isnan(dist))
                    else "—"
                )
                src = str(er.get("source") or "—")
                rows_rel.append([from_id, rel, to_id, dist_str, src])
            extra = f" · <i>+{len(e_rows) - MAX_REL} more</i>" if len(e_rows) > MAX_REL else ""
            parts.append(_sec(f"Relations · {len(e_rows)} edges{extra}"))
            parts.append(_tbl(["from", "relation", "to", "dist", "source"], rows_rel))

        # ── Depth ─────────────────────────────────────────────────────────────
        if not d_rows.empty:
            _DP = {
                "depth_jump_flag":      (True,  "depth_jump"),
                "any_depth_trigger":    (True,  "any_trigger"),
                "depth_coherence_flag": (False, "coherence"),
                "depth_validity_flag":  (False, "validity"),
            }
            rows_dep = []
            for _, dr in d_rows.iterrows():
                oid = dr.get("object_id", "?")
                med = dr.get("depth_median_m")
                iqr = dr.get("depth_iqr_m")
                vpr = dr.get("valid_depth_pixel_ratio")
                med_str = f"{float(med):.2f} m" if med is not None and not (isinstance(med, float) and np.isnan(med)) else "—"
                iqr_str = f"{float(iqr):.2f} m" if iqr is not None and not (isinstance(iqr, float) and np.isnan(iqr)) else "—"
                vpr_str = f"{float(vpr):.2f}"   if vpr is not None and not (isinstance(vpr, float) and np.isnan(vpr)) else "—"
                active = [
                    _chip(short, "#fef3cd", "#856404")
                    for fc, (bad_when, short) in _DP.items()
                    if dr.get(fc) == bad_when
                ]
                flags_html = " ".join(active) if active else "—"
                rows_dep.append([oid, med_str, iqr_str, vpr_str, flags_html])
            parts.append(_sec(f"Depth · {len(d_rows)} objects"))
            parts.append(_tbl(["object_id", "median", "iqr", "valid px", "flags"], rows_dep))

        if parts:
            st.markdown("".join(parts), unsafe_allow_html=True)

        # ── LLM findings for this frame ──────────────────────────────────────
        if llm_findings:
            frame_hits = [
                f for f in llm_findings
                if f.get("evidence", {}).get("frame_id") == frame_id
            ]
            for hit in frame_hits:
                st.markdown(f"💡 _{hit.get('claim', '')}_")

        # ── Raw debug ────────────────────────────────────────────────────────
        with st.expander("Raw rows for this frame"):
            if not det_row.empty:
                st.caption("Detection")
                st.json(det_row.iloc[0].to_dict())
            if not t_rows.empty:
                st.caption("Tracking")
                st.json(t_rows.to_dict(orient="records"))
            if not d_rows.empty:
                st.caption("Depth")
                st.json(d_rows.to_dict(orient="records"))
            if not n_rows.empty:
                st.caption("Scene Graph")
                st.json({
                    "nodes": n_rows.to_dict(orient="records"),
                    "edges": e_rows.to_dict(orient="records") if not e_rows.empty else [],
                })


# ---------------------------------------------------------------------------
# Frame panel (used by inline_frame_panel — unchanged from v2)
# ---------------------------------------------------------------------------

def render_frame_panel(
    frame_id: int,
    frames_dir: str | None,
    det_df: pd.DataFrame,
    track_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    caption_extra: str = "",
) -> None:
    """Render image-with-overlays (left) + compact info card (right)."""
    det_row = det_df[det_df["frame_id"] == frame_id] if not det_df.empty else pd.DataFrame()
    dets    = det_row["detections"].values[0] if not det_row.empty else []
    ts      = det_row["timestamp"].values[0]  if not det_row.empty else None

    t_rows  = track_df[track_df["frame_id"] == frame_id] if not track_df.empty else pd.DataFrame()
    tracked = [
        {"object_id": r.object_id, "bbox_xyxy": r.bbox_xyxy,
         "tracker_status": r.tracker_status, "tracker_confidence": r.tracker_confidence}
        for _, r in t_rows.iterrows()
    ] if not t_rows.empty else []

    d_rows = depth_df[depth_df["frame_id"] == frame_id] if not depth_df.empty else pd.DataFrame()
    depth_flagged: set[str] = set()
    if not d_rows.empty:
        for _, r in d_rows.iterrows():
            if r.get("depth_jump_flag") or r.get("any_depth_trigger") or not r.get("depth_validity_flag", True):
                depth_flagged.add(r["object_id"])

    ts_str  = f"  t={ts:.3f}s" if ts is not None else ""
    caption = f"Frame {frame_id}{ts_str}"
    if caption_extra:
        caption += f"  •  {caption_extra}"
    flags_on = []
    if not d_rows.empty and d_rows["depth_jump_flag"].any():
        flags_on.append("depth_jump")
    if not t_rows.empty and (t_rows["tracker_status"] != "ok").any():
        flags_on.append("tracker=" + t_rows["tracker_status"].iloc[0])
    if flags_on:
        caption += f"  🚩 {', '.join(flags_on)}"

    col_img, col_info = st.columns([2, 1])
    with col_img:
        if frames_dir and Path(frames_dir).exists():
            fp = find_frame_path(frames_dir, frame_id)
            if fp:
                img = draw_bboxes(fp, dets, tracked, depth_flagged)
                st.image(img, caption=caption, use_container_width=True)
            else:
                st.warning(f"Image not found: frame {frame_id}")
                st.caption(caption)
        else:
            st.warning("Frames folder not available.")
            st.caption(caption)

    with col_info:
        render_frame_info_card(
            frame_id, det_df, track_df, depth_df,
            nodes_df, edges_df,
            llm_findings=st.session_state.get("llm_findings"),
        )


# ---------------------------------------------------------------------------
# Inline frame panel (F3)
# ---------------------------------------------------------------------------

def inline_frame_panel(
    event,
    key: str,
    frame_ids: list[int],
    frames_dir: str | None,
    det_df: pd.DataFrame,
    track_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> None:
    """
    Called immediately after st.plotly_chart(..., on_select="rerun", key=key).
    Reads selection, stores index in session_state[f"{key}_idx"], renders
    Prev / Close / Next controls and a frame viewer below the chart.
    """
    points = _safe_points(event)
    if points:
        x_val = _get_x(points[0])
        if x_val is not None:
            last_x_key = f"{key}_last_event_x"
            # Streamlit preserves on_select state across reruns. Ignore stale
            # events whose x-value matches the last one we processed; only react
            # when the user has actually clicked a different point on the chart.
            if st.session_state.get(last_x_key) != x_val:
                st.session_state[f"{key}_idx"] = _nearest_frame(x_val, frame_ids)
                st.session_state[last_x_key] = x_val

    state_key = f"{key}_idx"
    if state_key not in st.session_state:
        return

    idx      = max(0, min(st.session_state[state_key], len(frame_ids) - 1))
    frame_id = frame_ids[idx]

    st.markdown("---")
    st.markdown(f"**🔍 Inline viewer — Frame {frame_id}** &nbsp; ({idx + 1}/{len(frame_ids)})")

    col_prev, col_close, col_next, _ = st.columns([1, 1, 1, 9])
    with col_prev:
        if st.button("← Prev", key=f"{key}_prev", disabled=(idx == 0)):
            st.session_state[state_key] = idx - 1
            st.rerun()
    with col_close:
        if st.button("✕ Close", key=f"{key}_close"):
            del st.session_state[state_key]
            st.session_state.pop(f"{key}_last_event_x", None)
            st.rerun()
    with col_next:
        if st.button("Next →", key=f"{key}_next", disabled=(idx >= len(frame_ids) - 1)):
            st.session_state[state_key] = idx + 1
            st.rerun()

    render_frame_panel(frame_id, frames_dir, det_df, track_df, depth_df, nodes_df, edges_df)


# ---------------------------------------------------------------------------
# Absent-frames viewer (F5)
# ---------------------------------------------------------------------------

def _diagnose_absent(
    frame_id: int,
    oid: str,
    det_df: pd.DataFrame,
    track_df: pd.DataFrame,
) -> str:
    in_det, det_conf = False, None
    if not det_df.empty:
        dr = det_df[det_df["frame_id"] == frame_id]
        if not dr.empty:
            for d in (dr["detections"].values[0] or []):
                if d.get("object_id") == oid:
                    in_det   = True
                    det_conf = d.get("confidence")
                    break

    in_track, track_status = False, None
    if not track_df.empty:
        tr = track_df[(track_df["frame_id"] == frame_id) & (track_df["object_id"] == oid)]
        if not tr.empty:
            in_track     = True
            track_status = tr["tracker_status"].values[0]

    if in_det and in_track:
        c = f"{det_conf:.3f}" if det_conf is not None else "?"
        return (
            f"🔴 **Possible SG bug**: Detector saw it (conf {c}), "
            f"tracker had it (status: {track_status}), but scene graph dropped it."
        )
    elif in_det and not in_track:
        c = f"{det_conf:.3f}" if det_conf is not None else "?"
        return f"ℹ️ Detector saw it (conf {c}) but tracker didn't — possibly mid-init."
    elif not in_det and in_track:
        if track_status == "lost":
            return "⚠️ Tracker lost it and detector confirmed it's gone — likely off-screen."
        return f"🟡 **Tracker hallucinating**: detector missed it but tracker reports status={track_status}."
    return "✅ Neither detector nor tracker saw it — object off-screen or occluded."


def _diagnose_absent_html(
    frame_id: int,
    oid: str,
    det_df: pd.DataFrame,
    track_df: pd.DataFrame,
) -> str:
    in_det, det_conf = False, None
    if not det_df.empty:
        dr = det_df[det_df["frame_id"] == frame_id]
        if not dr.empty:
            for d in (dr["detections"].values[0] or []):
                if d.get("object_id") == oid:
                    in_det   = True
                    det_conf = d.get("confidence")
                    break

    in_track, track_status = False, None
    if not track_df.empty:
        tr = track_df[(track_df["frame_id"] == frame_id) & (track_df["object_id"] == oid)]
        if not tr.empty:
            in_track     = True
            track_status = tr["tracker_status"].values[0]

    if in_det and in_track:
        diag = "Possible SG bug: detected + tracked but absent from scene graph"
        dc   = "#c0392b"
    elif in_det and not in_track:
        diag = "Detector saw it but tracker didn't — possibly mid-init"
        dc   = "#856404"
    elif not in_det and in_track:
        if track_status == "lost":
            diag = "Tracker lost it and detector confirmed gone — likely off-screen"
        else:
            diag = f"Tracker hallucination? detector missed it (status={track_status})"
        dc = "#856404"
    else:
        diag = "Off-screen or occluded — neither detector nor tracker saw it"
        dc   = "#1a6b1a"

    det_cell = (
        f"yes (conf {float(det_conf):.2f})" if in_det and det_conf is not None
        else ("yes" if in_det else "no")
    )
    trk_cell = f"yes (status={track_status})" if in_track else "no"
    rows = [
        ("detector saw it",  det_cell),
        ("tracker had it",   trk_cell),
        ("scene graph node", "absent"),
        ("diagnosis",        f'<span style="color:{dc};font-weight:500">{diag}</span>'),
    ]
    tds = "".join(
        f"<tr>"
        f"<td style='padding:3px 8px;border-bottom:1px solid #f0f0f0;"
        f"font-weight:500;color:#555;white-space:nowrap'>{sig}</td>"
        f"<td style='padding:3px 8px;border-bottom:1px solid #f0f0f0'>{val}</td>"
        f"</tr>"
        for sig, val in rows
    )
    return (
        f"<table style='border-collapse:collapse;width:100%;font-size:12px;margin:4px 0'>"
        f"{tds}</table>"
    )


def absent_frames_viewer(
    object_id: str,
    all_frames: list[int],
    nodes_df: pd.DataFrame,
    frames_dir: str | None,
    det_df: pd.DataFrame,
    track_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> None:
    state_key = f"flicker_absent_{object_id}_idx"

    present = (
        set(nodes_df[nodes_df["object_id"] == object_id]["frame_id"].tolist())
        if not nodes_df.empty else set()
    )
    absent = [f for f in all_frames if f not in present]

    if not absent:
        st.info(f"No absent frames for **{object_id}**.")
        return

    st.caption(f"{len(absent)} absent frame(s) for {object_id}")

    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    idx      = max(0, min(st.session_state[state_key], len(absent) - 1))
    frame_id = absent[idx]

    col_prev, col_close, col_next, _ = st.columns([1, 1, 1, 9])
    with col_prev:
        if st.button("← Prev", key=f"{state_key}_prev", disabled=(idx == 0)):
            st.session_state[state_key] = idx - 1
            st.rerun()
    with col_close:
        if st.button("✕ Close", key=f"{state_key}_close"):
            del st.session_state[state_key]
            st.rerun()
    with col_next:
        if st.button("Next →", key=f"{state_key}_next", disabled=(idx >= len(absent) - 1)):
            st.session_state[state_key] = idx + 1
            st.rerun()

    st.markdown(
        _diagnose_absent_html(frame_id, object_id, det_df, track_df),
        unsafe_allow_html=True,
    )
    render_frame_panel(frame_id, frames_dir, det_df, track_df, depth_df, nodes_df, edges_df,
                       caption_extra=f"absent: {object_id}")
