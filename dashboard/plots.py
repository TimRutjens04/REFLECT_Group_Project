"""Plotly figure builders — one per dashboard section.

Changes in v3:
  - plot_tracker_timeline:     4-row × 2-col layout; col2 = boxplot per metric;
                                shared_yaxes + spikemode="across" for hover context;
                                accepts selected_objects / selected_flags.
  - plot_detection_confidence: 1-row × 2-col layout; col2 = boxplot;
                                accepts selected_objects / selected_triggers.
  - plot_flag_strip:           New — thin strip of flag markers for Frame Viewer.

Functions that carry F4 (reference lines) return (fig, stats_dict).
Functions without F4 return just fig.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from metrics import reference_stats

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

TRIGGER_COLORS = {
    "init":                   "#1f77b4",
    "tracker_low_confidence": "#d62728",
    "depth_jump":             "#9467bd",
    "bbox_size_change":       "#ff7f0e",
    "drift":                  "#8c564b",
    "frame_counter_K":        "#7f7f7f",
    "manual":                 "#e377c2",
    "none":                   "#bcbd22",
}

TRIGGER_SYMBOLS = {
    "init":                   "star",
    "tracker_low_confidence": "triangle-up",
    "depth_jump":             "diamond",
    "bbox_size_change":       "square",
    "drift":                  "cross",
    "frame_counter_K":        "circle",
    "manual":                 "pentagon",
    "none":                   "x",
}

STATUS_COLORS = {
    "drifting": "orange",
    "lost":     "red",
    "recovered": "green",
    "occluded": "purple",
}

FLAG_COLORS = {
    "bbox_size_change_flag": "#ff7f0e",
    "drift_flag":            "#d62728",
    "frame_counter_K_flag":  "#7f7f7f",
    "any_recovery_trigger":  "#9467bd",
}


def _obj_color(oid: str, all_ids: list[str]) -> str:
    idx = all_ids.index(oid) if oid in all_ids else 0
    return _COLORS[idx % len(_COLORS)]


# ---------------------------------------------------------------------------
# Shape / annotation helpers
# ---------------------------------------------------------------------------

def _gt_band(fig, gt_start, gt_end, row=None, col=None):
    if gt_start is None or gt_end is None:
        return
    kwargs = dict(
        type="rect", xref="x", yref="paper",
        x0=gt_start, x1=gt_end, y0=0, y1=1,
        fillcolor="rgba(255,0,0,0.12)", line_width=0, layer="below",
    )
    if row is not None:
        kwargs["row"] = row
        kwargs["col"] = col
    fig.add_shape(**kwargs)


def _z_hover(frame_id, ts, metric, val, stats, extra="") -> str:
    mean_v = stats.get("mean", float("nan"))
    std_v  = stats.get("std",  float("nan"))
    if val is None or (isinstance(val, float) and np.isnan(val)):
        z_str = ""
    elif std_v > 0 and not np.isnan(mean_v):
        z = (val - mean_v) / std_v
        z_str = f"<br>z={z:+.2f}σ"
    else:
        z_str = ""
    ts_str  = f" t={ts:.3f}s" if ts is not None and not (isinstance(ts, float) and np.isnan(ts)) else ""
    val_str = f"{val:.4f}" if val is not None and not (isinstance(val, float) and np.isnan(val)) else "N/A"
    return f"frame={frame_id}{ts_str}<br>{metric}={val_str}{z_str}{extra}"


# ---------------------------------------------------------------------------
# Object count consistency (unchanged)
# ---------------------------------------------------------------------------

def plot_object_count_consistency(
    det_df: pd.DataFrame,
    track_frame_df: pd.DataFrame,
    sg_frame_df: pd.DataFrame,
    all_frames: list[int],
) -> go.Figure:
    fig = go.Figure()
    det_ran = det_df[det_df["detector_ran"]]
    fig.add_trace(go.Scatter(
        x=det_ran["frame_id"], y=det_ran["detection_count"],
        mode="lines+markers", name="detections (DINO frames)",
        line=dict(color="green"),
    ))
    if not track_frame_df.empty:
        fig.add_trace(go.Scatter(
            x=track_frame_df["frame_id"], y=track_frame_df["tracked_count"],
            mode="lines", name="tracked objects", line=dict(color="blue"),
        ))
    if not sg_frame_df.empty:
        fig.add_trace(go.Scatter(
            x=sg_frame_df["frame_id"], y=sg_frame_df["node_count"],
            mode="lines", name="scene graph nodes", line=dict(color="orange"),
        ))
    merged = pd.DataFrame({"frame_id": all_frames})
    if not track_frame_df.empty:
        merged = merged.merge(
            track_frame_df[["frame_id", "tracked_count"]].rename(columns={"tracked_count": "tc"}),
            on="frame_id", how="left",
        )
    else:
        merged["tc"] = float("nan")
    if not sg_frame_df.empty:
        merged = merged.merge(
            sg_frame_df[["frame_id", "node_count"]].rename(columns={"node_count": "nc"}),
            on="frame_id", how="left",
        )
    else:
        merged["nc"] = float("nan")
    for _, row in merged.iterrows():
        tc, nc = row.get("tc"), row.get("nc")
        if pd.notna(tc) and pd.notna(nc) and tc != nc:
            fig.add_vrect(
                x0=row["frame_id"] - 0.5, x1=row["frame_id"] + 0.5,
                fillcolor="rgba(255,200,0,0.25)", line_width=0, layer="below",
            )
    fig.update_layout(
        title="Object Count Consistency",
        xaxis_title="Frame ID", yaxis_title="Count",
        height=280, margin=dict(t=40, b=30),
        legend=dict(orientation="h"),
    )
    return fig


# ---------------------------------------------------------------------------
# Tracker Timeline — 4×2 subplot with boxplot (Change 6)
# ---------------------------------------------------------------------------

def plot_tracker_timeline(
    track_df: pd.DataFrame,
    det_df: pd.DataFrame,
    gt_start,
    gt_end,
    show_per_object: bool = True,
    selected_objects: list[str] | None = None,
    selected_flags: list[str] | None = None,
) -> tuple[go.Figure, dict]:
    """
    Layout: 4 rows × 2 cols.
      Row 1 col 1 — events strip (DINO + status transitions + selected flag markers)
      Row 1 col 2 — empty
      Rows 2-4 col 1 — metric lines (tracker_confidence / bbox_area_ratio / displacement)
      Rows 2-4 col 2 — boxplot (full sequence, unaffected by object filter)

    selected_objects=[] → empty annotation asking user to pick an object.
    selected_flags=[]   → no flag markers (DINO/status events always shown).
    Returns (fig, stats_dict).
    """
    if track_df.empty:
        return go.Figure(), {}

    if selected_objects is not None and len(selected_objects) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least one object in <b>Filters</b> above to see the timeline.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="gray"),
        )
        fig.update_layout(height=200)
        return fig, {}

    metrics_cfg = [
        ("tracker_confidence",      "Tracker Confidence"),
        ("bbox_area_ratio_to_init", "BBox Area Ratio to Init"),
        ("displacement_px",         "Displacement (px)"),
    ]
    N_metrics = len(metrics_cfg)
    all_ids   = sorted(track_df["object_id"].unique())
    disp_ids  = sorted(selected_objects) if selected_objects else all_ids
    x_min     = int(track_df["frame_id"].min())
    x_max     = int(track_df["frame_id"].max())

    subplot_titles = [
        "Events", "",
        "Tracker Confidence", "",
        "BBox Area Ratio to Init", "",
        "Displacement (px)", "",
    ]
    fig = make_subplots(
        rows=4, cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        column_widths=[0.88, 0.12],
        row_heights=[0.10, 0.30, 0.30, 0.30],
        vertical_spacing=0.05,
        horizontal_spacing=0.02,
        subplot_titles=subplot_titles,
    )

    # ── Row 1: Events strip ──────────────────────────────────────────────────
    det_ran = det_df[det_df["detector_ran"]]
    seen_triggers: set[str] = set()
    for _, drow in det_ran.iterrows():
        reason = drow.get("trigger_reason", "none")
        color  = TRIGGER_COLORS.get(reason, "#999999")
        sym    = TRIGGER_SYMBOLS.get(reason, "circle")
        fig.add_trace(go.Scatter(
            x=[drow["frame_id"]], y=[0.25],
            mode="markers",
            marker=dict(color=color, size=10, symbol=sym),
            name=f"DINO:{reason}", legendgroup=f"dino_{reason}",
            showlegend=(reason not in seen_triggers),
            hovertemplate=f"DINO trigger: {reason}<br>frame={drow['frame_id']}<extra></extra>",
        ), row=1, col=1)
        seen_triggers.add(reason)

    seen_status: set[str] = set()
    for oid in all_ids:
        sub    = track_df[track_df["object_id"] == oid].sort_values("frame_id")
        prev_s = None
        for _, r in sub.iterrows():
            s = r["tracker_status"]
            if s != prev_s and s != "ok":
                sc  = STATUS_COLORS.get(s, "gray")
                lk  = f"status→{s}"
                fig.add_trace(go.Scatter(
                    x=[r["frame_id"]], y=[0.75],
                    mode="markers",
                    marker=dict(color=sc, size=10, symbol="diamond"),
                    name=f"status→{s}", legendgroup=lk,
                    showlegend=(lk not in seen_status),
                    hovertemplate=f"{oid}: →{s}<br>frame={r['frame_id']}<extra></extra>",
                ), row=1, col=1)
                seen_status.add(lk)
            prev_s = s

    # Selected flag markers on events strip
    if selected_flags:
        flag_df = track_df.drop_duplicates(subset=["frame_id"])
        seen_flags: set[str] = set()
        for flag_col in selected_flags:
            if flag_col not in flag_df.columns:
                continue
            fired = flag_df[flag_df[flag_col] == True]
            if fired.empty:
                continue
            color = FLAG_COLORS.get(flag_col, "#aaaaaa")
            fig.add_trace(go.Scatter(
                x=fired["frame_id"].tolist(), y=[0.5] * len(fired),
                mode="markers",
                marker=dict(color=color, size=8, symbol="triangle-down"),
                name=flag_col, legendgroup=f"flag_{flag_col}",
                showlegend=(flag_col not in seen_flags),
                hovertemplate=f"{flag_col}<br>frame=%{{x}}<extra></extra>",
            ), row=1, col=1)
            seen_flags.add(flag_col)

    fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=2)

    # ── Rows 2-4: Metric lines + boxplots ───────────────────────────────────
    all_stats: dict[str, dict] = {}
    for row_idx, (col_name, _) in enumerate(metrics_cfg, 2):
        # Reference stats use full dataset
        stats = reference_stats(track_df[col_name].dropna())
        all_stats[col_name] = stats

        _gt_band(fig, gt_start, gt_end, row=row_idx, col=1)

        # Lines — filtered to selected_objects if provided
        disp_df = (
            track_df[track_df["object_id"].isin(disp_ids)]
            if disp_ids != all_ids else track_df
        )

        if show_per_object and not disp_df.empty:
            for oid in disp_ids:
                if oid not in disp_df["object_id"].values:
                    continue
                color = _obj_color(oid, all_ids)
                sub   = disp_df[disp_df["object_id"] == oid].sort_values("frame_id")
                vals  = sub[col_name].tolist()
                hover = [
                    _z_hover(r.frame_id, r.timestamp, col_name, v, stats,
                             extra=f"<br>status={r.tracker_status}")
                    for (_, r), v in zip(sub.iterrows(), vals)
                ]
                fig.add_trace(go.Scatter(
                    x=sub["frame_id"],
                    y=sub[col_name].where(pd.notna(sub[col_name]), None),
                    mode="lines", name=oid, legendgroup=oid,
                    showlegend=(row_idx == 2),
                    line=dict(color=color),
                    hovertext=hover, hoverinfo="text",
                ), row=row_idx, col=1)
        else:
            mean_by_frame = (
                track_df.groupby("frame_id")[col_name]
                .mean().reset_index()
            )
            hover = [
                _z_hover(fid, None, col_name, v, stats)
                for fid, v in zip(mean_by_frame["frame_id"], mean_by_frame[col_name])
            ]
            fig.add_trace(go.Scatter(
                x=mean_by_frame["frame_id"],
                y=mean_by_frame[col_name],
                mode="lines", name=f"{col_name} (mean)",
                line=dict(color="steelblue"),
                hovertext=hover, hoverinfo="text",
            ), row=row_idx, col=1)

        # Boxplot col 2 — always full sequence data
        box_vals = track_df[col_name].dropna().tolist()
        fig.add_trace(go.Box(
            y=box_vals,
            boxpoints="outliers",
            showlegend=False,
            name="",
            marker=dict(color="#1f77b4", size=3),
            line=dict(color="#1f77b4"),
            fillcolor="rgba(70,130,180,0.2)",
            hoverinfo="skip",
        ), row=row_idx, col=2)

        # Spike line on col1 y-axis extends "across" into the boxplot
        fig.update_yaxes(
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikedash="dot",
            spikecolor="gray",
            row=row_idx, col=1,
        )
        # Hide redundant y tick labels on col2 (same axis)
        fig.update_yaxes(showticklabels=False, row=row_idx, col=2)
        # Hide x tick labels on boxplot column
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                         row=row_idx, col=2)

    fig.update_xaxes(title_text="Frame ID", row=4, col=1)
    fig.update_layout(
        height=N_metrics * 220 + 80,
        title="Tracker Timeline",
        hovermode="closest",
        spikedistance=1000,
        margin=dict(t=60, b=30),
    )
    return fig, all_stats


# ---------------------------------------------------------------------------
# Detector Cadence (unchanged)
# ---------------------------------------------------------------------------

def plot_detector_cadence(
    track_df: pd.DataFrame,
    det_df: pd.DataFrame,
) -> go.Figure:
    if track_df.empty:
        return go.Figure()
    all_ids = sorted(track_df["object_id"].unique())
    det_ran = det_df[det_df["detector_ran"]]
    fig     = go.Figure()
    for oid in all_ids:
        color = _obj_color(oid, all_ids)
        sub   = track_df[track_df["object_id"] == oid].sort_values("frame_id")
        fig.add_trace(go.Scatter(
            x=sub["frame_id"], y=sub["frames_since_redetect"],
            mode="lines", name=oid, line=dict(color=color, shape="hv"),
        ))
    added: set[str] = set()
    for _, row in det_ran.iterrows():
        reason = row.get("trigger_reason", "none")
        color  = TRIGGER_COLORS.get(reason, "gray")
        sym    = TRIGGER_SYMBOLS.get(reason, "circle")
        fig.add_trace(go.Scatter(
            x=[row["frame_id"]], y=[-1],
            mode="markers",
            marker=dict(color=color, size=8, symbol=sym),
            name=f"DINO:{reason}", legendgroup=f"dino_{reason}",
            showlegend=(reason not in added),
            hovertemplate=f"trigger: {reason}<br>frame={row['frame_id']}<extra></extra>",
        ))
        added.add(reason)
    fig.update_layout(
        title="Detector Cadence",
        xaxis_title="Frame ID", yaxis_title="frames_since_redetect",
        height=300, margin=dict(t=40, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# Detection Confidence Timeline — 1×2 subplot with boxplot (Change 7)
# ---------------------------------------------------------------------------

def plot_detection_confidence(
    det_df: pd.DataFrame,
    gt_start,
    gt_end,
    selected_objects: list[str] | None = None,
    selected_triggers: list[str] | None = None,
) -> tuple[go.Figure, dict]:
    """
    Layout: 1 row × 2 cols (line | boxplot), shared_yaxes, spike across.
    selected_objects=[]  → annotation asking user to pick.
    selected_triggers    → non-selected points rendered at 20% opacity.
    Returns (fig, stats).
    """
    ran = det_df[det_df["detector_ran"]]
    records = []
    for _, row in ran.iterrows():
        for det in (row["detections"] or []):
            if det.get("is_selected", False) and det.get("confidence") is not None:
                records.append({
                    "frame_id":     row["frame_id"],
                    "timestamp":    row.get("timestamp"),
                    "object_id":    det["object_id"],
                    "label":        det.get("label", ""),
                    "confidence":   float(det["confidence"]),
                    "failure_mode": row.get("failure_mode"),
                    "trigger_reason": row.get("trigger_reason", "none"),
                })
    if not records:
        return go.Figure(), {}

    df      = pd.DataFrame(records)
    all_ids = sorted(df["object_id"].unique())
    stats   = reference_stats(df["confidence"])

    # Empty selection guard
    if selected_objects is not None and len(selected_objects) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least one object in <b>Filters</b> above to see the timeline.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="gray"),
        )
        fig.update_layout(height=200)
        return fig, stats

    disp_ids = selected_objects if selected_objects else all_ids

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.88, 0.12],
        shared_yaxes=True,
        horizontal_spacing=0.02,
        subplot_titles=["Detection Confidence", ""],
    )
    _gt_band(fig, gt_start, gt_end, row=1, col=1)

    x_min = int(df["frame_id"].min())
    x_max = int(df["frame_id"].max())
    n_pts  = len(df)
    sparse = n_pts < 5

    for oid in disp_ids:
        if oid not in df["object_id"].values:
            continue
        color = _obj_color(oid, all_ids)
        sub   = df[df["object_id"] == oid].sort_values("frame_id")

        def _opacity(trigger_reason):
            if not selected_triggers:
                return 1.0
            return 1.0 if trigger_reason in selected_triggers else 0.2

        # Split into full-opacity and faded groups if triggers filter is on
        if selected_triggers:
            sub_sel   = sub[sub["trigger_reason"].isin(selected_triggers)]
            sub_faded = sub[~sub["trigger_reason"].isin(selected_triggers)]
            groups = [(sub_sel, 1.0), (sub_faded, 0.2)]
        else:
            groups = [(sub, 1.0)]

        for grp, opacity in groups:
            if grp.empty:
                continue
            marker_colors = [
                "red" if (fm is not None and fm == fm) else color
                for fm in grp["failure_mode"]
            ]
            hover = [
                _z_hover(r.frame_id, r.timestamp, "confidence", r.confidence, stats,
                         extra=f"<br>failure_mode={r.failure_mode}" if r.failure_mode else "")
                for _, r in grp.iterrows()
            ]
            mode = "markers" if sparse else "lines+markers"
            fig.add_trace(go.Scatter(
                x=grp["frame_id"], y=grp["confidence"],
                mode=mode,
                name=oid if opacity == 1.0 else f"{oid} (filtered)",
                showlegend=(opacity == 1.0),
                line=dict(color=color),
                marker=dict(color=marker_colors, size=8),
                opacity=opacity,
                hovertext=hover, hoverinfo="text",
            ), row=1, col=1)

    if sparse:
        fig.add_annotation(
            text=f"Detection confidence is sparse (DINO ran {n_pts} times). "
                 "Boxplot summarises the few values available.",
            xref="paper", yref="paper", x=0.5, y=1.08,
            showarrow=False, font=dict(size=10, color="gray"),
        )

    # Boxplot col 2 — always full dataset
    fig.add_trace(go.Box(
        y=df["confidence"].tolist(),
        boxpoints="outliers",
        showlegend=False,
        name="",
        marker=dict(color="#1f77b4", size=3),
        line=dict(color="#1f77b4"),
        fillcolor="rgba(70,130,180,0.2)",
        hoverinfo="skip",
    ), row=1, col=2)

    fig.update_yaxes(
        showspikes=True, spikemode="across",
        spikethickness=1, spikedash="dot", spikecolor="gray",
        row=1, col=1,
    )
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_xaxes(title_text="Frame ID", row=1, col=1)
    fig.update_layout(
        yaxis_title="Confidence",
        height=320,
        hovermode="closest",
        spikedistance=1000,
        margin=dict(t=50, b=30),
    )
    return fig, stats


# ---------------------------------------------------------------------------
# Flag strip — thin marker strip for Frame Viewer (Change 4)
# ---------------------------------------------------------------------------

def plot_flag_strip(
    track_frame_df: pd.DataFrame,
    max_frame_id: int,
) -> go.Figure:
    """
    50px-tall Plotly strip; one colored marker per flag firing.
    Rendered above the Frame Viewer slider — not interactive (displayModeBar=False).
    """
    fig = go.Figure()

    if not track_frame_df.empty:
        has_any = False
        for flag_col, color in FLAG_COLORS.items():
            if flag_col not in track_frame_df.columns:
                continue
            fired = track_frame_df[track_frame_df[flag_col] == True]
            if fired.empty:
                continue
            has_any = True
            fig.add_trace(go.Scatter(
                x=fired["frame_id"].tolist(),
                y=[0] * len(fired),
                mode="markers",
                marker=dict(color=color, size=8, symbol="circle"),
                name=flag_col,
                hovertemplate=f"Frame %{{x}} · {flag_col}<extra></extra>",
            ))
        if not has_any:
            fig.add_annotation(
                text="No flags fired in this sequence.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=10, color="gray"),
            )
    else:
        fig.add_annotation(
            text="No flags fired in this sequence.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=10, color="gray"),
        )

    fig.update_xaxes(range=[0, max_frame_id], showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=50,
        margin=dict(t=5, b=5, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0, font=dict(size=10)),
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Depth Timeline — 2-col layout with boxplot reference (Step 4)
# ---------------------------------------------------------------------------

_DEPTH_FLAG_DISPLAY = {
    "depth_jump_flag":      ("#d62728", True,  "triangle-up", 0.2),
    "depth_coherence_flag": ("#ff7f0e", False, "square",      0.4),
    "depth_validity_flag":  ("#9467bd", False, "diamond",     0.6),
    "raw_depth_trigger":    ("#7f7f7f", True,  "circle",      0.75),
    "any_depth_trigger":    ("#e377c2", True,  "circle",      0.9),
}


def plot_depth_timeline(
    depth_df: pd.DataFrame,
    gt_start,
    gt_end,
    selected_objects: list[str] | None = None,
    selected_flags: list[str] | None = None,
) -> tuple[go.Figure, dict]:
    """
    Layout: (n+1) rows × 2 cols.
      Row 1 col 1 — flag events strip (filtered by selected_flags; all shown when None)
      Rows 2+ col 1 — depth_median_m lines per object + IQR band + valid_ratio overlay
      Rows 2+ col 2 — per-object boxplot (full-sequence reference, shared_yaxes)

    selected_objects=None → show all objects.
    selected_objects=[]   → annotation asking user to pick.
    selected_flags=None   → show all flag types in events strip.
    selected_flags=[]     → omit flag markers.
    Returns (fig, stats_dict).
    """
    if depth_df.empty:
        return go.Figure(), {}

    if selected_objects is not None and len(selected_objects) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least one object in <b>Filters</b> above to see the timeline.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="gray"),
        )
        fig.update_layout(height=200)
        return fig, {}

    all_ids  = sorted(depth_df["object_id"].unique())
    disp_ids = sorted(selected_objects) if selected_objects else all_ids
    n        = len(disp_ids)
    if n == 0:
        return go.Figure(), {}

    subplot_titles = ["Depth Flags", ""] + [t for oid in disp_ids for t in (oid, "")]
    fig = make_subplots(
        rows=n + 1, cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        column_widths=[0.88, 0.12],
        row_heights=[0.08] + [0.92 / n] * n,
        vertical_spacing=0.05,
        horizontal_spacing=0.02,
        subplot_titles=subplot_titles,
    )

    # ── Row 1: Flag events strip ─────────────────────────────────────────────
    active_flags = selected_flags if selected_flags is not None else list(_DEPTH_FLAG_DISPLAY.keys())
    seen_flag_names: set[str] = set()
    for flag_col, (fc, show_when, fs, ypos) in _DEPTH_FLAG_DISPLAY.items():
        if flag_col not in active_flags or flag_col not in depth_df.columns:
            continue
        if show_when:
            agg    = depth_df.groupby("frame_id")[flag_col].any()
            frames = agg[agg].index.tolist()
            label  = flag_col
        else:
            agg    = depth_df.groupby("frame_id")[flag_col].all()
            frames = agg[~agg].index.tolist()
            label  = f"{flag_col}_violation"
        if not frames:
            continue
        fig.add_trace(go.Scatter(
            x=frames, y=[ypos] * len(frames),
            mode="markers",
            marker=dict(color=fc, size=8, symbol=fs),
            name=label, legendgroup=f"dep_{flag_col}",
            showlegend=(flag_col not in seen_flag_names),
            hovertemplate=f"{label}<br>frame=%{{x}}<extra></extra>",
        ), row=1, col=1)
        seen_flag_names.add(flag_col)

    fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=2)
    fig.update_xaxes(visible=False, showgrid=False, row=1, col=2)

    # ── Rows 2+: Per-object depth lines + boxplot reference ──────────────────
    all_stats: dict[str, dict] = {}
    for i, oid in enumerate(disp_ids):
        row_idx = i + 2
        sub     = depth_df[depth_df["object_id"] == oid].sort_values("frame_id")
        color   = _obj_color(oid, all_ids)
        stats   = reference_stats(sub["depth_median_m"].dropna())
        all_stats[oid] = stats

        _gt_band(fig, gt_start, gt_end, row=row_idx, col=1)

        # IQR band
        med = sub["depth_median_m"]
        iqr = sub["depth_iqr_m"].fillna(0)
        fig.add_trace(go.Scatter(
            x=pd.concat([sub["frame_id"], sub["frame_id"][::-1]]).tolist(),
            y=pd.concat([med + iqr / 2, (med - iqr / 2)[::-1]]).tolist(),
            fill="toself", fillcolor="rgba(100,100,255,0.12)",
            line=dict(width=0), showlegend=False,
            name=f"{oid} IQR", hoverinfo="skip",
        ), row=row_idx, col=1)

        # Depth median line
        hover = [
            _z_hover(r.frame_id, r.get("timestamp"), "depth_median_m", r.depth_median_m, stats)
            for _, r in sub.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=sub["frame_id"], y=med,
            mode="lines", name=oid, legendgroup=oid,
            showlegend=True,
            line=dict(color=color),
            hovertext=hover, hoverinfo="text",
        ), row=row_idx, col=1)

        # Valid-ratio overlay (dotted)
        if "valid_depth_pixel_ratio" in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub["frame_id"], y=sub["valid_depth_pixel_ratio"],
                mode="lines", name=f"{oid} valid_ratio",
                legendgroup=f"{oid}_valid", showlegend=False,
                line=dict(color=color, dash="dot", width=1), opacity=0.45,
                hovertemplate="frame=%{x}<br>valid_ratio=%{y:.2f}<extra></extra>",
            ), row=row_idx, col=1)

        # Boxplot col 2 — per-object full-sequence reference
        box_vals = sub["depth_median_m"].dropna().tolist()
        fig.add_trace(go.Box(
            y=box_vals,
            boxpoints="outliers",
            showlegend=False, name="",
            marker=dict(color=color, size=3),
            line=dict(color=color),
            fillcolor="rgba(100,100,200,0.2)",
            hoverinfo="skip",
        ), row=row_idx, col=2)

        # Spike line on col1 extends across into col2 boxplot
        fig.update_yaxes(
            showspikes=True, spikemode="across",
            spikethickness=1, spikedash="dot", spikecolor="gray",
            row=row_idx, col=1,
        )
        fig.update_yaxes(showticklabels=False, row=row_idx, col=2)
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                         row=row_idx, col=2)

    fig.update_xaxes(title_text="Frame ID", row=n + 1, col=1)
    fig.update_layout(
        height=250 * n + 120,
        title="Depth Timeline",
        hovermode="closest",
        spikedistance=1000,
        margin=dict(t=60, b=30),
    )
    return fig, all_stats


def plot_depth_flag_strip(
    depth_df: pd.DataFrame,
    max_frame_id: int,
) -> go.Figure:
    """
    50px strip showing all depth flag firings across the full frame range.
    Coherence/validity flags are shown as violations (when False).
    """
    fig = go.Figure()
    has_any = False

    if not depth_df.empty:
        for flag_col, (color, show_when, _sym, _ypos) in _DEPTH_FLAG_DISPLAY.items():
            if flag_col not in depth_df.columns:
                continue
            if show_when:
                agg    = depth_df.groupby("frame_id")[flag_col].any()
                frames = agg[agg].index.tolist()
                label  = flag_col
            else:
                agg    = depth_df.groupby("frame_id")[flag_col].all()
                frames = agg[~agg].index.tolist()
                label  = f"{flag_col}_violation"
            if frames:
                has_any = True
                fig.add_trace(go.Scatter(
                    x=frames, y=[0] * len(frames),
                    mode="markers",
                    marker=dict(color=color, size=8, symbol="circle"),
                    name=label,
                    hovertemplate=f"Frame %{{x}} · {label}<extra></extra>",
                ))

    if not has_any:
        fig.add_annotation(
            text="No depth flags fired in this sequence.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=10, color="gray"),
        )

    fig.update_xaxes(range=[0, max_frame_id], showticklabels=False,
                     showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=50,
        margin=dict(t=5, b=5, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0, font=dict(size=10)),
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Scene Graph plots (unchanged)
# ---------------------------------------------------------------------------

def plot_sg_displacement(
    disp_df: pd.DataFrame,
    all_ids: list[str],
) -> tuple[go.Figure, dict]:
    fig = go.Figure()
    if disp_df.empty:
        return fig, {}
    stats = reference_stats(disp_df["displacement_3d_m"].dropna())
    x_min = int(disp_df["frame_id"].min())
    x_max = int(disp_df["frame_id"].max())
    for oid in sorted(disp_df["object_id"].unique()):
        color = _obj_color(oid, all_ids)
        sub   = disp_df[disp_df["object_id"] == oid].sort_values("frame_id")
        hover = [
            _z_hover(r.frame_id, r.get("timestamp"), "displacement_3d_m", r.displacement_3d_m, stats)
            for _, r in sub.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=sub["frame_id"], y=sub["displacement_3d_m"],
            mode="lines+markers", name=oid, line=dict(color=color),
            hovertext=hover, hoverinfo="text",
        ))
    fig.update_layout(
        title="3D Displacement Between Consecutive Frames",
        xaxis_title="Frame ID", yaxis_title="Displacement (m)",
        height=280, margin=dict(t=40, b=30, r=130),
    )
    return fig, stats


def plot_sg_jaccard(
    jac_df: pd.DataFrame,
    all_ids: list[str],
) -> tuple[go.Figure, dict]:
    fig = go.Figure()
    if jac_df.empty:
        return fig, {}
    stats = reference_stats(jac_df["jaccard"].dropna())
    for oid in sorted(jac_df["object_id"].unique()):
        color = _obj_color(oid, all_ids)
        sub   = jac_df[jac_df["object_id"] == oid].sort_values("frame_id")
        hover = [
            _z_hover(r.frame_id, None, "jaccard", r.jaccard, stats)
            for _, r in sub.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=sub["frame_id"], y=sub["jaccard"],
            mode="lines+markers", name=oid, line=dict(color=color),
            hovertext=hover, hoverinfo="text",
        ))
    mf = jac_df.groupby("frame_id")["jaccard"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=mf["frame_id"], y=mf["jaccard"],
        mode="lines", name="mean (all obj)",
        line=dict(color="black", dash="dash"),
    ))
    fig.update_layout(
        title="Neighbor Jaccard (near-edge consistency t→t+1)",
        xaxis_title="Frame ID", yaxis_title="Jaccard",
        yaxis_range=[0, 1.05],
        height=280, margin=dict(t=40, b=30, r=130),
    )
    return fig, stats


def plot_edge_flicker_heatmap(mat: pd.DataFrame) -> go.Figure:
    if mat.empty:
        return go.Figure()
    fig = go.Figure(data=go.Heatmap(
        z=mat.values,
        x=mat.columns.tolist(),
        y=mat.index.tolist(),
        colorscale="Reds",
        text=mat.values,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title="Edge Flicker Heatmap",
        height=280, margin=dict(t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Relation mix bar (Scene Graph tab)
# ---------------------------------------------------------------------------

_RELATION_COLORS = {
    "near":       "#1f77b4",
    "left_of":    "#ff7f0e",
    "above":      "#2ca02c",
    "on_top_of":  "#d62728",
    "inside":     "#9467bd",
}


def plot_relation_mix_bar(edges_df: pd.DataFrame) -> go.Figure:
    """Horizontal stacked bar showing edge count per relation type."""
    if edges_df.empty or "relation" not in edges_df.columns:
        return go.Figure()
    rel_counts = (
        edges_df["relation"].dropna().value_counts()
        .reset_index()
        .rename(columns={"index": "relation", "relation": "count"})
    )
    if rel_counts.empty:
        return go.Figure()
    fig = go.Figure()
    for _, r in rel_counts.iterrows():
        rel = str(r.iloc[0])
        cnt = int(r.iloc[1])
        fig.add_trace(go.Bar(
            name=rel,
            x=[cnt], y=["edge mix"],
            orientation="h",
            marker_color=_RELATION_COLORS.get(rel, "#aaaaaa"),
            text=[f"{rel} ({cnt})"],
            textposition="inside",
            hovertemplate=f"{rel}: {cnt}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        height=80,
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0, font=dict(size=10)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False)
    return fig
