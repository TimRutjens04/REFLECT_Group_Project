"""Scene-graph temporal consistency metrics.

All thresholds are documented inline so they can be tuned without hunting.
"""
import numpy as np
import pandas as pd

# Thresholds for auto-interpretation
FLICKER_RATE_HIGH = 0.15       # fraction of frames with disappear/reappear
DISPLACEMENT_HIGH_M = 0.03     # mean 3D displacement considered "high"
DISPLACEMENT_P95_HIGH_M = 0.08
JACCARD_LOW = 0.5              # mean Jaccard below this is "unstable neighbors"
EDGE_FLICKER_HIGH = 3          # edge on/off transitions considered "high"


def object_flicker_rate(nodes_df: pd.DataFrame, all_frames: list[int]) -> pd.DataFrame:
    """
    Per object: count frames it disappears then reappears / total frames.
    A disappearance is a frame where the object is absent after being present.
    """
    if nodes_df.empty:
        return pd.DataFrame(columns=["object_id", "total_frames", "absent_frames", "flicker_events", "flicker_rate"])

    all_ids = nodes_df["object_id"].unique()
    total = len(all_frames)
    records = []
    for oid in sorted(all_ids):
        present = set(nodes_df[nodes_df["object_id"] == oid]["frame_id"].tolist())
        absent = 0
        flicker = 0
        was_present = False
        was_absent = False
        for fid in sorted(all_frames):
            if fid in present:
                if was_absent:
                    flicker += 1
                was_present = True
                was_absent = False
            else:
                if was_present:
                    absent += 1
                was_absent = True
                was_present = False
        records.append({
            "object_id": oid,
            "total_frames": total,
            "absent_frames": total - len(present),
            "flicker_events": flicker,
            "flicker_rate": flicker / total if total > 0 else 0.0,
        })
    return pd.DataFrame(records)


def mean_3d_displacement(nodes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per object per consecutive frame pair: euclidean distance between position_3d.
    Returns (per_frame_df, summary_df).
    """
    if nodes_df.empty or not {"pos_x", "pos_y", "pos_z"}.issubset(nodes_df.columns):
        empty = pd.DataFrame(columns=["object_id", "frame_id", "displacement_3d_m"])
        summary = pd.DataFrame(columns=["object_id", "mean_disp_m", "p95_disp_m"])
        return empty, summary

    records = []
    all_ids = nodes_df["object_id"].unique()
    for oid in sorted(all_ids):
        sub = nodes_df[nodes_df["object_id"] == oid].sort_values("frame_id")
        xs = sub["pos_x"].values
        ys = sub["pos_y"].values
        zs = sub["pos_z"].values
        fids = sub["frame_id"].values
        for i in range(1, len(sub)):
            # skip if any position is null
            if any(v is None or (isinstance(v, float) and np.isnan(v))
                   for v in [xs[i], ys[i], xs[i-1], ys[i-1], zs[i], zs[i-1]]):
                continue
            d = float(np.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2 + (zs[i]-zs[i-1])**2))
            records.append({"object_id": oid, "frame_id": int(fids[i]), "displacement_3d_m": d})

    if not records:
        empty = pd.DataFrame(columns=["object_id", "frame_id", "displacement_3d_m"])
        summary = pd.DataFrame(columns=["object_id", "mean_disp_m", "p95_disp_m"])
        return empty, summary

    disp_df = pd.DataFrame(records)
    summary_rows = []
    for oid, grp in disp_df.groupby("object_id"):
        vals = grp["displacement_3d_m"].dropna().values
        summary_rows.append({
            "object_id": oid,
            "mean_disp_m": float(np.mean(vals)) if len(vals) else np.nan,
            "p95_disp_m": float(np.percentile(vals, 95)) if len(vals) else np.nan,
        })
    return disp_df, pd.DataFrame(summary_rows)


def neighbor_jaccard(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    relation_types: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per object per consecutive frame transition: Jaccard of neighbor sets.
    Pass relation_types=['near','inside'] to union neighbor sets across types;
    default ['near'] preserves Part B semantics.
    Returns (per_frame_df, summary_df).
    """
    if relation_types is None:
        relation_types = ["near"]
    if not edges_df.empty and relation_types:
        edges_df = edges_df[edges_df["relation"].isin(relation_types)].copy()
    if edges_df.empty or nodes_df.empty:
        empty = pd.DataFrame(columns=["object_id", "frame_id", "jaccard"])
        summary = pd.DataFrame(columns=["object_id", "mean_jaccard"])
        return empty, summary

    all_frames = sorted(nodes_df["frame_id"].unique())
    all_ids = nodes_df["object_id"].unique()

    # Build neighbor sets per (frame, object)
    def get_neighbors(fid, oid):
        if edges_df.empty:
            return set()
        mask = (edges_df["frame_id"] == fid) & (
            (edges_df["from_id"] == oid) | (edges_df["to_id"] == oid)
        )
        neighbors = set()
        for _, row in edges_df[mask].iterrows():
            neighbors.add(row["from_id"] if row["to_id"] == oid else row["to_id"])
        return neighbors

    records = []
    for oid in sorted(all_ids):
        present_frames = sorted(nodes_df[nodes_df["object_id"] == oid]["frame_id"].unique())
        for i in range(1, len(present_frames)):
            f0, f1 = present_frames[i-1], present_frames[i]
            # only consecutive frames
            if f1 != f0 + 1:
                continue
            n0 = get_neighbors(f0, oid)
            n1 = get_neighbors(f1, oid)
            union = n0 | n1
            inter = n0 & n1
            j = len(inter) / len(union) if union else 1.0
            records.append({"object_id": oid, "frame_id": f1, "jaccard": j})

    if not records:
        empty = pd.DataFrame(columns=["object_id", "frame_id", "jaccard"])
        summary = pd.DataFrame(columns=["object_id", "mean_jaccard"])
        return empty, summary

    jac_df = pd.DataFrame(records)
    summary_rows = [
        {"object_id": oid, "mean_jaccard": float(grp["jaccard"].mean())}
        for oid, grp in jac_df.groupby("object_id")
    ]
    return jac_df, pd.DataFrame(summary_rows)


def edge_flicker(
    edges_df: pd.DataFrame,
    all_frames: list[int],
    relation_types: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each undirected pair (a,b), count on↔off transitions across selected relation types.
    Default relation_types=['near'] preserves Part B semantics.
    Returns (flicker_df with pair/count, pivot heatmap matrix).
    """
    if relation_types is None:
        relation_types = ["near"]
    if not edges_df.empty and relation_types:
        edges_df = edges_df[edges_df["relation"].isin(relation_types)].copy()
    if edges_df.empty:
        return pd.DataFrame(columns=["pair", "flicker_count"]), pd.DataFrame()

    # Canonical pair (always sort)
    pairs = set()
    for _, row in edges_df.iterrows():
        pairs.add(tuple(sorted([row["from_id"], row["to_id"]])))

    records = []
    for a, b in sorted(pairs):
        transitions = 0
        was_on = False
        for fid in sorted(all_frames):
            mask = (edges_df["frame_id"] == fid) & (
                ((edges_df["from_id"] == a) & (edges_df["to_id"] == b)) |
                ((edges_df["from_id"] == b) & (edges_df["to_id"] == a))
            )
            on = mask.any()
            if on != was_on:
                transitions += 1
            was_on = on
        records.append({"pair": f"{a} ↔ {b}", "obj_a": a, "obj_b": b, "flicker_count": transitions})

    flicker_df = pd.DataFrame(records).sort_values("flicker_count", ascending=False).reset_index(drop=True)

    # Build heatmap matrix
    all_objs = sorted(set(flicker_df["obj_a"].tolist() + flicker_df["obj_b"].tolist()))
    mat = pd.DataFrame(0, index=all_objs, columns=all_objs)
    for _, row in flicker_df.iterrows():
        mat.loc[row["obj_a"], row["obj_b"]] = row["flicker_count"]
        mat.loc[row["obj_b"], row["obj_a"]] = row["flicker_count"]

    return flicker_df[["pair", "flicker_count"]], mat


def reference_stats(values: "pd.Series") -> dict:
    """
    Compute summary statistics for a series of values (pooled across all objects/frames).
    Returns dict with keys: min, p5, median, mean, p95, max, std.
    All values are float; NaN when the input is empty.
    Used by every numerical timeline for F4 reference lines and stats tables.
    """
    arr = values.dropna().values.astype(float)
    if len(arr) == 0:
        return {k: float("nan") for k in ["min", "p5", "median", "mean", "p95", "max", "std"]}
    return {
        "min":    float(np.min(arr)),
        "p5":     float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "mean":   float(np.mean(arr)),
        "p95":    float(np.percentile(arr, 95)),
        "max":    float(np.max(arr)),
        "std":    float(np.std(arr)),
    }


def auto_interpret(
    flicker_df: pd.DataFrame,
    disp_summary: pd.DataFrame,
    jac_summary: pd.DataFrame,
    edge_flicker_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    gt_start: float | None,
    gt_end: float | None,
    relation_types: list[str] | None = None,
) -> str:
    """Generate a plain-English summary using simple threshold rules."""
    lines = []

    if not disp_summary.empty:
        for _, row in disp_summary.iterrows():
            oid = row["object_id"]
            mean_d = row.get("mean_disp_m", np.nan)
            p95_d = row.get("p95_disp_m", np.nan)
            if not np.isnan(mean_d) and mean_d > DISPLACEMENT_HIGH_M:
                # Find frame range where displacement is high
                lines.append(
                    f"**{oid}** has high mean 3D displacement ({mean_d:.3f} m, "
                    f"95th pct {p95_d:.3f} m)."
                )

    if not tracking_df.empty and not disp_summary.empty:
        for _, row in disp_summary.iterrows():
            oid = row["object_id"]
            mean_d = row.get("mean_disp_m", np.nan)
            if np.isnan(mean_d) or mean_d <= DISPLACEMENT_HIGH_M:
                continue
            sub = tracking_df[tracking_df["object_id"] == oid]
            drifting = sub[sub["tracker_status"].isin(["drifting", "lost"])]
            if not drifting.empty:
                fids = drifting["frame_id"].tolist()
                lines.append(
                    f"  — This coincides with tracker status `{drifting['tracker_status'].iloc[0]}` "
                    f"at frames {fids[0]}–{fids[-1]}."
                )

    if not flicker_df.empty:
        high = flicker_df[flicker_df["flicker_rate"] > FLICKER_RATE_HIGH]
        for _, row in high.iterrows():
            lines.append(
                f"**{row['object_id']}** flickers (disappears and reappears) "
                f"{row['flicker_events']} time(s) — flicker rate {row['flicker_rate']:.1%}."
            )

    if not jac_summary.empty:
        low = jac_summary[jac_summary["mean_jaccard"] < JACCARD_LOW]
        for _, row in low.iterrows():
            lines.append(
                f"**{row['object_id']}** has unstable neighbor set "
                f"(mean Jaccard {row['mean_jaccard']:.2f} < {JACCARD_LOW})."
            )

    if not edge_flicker_df.empty:
        high_edges = edge_flicker_df[edge_flicker_df["flicker_count"] >= EDGE_FLICKER_HIGH]
        for _, row in high_edges.iterrows():
            lines.append(
                f"Edge **{row['pair']}** flickers {row['flicker_count']} time(s)."
            )

    if not lines:
        lines.append("No significant anomalies detected in the scene graph for this sequence.")

    if gt_start is not None:
        rel_str = ", ".join(relation_types) if relation_types else "near"
        lines.append(
            f"\n_GT failure window: {gt_start:.1f}s – {gt_end:.1f}s. "
            f"Neighbor stability computed across relation types: {rel_str}. "
            f"Thresholds: displacement > {DISPLACEMENT_HIGH_M} m, "
            f"flicker rate > {FLICKER_RATE_HIGH:.0%}, Jaccard < {JACCARD_LOW}, "
            f"edge flicker ≥ {EDGE_FLICKER_HIGH}._"
        )

    return "\n\n".join(lines)
