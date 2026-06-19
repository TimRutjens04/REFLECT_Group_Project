#!/usr/bin/env python3
"""
build.py — Read the REFLECT pipeline JSONL logs for *every* discoverable sequence
and emit a single self-contained index.html with all the data embedded as
`const PAYLOAD = {...}`. A dropdown in the page switches between sequences at
runtime with no server reload.

The generated dashboard plays the robot video while a playhead, live metric cards,
event badges, and a scene-graph relation strip stay in sync with the video time.

Run directly:   python "dashboard 2/build.py"
Or via:         python "dashboard 2/run.py"   (build + serve + open browser)

Sequences are auto-discovered under pipeline/real_world/jsonl/. DEFAULT_SEQUENCE
below only decides which one is shown when the page first loads.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import cv2  # only used to probe each video's natural pixel dimensions
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import requests  # used to call the local Ollama server for LLM commentary
except Exception:  # pragma: no cover - optional dependency
    requests = None

# --------------------------------------------------------------------------- #
# Configuration (build-time constants)
# --------------------------------------------------------------------------- #
DEFAULT_SEQUENCE = "putAppleBowl1"   # initial selection in the dropdown
PORT = 8765

# The two persistent ("main") object identities for the apple/bowl sequence. The
# build prefers these when present and otherwise falls back to whichever IDs
# appear in the most frames (see pick_main_objects).
EXPECTED_MAIN = ["red apple_1", "dark blue bowl_2"]

# Per-object-class colors, keyed by object_id. The front-end resolves colors by
# *label* at runtime (so all apples are red, all bowls blue, across sequences),
# using the label->color map derived from this constant plus a fallback palette.
OBJECT_COLORS = {
    "red apple_1": "#d62728",
    "dark blue bowl_2": "#1f3a93",
}

# Deterministic palette for object classes with no entry in OBJECT_COLORS.
FALLBACK_PALETTE = [
    "#2ca02c", "#9467bd", "#17becf", "#bcbd22", "#e377c2", "#8c564b", "#7f7f7f",
]

# Relation -> color (matches the strip legend in the HTML).
RELATION_COLORS = {
    "near": "#9467bd",
    "left_of": "#7f7f7f",
    "above": "#bcbd22",
    "on_top_of": "#e377c2",
    "inside": "#ff7f0e",
}

DASH_DIR = Path(__file__).resolve().parent          # ".../dashboard 2"
HTML_OUT = DASH_DIR / "index.html"
OVERVIEW_OUT = DASH_DIR / "overview.html"
VIDEO_OUT_DIR = DASH_DIR / "video"

# ---- LLM (Ollama) configuration for the overview-page commentary ----------- #
OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.1:8b"
OLLAMA_FALLBACK = "llama3:latest"   # used if the preferred model isn't installed
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT_S   = 60
PROMPT_VERSION  = "overview_v1.0"
LLM_CACHE_DIR   = DASH_DIR / ".llm_cache"


# --------------------------------------------------------------------------- #
# Path discovery
# --------------------------------------------------------------------------- #
def find_repo_root(start: Path) -> Path:
    """Walk up from `start` until we find a dir containing both pipeline/ and
    example_data/ (the REFLECT project root)."""
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "pipeline").is_dir() and (cand / "example_data").is_dir():
            return cand
    raise SystemExit(
        "Could not locate the REFLECT project root (a directory containing "
        "both 'pipeline/' and 'example_data/') above "
        f"{start}. Run this from inside the REFLECT_Group_Project-1 tree."
    )


def discover_sequences(jsonl_root: Path) -> list[str]:
    """Return sorted sequence ids under `jsonl_root` that have at minimum the two
    required logs ({seq}__detection.jsonl and {seq}__tracking.jsonl). Sequences
    missing a required log are skipped with a one-line warning."""
    found: list[str] = []
    if not jsonl_root.is_dir():
        return found
    for sub in sorted(jsonl_root.iterdir()):
        if not sub.is_dir():
            continue
        seq = sub.name
        required = {
            "detection": sub / f"{seq}__detection.jsonl",
            "tracking": sub / f"{seq}__tracking.jsonl",
        }
        missing = [name for name, p in required.items() if not p.exists()]
        if missing:
            print(f"  ! skipping '{seq}': missing required {', '.join(missing)}")
            continue
        found.append(seq)
    return found


# --------------------------------------------------------------------------- #
# JSONL loading (tolerant: returns [] for an absent optional log)
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Helpers for the heterogeneous / evolving schema
# --------------------------------------------------------------------------- #
def edge_endpoints(edge: dict) -> tuple[str | None, str | None]:
    """Scene-graph edges use either the old (from/to) or new
    (from_object_id/to_object_id) spelling."""
    a = edge.get("from", edge.get("from_object_id"))
    b = edge.get("to", edge.get("to_object_id"))
    return a, b


def obj_status(obj: dict) -> str | None:
    """Node status uses 'state' (old) or 'status' (new)."""
    return obj.get("tracker_status", obj.get("status", obj.get("state")))


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def compute_fps(tracking: list[dict]) -> float:
    ts = [r["timestamp"] for r in tracking if r.get("timestamp") is not None]
    ts.sort()
    deltas = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    if not deltas:
        return 30.0
    mean_dt = sum(deltas) / len(deltas)
    return round(1.0 / mean_dt, 4) if mean_dt > 0 else 30.0


def pick_main_objects(tracking: list[dict]) -> list[str]:
    """Return the two longest-lived object_ids. Prefer EXPECTED_MAIN when both
    are present; otherwise fall back to the two most-frequent IDs."""
    counts: dict[str, int] = {}
    for r in tracking:
        for o in r.get("tracked_objects", []):
            oid = o.get("object_id")
            if oid:
                counts[oid] = counts.get(oid, 0) + 1
    if all(o in counts for o in EXPECTED_MAIN):
        return list(EXPECTED_MAIN)
    ranked = sorted(counts, key=lambda k: counts[k], reverse=True)
    return ranked[:2]


def mmss_to_seconds(s: str) -> float:
    parts = [int(p) for p in str(s).split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(s)


def probe_video_dims(path: Path, default=(720, 720)) -> dict:
    """Return the video's natural pixel dimensions via OpenCV. Falls back to
    `default` (and warns) when cv2 is unavailable or the file can't be read —
    never crashes the build. bbox coordinates are in this same pixel space."""
    dw, dh = default
    if cv2 is None:
        print(f"    ! OpenCV not installed; using default video_dims {dw}x{dh} "
              f"for {path.name}")
        return {"width": dw, "height": dh}
    if not path.exists():
        print(f"    ! video missing, can't probe dims: {path}; "
              f"using default {dw}x{dh}")
        return {"width": dw, "height": dh}
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        print(f"    ! could not read dims from {path}; using default {dw}x{dh}")
        return {"width": dw, "height": dh}
    return {"width": w, "height": h}


def _percentile(sorted_vals: list, p: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def unique_ids_by_label(rows: list[dict], extract) -> dict:
    """Count unique object_ids per label (label = oid with the _N suffix stripped)."""
    by_label: dict[str, set] = {}
    for r in rows:
        for oid in extract(r):
            if not oid:
                continue
            label = oid.rsplit("_", 1)[0]
            by_label.setdefault(label, set()).add(oid)
    return {lbl: len(s) for lbl, s in by_label.items()}


def _ids_set(rows: list[dict], extract) -> set:
    s: set = set()
    for r in rows:
        for oid in extract(r):
            if oid:
                s.add(oid)
    return s


def build_data(repo: Path, seq_id: str) -> dict:
    """Build the per-sequence payload (same shape the old single-sequence DATA
    had). Optional modules (depth, scene_graph, validation) yield empty
    series/strip/transitions when their JSONL is absent."""
    jdir = repo / "pipeline" / "real_world" / "jsonl" / seq_id
    detection = load_jsonl(jdir / f"{seq_id}__detection.jsonl")
    tracking = load_jsonl(jdir / f"{seq_id}__tracking.jsonl")
    depth = load_jsonl(jdir / f"{seq_id}__depth.jsonl")
    scene_graph = load_jsonl(jdir / f"{seq_id}__scene_graph.jsonl")
    validation = load_jsonl(jdir / f"{seq_id}__validation.jsonl")

    for r in (tracking, depth, scene_graph, validation, detection):
        r.sort(key=lambda x: x.get("frame_id", 0))

    fps = compute_fps(tracking)
    all_fids = [r["frame_id"] for r in tracking] or [0]
    n_frames = max(all_fids) + 1
    total_seconds = round(max((r.get("timestamp", 0.0) for r in tracking),
                              default=0.0), 3)

    main_objs = pick_main_objects(tracking)
    main_set = set(main_objs)

    # Labels only — colors are resolved by label on the front-end so the same
    # physical class shows the same color across sequences.
    per_object = {oid: {"label": oid.rsplit("_", 1)[0]} for oid in main_objs}

    # ---- Ground truth (None when this sequence has no GT entry) ------------
    gt = None
    tasks_path = repo / "example_data" / "tasks_real_world.json"
    if tasks_path.exists():
        tasks = json.load(tasks_path.open())
        entry = None
        for v in (tasks.values() if isinstance(tasks, dict) else tasks):
            if isinstance(v, dict) and v.get("general_folder_name") == seq_id:
                entry = v
                break
        if entry:
            step = entry.get("gt_failure_step") or []
            gt = {
                "task_name": entry.get("name", ""),
                "failure_reason": entry.get("gt_failure_reason", ""),
                "success_condition": entry.get("success_condition", ""),
                "failure_window_s": None,
                "failure_window_frames": None,
            }
            if len(step) == 2:
                s0, s1 = mmss_to_seconds(step[0]), mmss_to_seconds(step[1])
                gt["failure_window_s"] = [s0, s1]
                gt["failure_window_frames"] = [int(round(s0 * fps)),
                                               int(round(s1 * fps))]

    # ---- Time series (tracker confidence + depth median) ------------------
    tracker_conf = {oid: [] for oid in main_objs}
    for r in tracking:
        fid = r["frame_id"]
        for o in r.get("tracked_objects", []):
            oid = o.get("object_id")
            if oid in main_set and o.get("tracker_confidence") is not None:
                tracker_conf[oid].append({"x": fid, "y": round(o["tracker_confidence"], 4)})

    # depth_median always present as a key; arrays empty when no depth log.
    depth_median = {oid: [] for oid in main_objs}
    for r in depth:
        fid = r["frame_id"]
        for o in r.get("per_object_depth", []):
            oid = o.get("object_id")
            if oid in main_set and o.get("depth_median_m") is not None:
                depth_median[oid].append({"x": fid, "y": round(o["depth_median_m"], 4)})

    # ---- Flag events ------------------------------------------------------
    flag_events = []
    seen = set()

    def add_flag(frame, ftype, obj=None):
        key = (frame, ftype, obj)
        if key not in seen:
            seen.add(key)
            ev = {"frame": frame, "type": ftype}
            if obj is not None:
                ev["object"] = obj
            flag_events.append(ev)

    for r in tracking:
        fid = r["frame_id"]
        fl = r.get("flags", {})
        if fl.get("drift_flag"):
            add_flag(fid, "drift")
        if fl.get("bbox_size_change_flag"):
            add_flag(fid, "bbox_change")
        if fl.get("any_recovery_trigger"):
            add_flag(fid, "recovery_trigger")

    for r in depth:
        fid = r["frame_id"]
        for o in r.get("per_object_depth", []):
            oid = o.get("object_id")
            if oid not in main_set:
                continue
            if o.get("depth_jump_flag"):
                add_flag(fid, "depth_jump", oid)
            if o.get("any_depth_trigger"):
                add_flag(fid, "depth_trigger", oid)

    flag_events.sort(key=lambda e: (e["frame"], e["type"]))

    # ---- DINO events ------------------------------------------------------
    dino_events = [
        {"frame": r["frame_id"], "reason": r.get("trigger_reason", "")}
        for r in detection
        if r.get("detector_ran")
    ]

    # ---- Relation strip (active relation between the two main objects) -----
    # Empty list when there is no scene_graph log.
    relation_strip = []
    if scene_graph:
        rel_by_frame: dict[int, str | None] = {}
        for r in scene_graph:
            fid = r["frame_id"]
            rel = None
            for e in r.get("edges", []):
                a, b = edge_endpoints(e)
                if a in main_set and b in main_set and {a, b} == main_set:
                    rel = e.get("relation")
                    break
            rel_by_frame[fid] = rel

        cur_rel = "__INIT__"
        band_start = 0
        for f in range(n_frames):
            rel = rel_by_frame.get(f)
            if rel != cur_rel:
                if cur_rel != "__INIT__":
                    relation_strip.append({"from": band_start, "to": f, "relation": cur_rel})
                cur_rel = rel
                band_start = f
        if cur_rel != "__INIT__":
            relation_strip.append({"from": band_start, "to": n_frames, "relation": cur_rel})

    # ---- Validation status transitions + drifting frames ------------------
    status_transitions = []
    drifting_frames = set()
    last_status: dict[str, str] = {}
    for r in validation:
        fid = r["frame_id"]
        for o in r.get("tracked_objects", []):
            oid = o.get("object_id")
            if oid not in main_set:
                continue
            st = obj_status(o)
            if st == "drifting":
                drifting_frames.add(fid)
            prev = last_status.get(oid)
            if prev is not None and prev != st:
                status_transitions.append(
                    {"frame": fid, "object": oid, "from": prev, "to": st}
                )
            last_status[oid] = st

    # ---- Per-frame bounding boxes (ALL objects, for the live video overlay) -
    # Keyed by frame_id (int -> str on JSON serialization; JS numeric lookup
    # works). Frames with no objects are simply omitted -> the overlay clears.
    bbox_tracker: dict[int, list] = {}
    for r in tracking:
        fid = r["frame_id"]
        items = []
        for o in r.get("tracked_objects", []):
            oid = o.get("object_id")
            bb = o.get("bbox_xyxy")
            if not oid or not bb:
                continue
            conf = o.get("tracker_confidence")
            items.append({
                "oid": oid,
                "label": oid.rsplit("_", 1)[0],
                "bbox": [round(c, 1) for c in bb],
                "conf": round(conf, 3) if conf is not None else None,
                "status": o.get("tracker_status"),
            })
        if items:
            bbox_tracker[fid] = items

    bbox_detector: dict[int, list] = {}
    for r in detection:
        if not r.get("detector_ran"):
            continue
        fid = r["frame_id"]
        items = []
        for d in r.get("detections", []):
            bb = d.get("bbox_xyxy")
            if not bb:
                continue
            conf = d.get("confidence")
            items.append({
                "oid": d.get("object_id"),
                "label": d.get("label"),
                "bbox": [round(c, 1) for c in bb],
                "conf": round(conf, 3) if conf is not None else None,
                "is_selected": bool(d.get("is_selected")),
            })
        if items:
            bbox_detector[fid] = items

    # ---- Per-frame scene-graph node ids (for the "current objects" chips) ----
    sg_by_frame: dict[int, list] = {}
    for r in scene_graph:
        fid = r["frame_id"]
        oids = [n.get("object_id") for n in r.get("nodes", []) if n.get("object_id")]
        if oids:
            sg_by_frame[fid] = oids

    video_dims = probe_video_dims(
        repo / "example_data" / "real_data" / seq_id / "videos" / "color.mp4"
    )

    # ---- Overview summaries (general info, DINO stats, object inventory) -----
    extractors = {
        "detection":   lambda r: [d.get("object_id") for d in r.get("detections", [])],
        "tracking":    lambda r: [o.get("object_id") for o in r.get("tracked_objects", [])],
        "depth":       lambda r: [o.get("object_id") for o in r.get("per_object_depth", [])],
        "scene_graph": lambda r: [n.get("object_id") for n in r.get("nodes", [])],
        "validation":  lambda r: [o.get("object_id") for o in r.get("tracked_objects", [])],
    }
    rows_by_mod = {
        "detection": detection, "tracking": tracking, "depth": depth,
        "scene_graph": scene_graph, "validation": validation,
    }
    modules_available = [m for m in ("detection", "tracking", "depth", "scene_graph", "validation")
                         if rows_by_mod[m]]

    dino_runs = [r for r in detection if r.get("detector_ran")]
    reasons = Counter(r.get("trigger_reason", "") for r in dino_runs)
    n_runs = len(dino_runs)
    recovery_rate = (round((n_runs - reasons.get("frame_counter_K", 0)
                            - reasons.get("init", 0)) / n_runs, 3) if n_runs else 0.0)
    dino_stats = {
        "n_runs": n_runs,
        "trigger_reasons": dict(reasons),
        "recovery_rate": recovery_rate,
        "recovery_rate_note": "fraction of calls triggered by something other than frame_counter_K or init",
    }

    object_inventory = {m: unique_ids_by_label(rows_by_mod[m], extractors[m])
                        for m in modules_available}

    overview = {
        "duration_s": total_seconds,
        "n_frames": n_frames,
        "modules_available": modules_available,
        "dino_stats": dino_stats,
        "object_inventory": object_inventory,
        "llm_overview": None,   # filled in main() after model selection
    }

    # ---- LLM input summary (all floats rounded so the cache hash is stable) --
    gt_win = gt.get("failure_window_frames") if gt else None

    def conf_stats(oid):
        pts = tracker_conf.get(oid, [])
        vals = sorted(p["y"] for p in pts)
        if not vals:
            return None
        p5 = _percentile(vals, 5)
        low = [p["x"] for p in pts if p["y"] < p5]
        in_gt = sum(1 for f in low if gt_win[0] <= f <= gt_win[1]) if gt_win else 0
        return {
            "min": round(vals[0], 3), "p5": round(p5, 3),
            "mean": round(sum(vals) / len(vals), 3),
            "median": round(_percentile(vals, 50), 3),
            "max": round(vals[-1], 3),
            "low_conf_frames": len(low),
            "low_conf_frames_in_gt": in_gt,
        }

    tracker_conf_stats = {}
    for oid in main_objs:
        s = conf_stats(oid)
        if s:
            tracker_conf_stats[oid] = s

    # Cross-module mismatches over the tracker-namespace modules only. Detection
    # uses its own detection-id namespace, so comparing it here would be noise.
    ns_mods = {m: _ids_set(rows_by_mod[m], extractors[m])
               for m in ("tracking", "depth", "scene_graph", "validation") if rows_by_mod[m]}
    all_ns_ids = set().union(*ns_mods.values()) if ns_mods else set()
    mismatches = []
    for oid in sorted(all_ns_ids):
        missing = [m for m in ns_mods if oid not in ns_mods[m]]
        if missing:
            mismatches.append({
                "object_id": oid,
                "present_in": [m for m in ns_mods if oid in ns_mods[m]],
                "missing_from": missing,
            })

    blackout = [f for f in range(n_frames) if f not in bbox_tracker]
    bo_ranges: list[list[int]] = []
    for f in blackout:
        if bo_ranges and f == bo_ranges[-1][1] + 1:
            bo_ranges[-1][1] = f
        else:
            bo_ranges.append([f, f])

    llm_input = {
        "sequence_id": seq_id,
        "task_name": gt.get("task_name") if gt else None,
        "fps": fps,
        "n_frames": n_frames,
        "duration_s": total_seconds,
        "gt": None if not gt else {
            "failure_reason": gt.get("failure_reason"),
            "failure_window_s": gt.get("failure_window_s"),
            "failure_window_frames": gt.get("failure_window_frames"),
            "success_condition": gt.get("success_condition"),
        },
        "dino": dino_stats,
        "object_inventory": object_inventory,
        "tracker_confidence_stats": tracker_conf_stats,
        "cross_module_mismatches": mismatches,
        "blackout_frames": {"count": len(blackout), "ranges": bo_ranges[:25]},
    }

    return {
        "sequence_id": seq_id,
        "fps": fps,
        "n_frames": n_frames,
        "total_seconds": total_seconds,
        "gt": gt,
        "main_objects": main_objs,
        "per_object": per_object,
        "series": {
            "tracker_confidence": tracker_conf,
            "depth_median": depth_median,
        },
        "flag_events": flag_events,
        "dino_events": dino_events,
        "relation_strip": relation_strip,
        "status_transitions": status_transitions,
        "drifting_frames": sorted(drifting_frames),
        "bbox_by_frame": {"tracker": bbox_tracker, "detector": bbox_detector},
        "sg_by_frame": sg_by_frame,
        "video_dims": video_dims,
        "available": {
            "depth": bool(depth),
            "scene_graph": bool(scene_graph),
            "validation": bool(validation),
        },
        "counts": {
            "detection": len(detection),
            "tracking": len(tracking),
            "depth": len(depth),
            "scene_graph": len(scene_graph),
            "validation": len(validation),
        },
        "overview": overview,
        "_llm_input": llm_input,   # popped in main() before embedding (not shipped)
    }


# --------------------------------------------------------------------------- #
# Video copy / symlink (per-sequence subdirectory)
# --------------------------------------------------------------------------- #
def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink src->dst, falling back to a copy. Leaves an existing, correct
    symlink/copy in place."""
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return
        except OSError:
            pass
        dst.unlink()
    try:
        dst.symlink_to(src)
        print(f"    symlinked {dst.parent.name}/color.mp4 -> {src}")
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        print(f"    copied {dst.parent.name}/color.mp4 ({src.stat().st_size/1e6:.1f} MB)")


def stage_videos(repo: Path, sequences: list[str]) -> None:
    """Stage each sequence's color.mp4 into dashboard 2/video/{seq}/color.mp4 and
    drop a source.json next to it for debugging. Missing source videos are
    warned about but do not abort the build."""
    VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove the stale top-level video/color.mp4 left by older single-sequence builds.
    legacy = VIDEO_OUT_DIR / "color.mp4"
    if legacy.exists() or legacy.is_symlink():
        try:
            legacy.unlink()
        except OSError:
            pass

    for seq in sequences:
        src = repo / "example_data" / "real_data" / seq / "videos" / "color.mp4"
        dst_dir = VIDEO_OUT_DIR / seq
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "color.mp4"
        exists = src.exists()
        if exists:
            _link_or_copy(src, dst)
        else:
            print(f"    ! video missing for '{seq}': {src} "
                  f"(sequence kept; player will show a placeholder)")
            if dst.is_symlink() and not dst.exists():
                dst.unlink()  # clear a dangling link from a previous build
        (dst_dir / "source.json").write_text(
            json.dumps({"src": str(src), "exists": exists}, indent=2),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# LLM commentary (Ollama) for the overview page — build-time only, cached
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are reviewing a single robot perception sequence. The pipeline runs Grounding DINO
for object detection, a single-object tracker propagating between detections, a depth
consistency module, a scene-graph builder, and a validation module. You will be given
a structured summary of one sequence's pipeline behavior. Produce a concise overview
that helps a human reviewer understand at a glance what happened in this sequence.

GLOSSARY
- DINO recovery rate: fraction of DINO calls triggered by something other than the
  routine 30-frame timeout (frame_counter_K). High = the tracker frequently asked for
  re-detection because it was struggling.
- Unique object IDs: number of distinct object identities each module emitted for a
  given label across the full sequence. Multiple unique IDs for the same physical
  object indicate identity instability.
- GT failure window: the ground-truth-annotated frame range where the task is
  considered to have failed.
- tracker_confidence: 0 to 1 score per (frame, object); lower values mean less reliable
  tracking.
- Cross-module mismatch: an object_id present in one module's log but not another's.

INSTRUCTIONS
- Output ONLY valid JSON in this exact shape:
  {
    "summary": "...",
    "highlights": [
      {"text": "...", "severity": "info" | "warning" | "alert"}
    ]
  }
- Maximum 6 highlights. Prefer 4-5 substantive ones over a long list of generic ones.
- Severity rules:
  - "alert" = the claim references the GT failure window, OR describes a clear pipeline
    failure (an object_id in one module that isn't in another, a flag firing at a
    frame inside the GT window, etc.).
  - "warning" = a value is past P95 or below P5 of its distribution, OR a cross-module
    disagreement that isn't necessarily a failure but is worth flagging.
  - "info" = stable, descriptive observations.
- Reference specific frame numbers, object_ids, and metric values from the input.
  Do not invent values that are not present in the input.
- Do not add any text outside the JSON.
- If you cannot find enough material for a highlight, leave the list shorter rather
  than padding with generic observations.
"""


def ollama_list_models() -> set | None:
    """Set of installed model names, or None if Ollama is unreachable."""
    if requests is None:
        return None
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if resp.status_code != 200:
            return None
        return {m.get("name") for m in resp.json().get("models", []) if m.get("name")}
    except Exception:
        return None


def select_model() -> str | None:
    """Pick the preferred model, else the fallback, else None (logs the choice)."""
    models = ollama_list_models()
    if models is None:
        print("  LLM: Ollama not reachable (GET /api/tags failed) — commentary unavailable.")
        return None
    if OLLAMA_MODEL in models:
        print(f"  LLM: using model '{OLLAMA_MODEL}'")
        return OLLAMA_MODEL
    if OLLAMA_FALLBACK in models:
        print(f"  LLM: preferred '{OLLAMA_MODEL}' not installed; using fallback '{OLLAMA_FALLBACK}'")
        return OLLAMA_FALLBACK
    print(f"  LLM: neither '{OLLAMA_MODEL}' nor '{OLLAMA_FALLBACK}' installed "
          f"(have: {', '.join(sorted(models)) or 'none'}) — commentary unavailable.")
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _placeholder_llm(input_hash: str) -> dict:
    return {
        "summary": "",
        "highlights": [],
        "_meta": {
            "model": "unavailable",
            "prompt_version": PROMPT_VERSION,
            "generated_at": _utc_now(),
            "input_hash": input_hash,
        },
    }


def _call_ollama(model: str, summary: dict) -> dict | None:
    """Call Ollama /api/chat and return a validated {summary, highlights} dict, or
    None on any failure. Never raises."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(summary, indent=2)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": LLM_TEMPERATURE},
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=LLM_TIMEOUT_S)
    except Exception as e:
        print(f"      ! Ollama request failed: {e}")
        return None
    if resp.status_code != 200:
        body = ""
        try:
            body = resp.text[:200]
        except Exception:
            pass
        print(f"      ! Ollama HTTP {resp.status_code}: {body}")
        return None
    try:
        content = resp.json()["message"]["content"]
        obj = json.loads(content)
    except Exception as e:
        print(f"      ! could not parse LLM JSON response: {e}")
        return None
    summary_txt = obj.get("summary")
    if not isinstance(summary_txt, str) or not summary_txt.strip():
        print("      ! LLM response missing a 'summary' string")
        return None
    highlights = []
    for h in (obj.get("highlights") or []):
        if not isinstance(h, dict):
            continue
        text, sev = h.get("text"), h.get("severity")
        if isinstance(text, str) and text.strip() and sev in ("info", "warning", "alert"):
            highlights.append({"text": text.strip(), "severity": sev})
        if len(highlights) >= 6:
            break
    return {"summary": summary_txt.strip(), "highlights": highlights}


def get_llm_overview(model: str | None, seq_id: str, summary: dict) -> dict:
    """Return the llm_overview dict, using the on-disk cache when possible.
    Placeholders (Ollama down / call failed) are NOT cached, so a later run retries."""
    input_hash = hashlib.sha1(json.dumps(summary, sort_keys=True).encode()).hexdigest()[:12]
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = LLM_CACHE_DIR / f"{seq_id}_{PROMPT_VERSION}_{input_hash}.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            print(f"  · LLM overview for {seq_id}: cache hit ({cache_path.name})")
            return data
        except Exception:
            print(f"  · LLM overview for {seq_id}: cache unreadable, regenerating")
    if model is None:
        return _placeholder_llm(input_hash)
    print(f"  · calling Ollama for {seq_id} (model '{model}')...")
    result = _call_ollama(model, summary)
    if result is None:
        print(f"      → call failed; rendering 'unavailable' for {seq_id}")
        return _placeholder_llm(input_hash)
    llm = {
        "summary": result["summary"],
        "highlights": result["highlights"],
        "_meta": {
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "generated_at": _utc_now(),
            "input_hash": input_hash,
        },
    }
    cache_path.write_text(json.dumps(llm, indent=2), encoding="utf-8")
    print(f"      → wrote new commentary to cache ({cache_path.name})")
    return llm


def ensure_gitignore() -> None:
    """Make sure dashboard 2/.gitignore ignores the LLM cache dir."""
    gi = DASH_DIR / ".gitignore"
    entry = ".llm_cache/"
    lines = gi.read_text().splitlines() if gi.exists() else []
    if entry not in [l.strip() for l in lines]:
        lines.append(entry)
        gi.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return HTML_TEMPLATE.replace("/*__DATA_JSON__*/null", payload_json)


def render_overview_html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return OVERVIEW_TEMPLATE.replace("/*__DATA_JSON__*/null", payload_json)


def main() -> None:
    repo = find_repo_root(DASH_DIR)
    jsonl_root = repo / "pipeline" / "real_world" / "jsonl"
    print(f"REFLECT root: {repo}")

    sequences = discover_sequences(jsonl_root)
    if not sequences:
        raise SystemExit(f"No usable sequences found under {jsonl_root}")

    default = DEFAULT_SEQUENCE if DEFAULT_SEQUENCE in sequences else sequences[0]
    print(f"Discovered {len(sequences)} sequence(s): {', '.join(sequences)} "
          f"(default: {default})")

    model = select_model()   # once, before the per-sequence loop

    label_colors = {oid.rsplit("_", 1)[0].lower(): c
                    for oid, c in OBJECT_COLORS.items()}

    payload = {
        "default_sequence": default,
        "sequences": {},
        "build_time": datetime.now().isoformat(timespec="seconds"),
        "label_colors": label_colors,
        "fallback_palette": FALLBACK_PALETTE,
        "relation_colors": RELATION_COLORS,
        "llm_preferred_model": OLLAMA_MODEL,
    }

    for seq in sequences:
        d = build_data(repo, seq)
        summary = d.pop("_llm_input", {})
        d["overview"]["llm_overview"] = get_llm_overview(model, seq, summary)
        payload["sequences"][seq] = d
        avail = d["available"]
        flags = [m for m in ("depth", "scene_graph", "validation") if not avail[m]]
        note = f" [missing: {', '.join(flags)}]" if flags else ""
        bbf = d["bbox_by_frame"]
        trk_boxes = sum(len(v) for v in bbf["tracker"].values())
        det_boxes = sum(len(v) for v in bbf["detector"].values())
        print(f"  · {seq}: {d['n_frames']} frames @ {d['fps']} fps "
              f"({d['total_seconds']}s); main {d['main_objects']}; "
              f"{len(d['dino_events'])} DINO, {len(d['flag_events'])} flags, "
              f"{len(d['relation_strip'])} bands{note}")
        print(f"      bbox: tracker {len(bbf['tracker'])} frames/{trk_boxes} boxes, "
              f"detector {len(bbf['detector'])} frames/{det_boxes} boxes; "
              f"video_dims {d['video_dims']['width']}x{d['video_dims']['height']}")

    # index.html keeps its original payload shape — strip the overview-only keys
    # so the live dashboard isn't bloated with commentary it never reads.
    index_payload = {k: v for k, v in payload.items() if k != "llm_preferred_model"}
    index_payload["sequences"] = {
        sid: {k: v for k, v in d.items() if k != "overview"}
        for sid, d in payload["sequences"].items()
    }
    HTML_OUT.write_text(render_html(index_payload), encoding="utf-8")
    print(f"  wrote {HTML_OUT}")
    OVERVIEW_OUT.write_text(render_overview_html(payload), encoding="utf-8")
    print(f"  wrote {OVERVIEW_OUT}")

    ensure_gitignore()

    print("Staging videos:")
    stage_videos(repo, sequences)
    print("Build complete.")


# --------------------------------------------------------------------------- #
# HTML / CSS / JS template
# The token  /*__DATA_JSON__*/null  is replaced with the embedded PAYLOAD object.
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>REFLECT — synchronized video + time-series dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
<style>
  :root {
    --bg: #0f1419;
    --panel: #161c24;
    --panel-2: #1c242e;
    --border: #2a3441;
    --text: #e6edf3;
    --muted: #8b98a5;
    --accent: #ffb000;
    --gt: #d62728;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
    height: 100vh;
    overflow: hidden;
    display: grid;
    grid-template-rows: 50px minmax(260px, 2.3fr) 1.15fr 1.15fr 84px;
    gap: 7px;
    padding: 7px 9px;
  }

  /* ---------- Header ---------- */
  header {
    display: flex; align-items: center; gap: 14px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 0 14px; min-width: 0;
  }
  header .seq { font-weight: 700; font-size: 15px; white-space: nowrap; }
  header .task { color: var(--muted); white-space: nowrap; }
  header .reason {
    color: var(--text); flex: 1 1 auto; min-width: 0;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    font-style: italic; opacity: .9;
  }
  header .reason b { color: var(--gt); font-style: normal; }
  header .reason.empty { opacity: .5; }

  .seq-switcher {
    display: flex; flex-direction: column; justify-content: center;
    gap: 2px; min-width: 0; flex: 0 0 auto;
  }
  select#sequence-select {
    background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 4px 8px; font-size: 12px; font-family: inherit;
    max-width: 280px; cursor: pointer; font-weight: 600;
  }
  .showing {
    color: var(--muted); font-size: 10px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; max-width: 280px;
  }
  .showing b { color: var(--text); font-weight: 600; }

  header .readout {
    font-variant-numeric: tabular-nums; font-weight: 600;
    background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; white-space: nowrap; flex: 0 0 auto;
  }
  header .readout .t { color: var(--accent); }

  /* ---------- Row 2: video + side ---------- */
  .stage { display: grid; grid-template-columns: 1.55fr 1fr; gap: 7px; min-height: 0; }
  .video-wrap {
    position: relative;
    background: #000; border: 1px solid var(--border); border-radius: 8px;
    overflow: hidden; display: flex; align-items: center; justify-content: center;
    min-height: 0;
  }
  video { width: 100%; height: 100%; object-fit: contain; background: #000; }
  .video-msg {
    position: absolute; top: 8px; left: 8px; right: 8px; z-index: 5; display: none;
    background: rgba(214,39,40,0.92); color: #fff; padding: 7px 10px;
    border-radius: 6px; font-size: 12px; text-align: center;
  }
  .video-msg.show { display: block; }
  /* bbox overlay canvas — positioned to the rendered video rect in JS */
  #bbox-overlay {
    position: absolute; top: 0; left: 0; z-index: 3; pointer-events: none;
  }

  .side { display: grid; grid-template-rows: auto 1fr; gap: 7px; min-height: 0; }
  .metrics {
    display: grid; grid-template-columns: repeat(3, 1fr);
    grid-auto-rows: 1fr; gap: 7px;
  }
  .metric-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 7px 10px; display: flex; flex-direction: column; justify-content: center;
    min-width: 0;
  }
  .metric-card .label {
    color: var(--muted); font-size: 10.5px; text-transform: uppercase;
    letter-spacing: .04em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .metric-card .value {
    font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums;
    line-height: 1.15; white-space: nowrap;
  }
  .metric-card .value .unit { font-size: 12px; font-weight: 500; color: var(--muted); }

  .badges-panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 10px; display: flex; flex-direction: column; min-height: 0; overflow: hidden;
  }
  .badges-panel .ttl {
    color: var(--muted); font-size: 10.5px; text-transform: uppercase;
    letter-spacing: .04em; margin-bottom: 7px;
  }
  .badges { display: flex; flex-wrap: wrap; gap: 6px; align-content: flex-start; overflow: hidden; }
  .badge {
    font-size: 11.5px; font-weight: 600; padding: 4px 10px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--panel-2); color: var(--muted);
    white-space: nowrap; transition: background .12s, color .12s, border-color .12s, box-shadow .12s;
    display: inline-flex; align-items: center; gap: 6px;
  }
  .badge .dot {
    width: 7px; height: 7px; border-radius: 50%; background: currentColor; opacity: .45;
  }
  .badge.active { color: #fff; opacity: 1; }
  .badge.active .dot { opacity: 1; background: #fff; }
  .badge.active.c-gt       { background: var(--gt);      border-color: var(--gt); }
  .badge.active.c-dino     { background: #ff7f0e;        border-color: #ff7f0e; }
  .badge.active.c-drift    { background: #d62728;        border-color: #d62728; }
  .badge.active.c-depth    { background: #9467bd;        border-color: #9467bd; }
  .badge.active.c-recovery { background: #e0a800; border-color: #e0a800; color:#1a1a1a; }
  .badge.active.c-recovery .dot { background:#1a1a1a; }
  .badge.active.c-status   { background: #d62728;        border-color: #d62728; }
  .badge.c-relation {
    background: var(--panel-2); color: var(--text); border-color: var(--border);
  }
  .badge.c-relation b { color: var(--accent); }

  /* ---------- "current objects" block (below the event badges) ---------- */
  .obj-block {
    margin-top: 8px; padding-top: 7px; border-top: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 2px; flex: 0 0 auto;
  }
  .obj-row { display: flex; align-items: flex-start; gap: 8px; min-height: 20px; }
  .obj-row-label {
    color: var(--muted); font-size: 10.5px; text-transform: uppercase;
    letter-spacing: .04em; white-space: nowrap; flex: 0 0 86px; padding-top: 4px;
  }
  .obj-row-label b { color: var(--text); font-weight: 700; }
  .obj-chips { display: flex; flex-wrap: wrap; align-items: center; min-width: 0; }
  .obj-chip {
    display: inline-block; padding: 2px 8px; margin: 2px;
    border-radius: 12px; font-size: 12px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    border: 1px solid currentColor; line-height: 1.35; white-space: nowrap;
    transition: opacity 180ms ease, transform 180ms ease;
  }
  .obj-chip.entering, .obj-chip.exiting { opacity: 0; transform: scale(0.85); }
  .obj-chip.phantom { border-style: dashed; }

  /* ---------- Chart panels ---------- */
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    position: relative; min-height: 0; padding: 4px 8px 2px 8px; overflow: hidden;
  }
  .panel .panel-ttl {
    position: absolute; top: 5px; left: 12px; z-index: 3;
    color: var(--muted); font-size: 10.5px; text-transform: uppercase;
    letter-spacing: .04em; pointer-events: none;
  }
  .panel .legend {
    position: absolute; top: 4px; right: 10px; z-index: 3;
    display: flex; gap: 12px; font-size: 10.5px; pointer-events: none;
  }
  .panel .legend span { display: inline-flex; align-items: center; gap: 4px; color: var(--muted); }
  .panel .legend i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  .chart-host { position: absolute; inset: 0; }
  canvas { display: block; }

  /* per-panel "data not available" message */
  .panel-msg {
    position: absolute; inset: 22px 0 0 0; display: none;
    align-items: center; justify-content: center; z-index: 4;
    color: var(--muted); font-size: 12.5px; font-style: italic;
    pointer-events: none; text-align: center; padding: 0 12px;
  }
  .panel-msg.show { display: flex; }

  /* playhead overlay (one per panel) */
  .playhead {
    position: absolute; top: 0; bottom: 0; left: 0; width: 0;
    border-left: 2px solid #ffffff; box-shadow: 0 0 6px rgba(255,255,255,.5);
    pointer-events: none; z-index: 5; will-change: transform;
  }

  /* relation strip */
  .strip-host { position: absolute; inset: 0; }
  .strip-legend {
    position: absolute; bottom: 3px; left: 12px; right: 10px; z-index: 3;
    display: flex; flex-wrap: wrap; gap: 10px; font-size: 10px; pointer-events: none;
  }
  .strip-legend span { display: inline-flex; align-items: center; gap: 4px; color: var(--muted); }
  .strip-legend i { width: 11px; height: 9px; border-radius: 2px; display: inline-block; }
</style>
</head>
<body>

<header>
  <span class="seq" id="seqId">—</span>
  <span class="task" id="taskName">—</span>
  <span class="reason" id="reason"></span>
  <div class="seq-switcher">
    <select id="sequence-select"></select>
    <span class="showing" id="showing"></span>
  </div>
  <span class="readout">Frame <span id="frameNo">0</span> / <span id="frameTot">0</span> &middot; t = <span class="t" id="timeNo">0.00</span> s</span>
</header>

<section class="stage">
  <div class="video-wrap">
    <div class="video-msg" id="video-msg"></div>
    <video id="video" controls preload="auto" playsinline>
      Your browser does not support the video tag.
    </video>
    <canvas id="bbox-overlay"></canvas>
  </div>
  <div class="side">
    <div class="metrics" id="metrics">
      <div class="metric-card"><div class="label">Frame</div><div class="value" id="m-frame">0</div></div>
      <div class="metric-card"><div class="label">Time</div><div class="value"><span id="m-time">0.00</span><span class="unit"> s</span></div></div>
      <div class="metric-card"><div class="label">Relation</div><div class="value" id="m-rel" style="font-size:16px">—</div></div>
      <div class="metric-card"><div class="label" id="m-conf-a-l">Conf · A</div><div class="value" id="m-conf-a">—</div></div>
      <div class="metric-card"><div class="label" id="m-conf-b-l">Conf · B</div><div class="value" id="m-conf-b">—</div></div>
      <div class="metric-card"><div class="label" id="m-dep-a-l">Depth · A</div><div class="value" id="m-dep-a">—<span class="unit"> m</span></div></div>
    </div>
    <div class="badges-panel">
      <div class="ttl">Active events @ playhead</div>
      <div class="badges" id="badges"></div>
      <div class="obj-block" id="obj-block">
        <div class="obj-row">
          <span class="obj-row-label">Tracker (<b id="trk-count">0</b>)</span>
          <span class="obj-chips" id="trk-chips"></span>
        </div>
        <div class="obj-row">
          <span class="obj-row-label">Scene graph (<b id="sg-count">0</b>)</span>
          <span class="obj-chips" id="sg-chips"></span>
        </div>
      </div>
    </div>
  </div>
</section>

<div class="panel" id="panel-conf">
  <div class="panel-ttl">Tracker confidence</div>
  <div class="legend" id="legend-conf"></div>
  <div class="chart-host"><canvas id="chartConf"></canvas></div>
  <div class="playhead" id="ph-conf"></div>
</div>

<div class="panel" id="panel-depth">
  <div class="panel-ttl">Depth median (m)</div>
  <div class="legend" id="legend-depth"></div>
  <div class="chart-host"><canvas id="chartDepth"></canvas></div>
  <div class="panel-msg" id="depth-msg">Depth not available for this sequence.</div>
  <div class="playhead" id="ph-depth"></div>
</div>

<div class="panel" id="panel-strip">
  <div class="panel-ttl" id="strip-ttl">Scene-graph relation</div>
  <div class="strip-host"><canvas id="stripCanvas"></canvas></div>
  <div class="strip-legend" id="strip-legend"></div>
  <div class="panel-msg" id="strip-msg">Scene graph not available for this sequence.</div>
  <div class="playhead" id="ph-strip"></div>
</div>

<script>
const PAYLOAD = /*__DATA_JSON__*/null;
</script>
<script>
(function () {
  "use strict";

  // ---- top-level (build-time constant) maps ----
  const SEQS = PAYLOAD.sequences;
  const RELCOL = PAYLOAD.relation_colors || {};
  const LABEL_COLORS = PAYLOAD.label_colors || {};
  const FALLBACK = PAYLOAD.fallback_palette || ["#2ca02c", "#9467bd", "#17becf", "#bcbd22"];
  const KEYWORD_COLORS = {};
  Object.keys(LABEL_COLORS).forEach(lbl => {
    const w = lbl.split(/\s+/).pop();
    if (w) KEYWORD_COLORS[w] = LABEL_COLORS[lbl];
  });

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  // Resolve an object color by label so the same class is consistent across
  // sequences (all apples red, all bowls blue); fall back to a stable palette.
  function resolveColor(label, idx) {
    const key = String(label || "").toLowerCase().trim();
    if (LABEL_COLORS[key]) return LABEL_COLORS[key];
    const words = key.split(/\s+/);
    for (const w in KEYWORD_COLORS) { if (words.includes(w)) return KEYWORD_COLORS[w]; }
    return FALLBACK[idx % FALLBACK.length];
  }
  const relLabel = r => (r ? r.replace(/_/g, " ") : "(none)");
  const $ = id => document.getElementById(id);
  const fmt = (v, d) => (v == null ? "—" : Number(v).toFixed(d));

  // ---- module-level per-sequence state (recomputed on every switch) ----
  let CURRENT = null;
  let FPS = 30, NF = 1, MAXF = 0;
  let MAIN = [], OA = null, OB = null, COL = {};
  let gt = {}, gtFrames = null;
  let colA = "#888", colB = "#888";
  let confFill = {}, depFill = {};
  let dinoFrames = [], driftFrames = [], recoveryFrames = [], depthJumpFrames = [], driftingFrames = [];
  let confChart = null, depChart = null;
  let geom = { left: 50, width: 0 };
  let curSid = null;
  let mainSet = new Set();   // main object_ids of CURRENT (for phantom detection)
  let rafId = null;          // bbox overlay requestAnimationFrame handle
  const Y_AXIS_W = 50;

  // ---- DOM refs ----
  const video = $("video");
  const stripCanvas = $("stripCanvas");
  const stripCtx = stripCanvas.getContext("2d");
  const phConf = $("ph-conf"), phDepth = $("ph-depth"), phStrip = $("ph-strip");
  const bboxCanvas = $("bbox-overlay");
  const bboxCtx = bboxCanvas.getContext("2d");

  // ---- generic helpers ----
  function buildFill(points) {
    const arr = new Array(NF).fill(null);
    let pi = 0, last = null;
    const pts = points || [];
    for (let f = 0; f < NF; f++) {
      while (pi < pts.length && pts[pi].x <= f) { last = pts[pi].y; pi++; }
      arr[f] = last;
    }
    return arr;
  }
  const sortedNums = a => a.slice().sort((x, y) => x - y);
  function nearWithin(sorted, f, tol) {
    if (!sorted.length) return false;
    let lo = 0, hi = sorted.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] < f - tol) lo = mid + 1;
      else if (sorted[mid] > f + tol) hi = mid - 1;
      else return true;
    }
    return false;
  }
  function contains(sorted, f) {
    let lo = 0, hi = sorted.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] < f) lo = mid + 1;
      else if (sorted[mid] > f) hi = mid - 1;
      else return true;
    }
    return false;
  }
  function relationAt(f) {
    const bands = (CURRENT && CURRENT.relation_strip) || [];
    let lo = 0, hi = bands.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1, b = bands[mid];
      if (f < b.from) hi = mid - 1;
      else if (f >= b.to) lo = mid + 1;
      else return b.relation;
    }
    return null;
  }

  // ---- GT window background plugin (reads module-level gtFrames) ----
  const gtBandPlugin = {
    id: "gtBand",
    beforeDatasetsDraw(chart) {
      if (!gtFrames) return;
      const { ctx, chartArea: ca, scales: { x } } = chart;
      const x0 = x.getPixelForValue(gtFrames[0]);
      const x1 = x.getPixelForValue(gtFrames[1]);
      ctx.save();
      ctx.fillStyle = "rgba(214,39,40,0.10)";
      ctx.fillRect(x0, ca.top, x1 - x0, ca.bottom - ca.top);
      ctx.strokeStyle = "rgba(214,39,40,0.45)";
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0 + .5, ca.top); ctx.lineTo(x0 + .5, ca.bottom);
      ctx.moveTo(x1 - .5, ca.top); ctx.lineTo(x1 - .5, ca.bottom);
      ctx.stroke();
      ctx.restore();
    }
  };

  // Force a constant y-axis width so all panels share the same plot left edge.
  const fixYAxis = scale => { scale.width = Y_AXIS_W; };
  const baseOpts = (yTitle, yMin, yMax) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    parsing: false,
    normalized: true,
    interaction: { mode: "index", intersect: false },
    layout: { padding: { left: 0, right: 14, top: 22, bottom: 2 } },
    scales: {
      x: {
        type: "linear", min: 0, max: MAXF,
        ticks: { color: "#8b98a5", maxTicksLimit: 12, font: { size: 10 } },
        grid: { color: "rgba(255,255,255,0.04)" },
        title: { display: false },
      },
      y: {
        min: yMin, max: yMax, afterFit: fixYAxis,
        ticks: { color: "#8b98a5", font: { size: 10 } },
        grid: { color: "rgba(255,255,255,0.06)" },
        title: { display: true, text: yTitle, color: "#8b98a5", font: { size: 10 } },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        enabled: true, mode: "index", intersect: false,
        filter: it => it.dataset.showLine !== false,
        callbacks: { title: items => "Frame " + (items.length ? items[0].parsed.x : "") },
      },
    },
    elements: { point: { radius: 0 }, line: { borderWidth: 1.6, tension: 0.15 } },
  });

  function lineDataset(oid, points, color) {
    return {
      label: (COL[oid] && COL[oid].label) || oid || "",
      data: points || [],
      borderColor: color,
      backgroundColor: color,
      pointRadius: 0,
      spanGaps: true,
    };
  }
  function markerDataset(label, frames, yVal, color, rot) {
    return {
      label,
      data: frames.map(f => ({ x: f, y: yVal })),
      showLine: false,
      pointStyle: "triangle",
      rotation: rot || 0,
      radius: 4,
      pointRadius: 4,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 0,
    };
  }

  function destroyCharts() {
    if (confChart) { confChart.destroy(); confChart = null; }
    if (depChart) { depChart.destroy(); depChart = null; }
  }
  function buildCharts() {
    const S = CURRENT.series;
    confChart = new Chart($("chartConf"), {
      type: "line",
      data: {
        datasets: [
          lineDataset(OA, S.tracker_confidence[OA], colA),
          lineDataset(OB, S.tracker_confidence[OB], colB),
          markerDataset("DINO ran", dinoFrames, 0.03, "rgba(160,170,180,0.7)", 0),
          markerDataset("drift", driftFrames, 0.10, "#d62728", 180),
        ],
      },
      options: baseOpts("confidence", 0, 1),
      plugins: [gtBandPlugin],
    });

    let lo = Infinity, hi = -Infinity;
    [OA, OB].forEach(o => (S.depth_median[o] || []).forEach(p => {
      if (p.y < lo) lo = p.y; if (p.y > hi) hi = p.y;
    }));
    if (!isFinite(lo)) { lo = 0; hi = 2; }
    const pad = (hi - lo) * 0.12 || 0.1;
    const dLo = Math.max(0, +(lo - pad).toFixed(2));
    const dHi = +(hi + pad).toFixed(2);
    depChart = new Chart($("chartDepth"), {
      type: "line",
      data: {
        datasets: [
          lineDataset(OA, S.depth_median[OA], colA),
          lineDataset(OB, S.depth_median[OB], colB),
          markerDataset("depth jump", depthJumpFrames, dLo + (dHi - dLo) * 0.05, "#9467bd", 0),
        ],
      },
      options: baseOpts("metres", dLo, dHi),
      plugins: [gtBandPlugin],
    });
  }

  function fillLegend(elId, items) {
    $(elId).innerHTML = items.map(
      it => `<span><i style="background:${it.c}"></i>${escapeHtml(it.t)}</span>`
    ).join("");
  }
  function fillChartLegends() {
    const la = (COL[OA] && COL[OA].label) || "A";
    const lb = (COL[OB] && COL[OB].label) || "B";
    fillLegend("legend-conf", [
      { c: colA, t: la }, { c: colB, t: lb },
      { c: "rgba(160,170,180,0.9)", t: "DINO" }, { c: "#d62728", t: "drift" },
    ]);
    fillLegend("legend-depth", [
      { c: colA, t: la }, { c: colB, t: lb }, { c: "#9467bd", t: "depth jump" },
    ]);
  }

  // ---- relation strip (own canvas) ----
  function drawStrip() {
    const host = stripCanvas.parentElement;
    const cssW = host.clientWidth, cssH = host.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    stripCanvas.width = cssW * dpr;
    stripCanvas.height = cssH * dpr;
    stripCanvas.style.width = cssW + "px";
    stripCanvas.style.height = cssH + "px";
    stripCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    stripCtx.clearRect(0, 0, cssW, cssH);

    const left = plotLeft();
    const right = cssW - 14;
    const width = right - left;
    const topPad = 22, botPad = 18;
    const top = topPad, h = cssH - topPad - botPad;
    if (width <= 0 || h <= 0) return;

    if (gtFrames) {
      const gx0 = left + (gtFrames[0] / MAXF) * width;
      const gx1 = left + (gtFrames[1] / MAXF) * width;
      stripCtx.fillStyle = "rgba(214,39,40,0.10)";
      stripCtx.fillRect(gx0, top, gx1 - gx0, h);
    }
    for (const b of (CURRENT.relation_strip || [])) {
      const x0 = left + (b.from / MAXF) * width;
      const x1 = left + (b.to / MAXF) * width;
      stripCtx.fillStyle = (b.relation && RELCOL[b.relation]) ? RELCOL[b.relation] : "rgba(255,255,255,0.05)";
      stripCtx.fillRect(x0, top, Math.max(1, x1 - x0), h);
    }
    stripCtx.fillStyle = "#8b98a5";
    stripCtx.font = "10px -apple-system, sans-serif";
    stripCtx.textAlign = "center";
    const nTicks = 10;
    for (let i = 0; i <= nTicks; i++) {
      const f = Math.round((MAXF * i) / nTicks);
      const x = left + (f / MAXF) * width;
      stripCtx.fillText(String(f), x, cssH - 6);
    }
  }

  // ---- shared plot geometry ----
  function plotLeft() { return confChart && confChart.chartArea ? confChart.chartArea.left : Y_AXIS_W; }
  function plotRight() { return confChart && confChart.chartArea ? confChart.chartArea.right : 0; }
  function recomputeGeom() { geom.left = plotLeft(); geom.width = plotRight() - plotLeft(); }
  function positionPlayheads(frameId) {
    const frac = MAXF > 0 ? frameId / MAXF : 0;
    const tf = "translateX(" + (geom.left + frac * geom.width).toFixed(1) + "px)";
    phConf.style.transform = tf; phDepth.style.transform = tf; phStrip.style.transform = tf;
  }

  // ---- video bbox overlay (rAF-driven; canvas is in source-pixel space) ----
  // The canvas internal resolution == video.videoWidth/Height, so bbox coords
  // are drawn raw. CSS positions the canvas onto the letterboxed video rect.
  function computeVideoRect() {
    const vw = video.videoWidth, vh = video.videoHeight;
    const ew = video.clientWidth, eh = video.clientHeight;
    if (!vw || !vh || !ew || !eh) return null;
    const scale = Math.min(ew / vw, eh / vh);
    const rw = vw * scale, rh = vh * scale;
    return { x: (ew - rw) / 2, y: (eh - rh) / 2, w: rw, h: rh, scale };
  }
  function sizeBboxCanvas() {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;
    if (bboxCanvas.width !== vw) bboxCanvas.width = vw;
    if (bboxCanvas.height !== vh) bboxCanvas.height = vh;
    const r = computeVideoRect();
    if (!r) return;
    bboxCanvas.style.left = r.x + "px";
    bboxCanvas.style.top = r.y + "px";
    bboxCanvas.style.width = r.w + "px";
    bboxCanvas.style.height = r.h + "px";
  }
  function clearBbox() { bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height); }
  function statusColor(status) {
    if (status === "lost") return "#d62728";
    if (status === "drifting" || status === "occluded") return "#ff7f0e";
    return "#1f77b4";
  }
  // s = display scale; line widths & font are divided by it so they render at a
  // constant on-screen size regardless of the source resolution.
  function drawBox(bb, color, label, s) {
    if (!bb || bb.length < 4) return;
    const x1 = bb[0], y1 = bb[1], x2 = bb[2], y2 = bb[3];
    const ctx = bboxCtx;
    ctx.lineWidth = 3 / s;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const fontPx = 12 / s, padX = 3 / s, padY = 2 / s;
    ctx.font = fontPx + "px ui-monospace, Menlo, monospace";
    ctx.textBaseline = "top";
    const tw = ctx.measureText(label).width;
    const bh = fontPx + 2 * padY;
    let ly = y1 - bh;
    if (ly < 0) ly = y1;   // keep label visible for boxes at the top edge
    ctx.fillStyle = color;
    ctx.fillRect(x1, ly, tw + 2 * padX, bh);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x1 + padX, ly + padY);
  }
  function drawBboxes() {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;   // metadata not ready — skip, retry next tick
    if (bboxCanvas.width !== vw || bboxCanvas.height !== vh) sizeBboxCanvas();
    clearBbox();
    if (!CURRENT || !CURRENT.bbox_by_frame) return;
    const r = computeVideoRect();
    const s = r ? r.scale : 1;
    const f = Math.round((video.currentTime || 0) * FPS);
    const bbf = CURRENT.bbox_by_frame;
    const trk = (bbf.tracker && bbf.tracker[f]) || [];
    const det = (bbf.detector && bbf.detector[f]) || [];
    trk.forEach(b => drawBox(b.bbox, statusColor(b.status), "TRK " + b.oid, s));
    det.forEach(b => drawBox(
      b.bbox, "#2ca02c",
      "DET " + (b.label || "") + (b.conf != null ? " " + b.conf.toFixed(2) : ""), s));
  }
  function startBboxLoop() {
    if (rafId !== null) return;
    (function tick() { drawBboxes(); rafId = requestAnimationFrame(tick); })();
  }
  function stopBboxLoop() {
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
    drawBboxes();   // one final draw so the paused frame looks right
  }

  // ---- badges (DOM built once; tests read module-level state) ----
  const badgeDefs = [
    { id: "gt", cls: "c-gt", label: "Inside GT failure window",
      test: f => gtFrames && f >= gtFrames[0] && f <= gtFrames[1] },
    { id: "dino", cls: "c-dino", label: "DINO ran",
      test: f => nearWithin(dinoFrames, f, 3) },
    { id: "drift", cls: "c-drift", label: "Drift",
      test: f => nearWithin(driftFrames, f, 3) },
    { id: "depth", cls: "c-depth", label: "Depth jump",
      test: f => nearWithin(depthJumpFrames, f, 3) },
    { id: "recovery", cls: "c-recovery", label: "Recovery",
      test: f => nearWithin(recoveryFrames, f, 3) },
    { id: "status", cls: "c-status", label: "Status: drifting",
      test: f => contains(driftingFrames, f) },
  ];
  const badgeNodes = {};
  let relBadge = null;
  function buildBadgesDOM() {
    const badgesEl = $("badges");
    badgesEl.innerHTML = "";
    badgeDefs.forEach(d => {
      const el = document.createElement("span");
      el.className = "badge " + d.cls;
      el.innerHTML = '<span class="dot"></span>' + escapeHtml(d.label);
      badgesEl.appendChild(el);
      badgeNodes[d.id] = el;
    });
    relBadge = document.createElement("span");
    relBadge.className = "badge c-relation";
    badgesEl.appendChild(relBadge);
  }

  // ---- live update ----
  function updateLiveMetrics(f) {
    $("m-frame").textContent = f;
    $("m-time").textContent = (f / FPS).toFixed(2);
    $("m-conf-a").textContent = fmt(confFill[OA] ? confFill[OA][f] : null, 2);
    $("m-conf-b").textContent = fmt(confFill[OB] ? confFill[OB][f] : null, 2);
    $("m-dep-a").innerHTML = fmt(depFill[OA] ? depFill[OA][f] : null, 2) + '<span class="unit"> m</span>';
    const rel = relationAt(f);
    const m = $("m-rel");
    m.textContent = relLabel(rel);
    m.style.color = rel && RELCOL[rel] ? RELCOL[rel] : "var(--muted)";
  }
  function updateBadges(f) {
    badgeDefs.forEach(d => badgeNodes[d.id].classList.toggle("active", !!d.test(f)));
    relBadge.innerHTML = "Relation: <b>&nbsp;" + escapeHtml(relLabel(relationAt(f))) + "</b>";
  }

  // ---- "current objects" chips (tracker + scene graph), with fade in/out ----
  function hexToRgba(hex, a) {
    const m = String(hex).replace("#", "");
    const r = parseInt(m.slice(0, 2), 16), g = parseInt(m.slice(2, 4), 16), b = parseInt(m.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + a + ")";
  }
  function chipColor(oid) {
    const label = String(oid).replace(/_\d+$/, "").toLowerCase();
    if (LABEL_COLORS[label]) return LABEL_COLORS[label];
    const words = label.split(/\s+/);
    for (const w in KEYWORD_COLORS) { if (words.includes(w)) return KEYWORD_COLORS[w]; }
    return "#8b98a5";   // neutral gray for unknown labels
  }
  const isPhantom = oid => !mainSet.has(oid);
  function makeChip(oid) {
    const c = chipColor(oid);
    const el = document.createElement("div");
    el.className = "obj-chip entering" + (isPhantom(oid) ? " phantom" : "");
    el.dataset.oid = oid;
    el.textContent = oid;
    el.style.color = c;
    el.style.background = hexToRgba(c, 0.12);
    return el;
  }
  function diffChips(contId, countId, desiredOids) {
    const cont = $(contId);
    const desired = new Set(desiredOids);
    const existing = new Map();
    cont.querySelectorAll(".obj-chip").forEach(el => existing.set(el.dataset.oid, el));
    // exits: fade/scale out, then remove (with a timeout fallback)
    existing.forEach((el, oid) => {
      if (!desired.has(oid) && !el.classList.contains("exiting")) {
        el.classList.add("exiting");
        el.addEventListener("transitionend", () => {
          if (el.classList.contains("exiting")) el.remove();
        }, { once: true });
        setTimeout(() => {
          if (el.isConnected && el.classList.contains("exiting")) el.remove();
        }, 260);
      }
    });
    // entries: create + animate in (or revive a chip that was mid-exit)
    desired.forEach(oid => {
      const ex = existing.get(oid);
      if (ex) { ex.classList.remove("exiting"); return; }
      const chip = makeChip(oid);
      cont.appendChild(chip);
      requestAnimationFrame(() => requestAnimationFrame(() => chip.classList.remove("entering")));
    });
    $(countId).textContent = desired.size;
  }
  function clearChips() {
    $("trk-chips").innerHTML = ""; $("sg-chips").innerHTML = "";
    $("trk-count").textContent = "0"; $("sg-count").textContent = "0";
  }
  function updateObjChips(f) {
    if (!CURRENT) return;
    const bbf = CURRENT.bbox_by_frame || {};
    const trkList = (bbf.tracker && bbf.tracker[f]) ? bbf.tracker[f].map(o => o.oid) : [];
    const sgList = (CURRENT.sg_by_frame && CURRENT.sg_by_frame[f]) ? CURRENT.sg_by_frame[f] : [];
    diffChips("trk-chips", "trk-count", trkList);
    diffChips("sg-chips", "sg-count", sgList);
  }
  function update(t) {
    let f = Math.round(t * FPS);
    if (f < 0) f = 0; if (f > MAXF) f = MAXF;
    $("frameNo").textContent = f;
    $("timeNo").textContent = (f / FPS).toFixed(2);
    positionPlayheads(f);
    updateLiveMetrics(f);
    updateBadges(f);
    updateObjChips(f);
  }

  // ---- per-sequence derived state ----
  function computeDerived() {
    FPS = CURRENT.fps || 30;
    NF = CURRENT.n_frames || 1;
    MAXF = NF - 1;
    MAIN = (CURRENT.main_objects || []).slice();
    mainSet = new Set(CURRENT.main_objects || []);
    OA = MAIN[0] || null;
    OB = MAIN[1] || MAIN[0] || null;
    const ai = MAIN.findIndex(o => String(o).toLowerCase().includes("apple"));
    if (ai === 1) { OA = MAIN[1]; OB = MAIN[0]; }
    COL = CURRENT.per_object || {};
    gt = CURRENT.gt || {};
    gtFrames = (gt.failure_window_frames && gt.failure_window_frames.length === 2)
      ? gt.failure_window_frames : null;
    colA = resolveColor(OA && COL[OA] ? COL[OA].label : OA, 0);
    colB = resolveColor(OB && COL[OB] ? COL[OB].label : OB, 1);

    const S = CURRENT.series;
    confFill = {}; depFill = {};
    [OA, OB].forEach(o => {
      if (!o) return;
      confFill[o] = buildFill(S.tracker_confidence[o]);
      depFill[o] = buildFill(S.depth_median[o]);
    });

    const fe = CURRENT.flag_events || [];
    dinoFrames = sortedNums((CURRENT.dino_events || []).map(e => e.frame));
    driftFrames = sortedNums(fe.filter(e => e.type === "drift").map(e => e.frame));
    recoveryFrames = sortedNums(fe.filter(e => e.type === "recovery_trigger").map(e => e.frame));
    depthJumpFrames = sortedNums(fe.filter(e => e.type === "depth_jump").map(e => e.frame));
    driftingFrames = CURRENT.drifting_frames || [];
  }

  // ---- video source ----
  function setVideoSrc(sid) {
    $("video-msg").classList.remove("show");
    curSid = sid;
    video.src = "video/" + sid + "/color.mp4";
    video.load();
    video.addEventListener("loadedmetadata", () => {
      try { video.currentTime = 0; } catch (e) {}
      update(0);
    }, { once: true });
  }

  // ---- the single source of truth: load one sequence into the page ----
  function loadSequence(sid) {
    CURRENT = SEQS[sid];
    computeDerived();
    clearChips();   // immediate clear on switch; frame-0 chips animate in via update(0)
    clearBbox();

    // header text
    $("seqId").textContent = CURRENT.sequence_id;
    $("frameTot").textContent = MAXF;
    const reasonEl = $("reason");
    if (gt.task_name) {
      $("taskName").textContent = "“" + gt.task_name + "”";
    } else {
      $("taskName").textContent = "";
    }
    if (gt.failure_reason) {
      reasonEl.className = "reason";
      reasonEl.innerHTML = "<b>GT failure:</b> " + escapeHtml(gt.failure_reason);
    } else {
      reasonEl.className = "reason empty";
      reasonEl.textContent = "No ground-truth annotation for this sequence.";
    }

    // metric card labels + object colors
    const la = (COL[OA] && COL[OA].label) || "A";
    const lb = (COL[OB] && COL[OB].label) || "B";
    $("m-conf-a-l").textContent = "Conf · " + la;
    $("m-conf-b-l").textContent = "Conf · " + lb;
    $("m-dep-a-l").textContent = "Depth · " + la;
    $("m-conf-a").style.color = colA;
    $("m-conf-b").style.color = colB;
    $("m-dep-a").style.color = colA;
    $("strip-ttl").textContent = "Scene-graph relation · " + la + " ↔ " + lb;

    // caption + dropdown sync
    $("showing").innerHTML = "Currently showing: <b>" + escapeHtml(CURRENT.sequence_id) +
      "</b> · " + NF + " frames · " + (CURRENT.total_seconds || 0).toFixed(1) + "s";
    $("sequence-select").value = sid;

    // availability messages
    const avail = CURRENT.available || {};
    $("depth-msg").classList.toggle("show", !avail.depth);
    $("strip-msg").classList.toggle("show", !avail.scene_graph);

    // video
    setVideoSrc(sid);

    // charts + strip
    destroyCharts();
    buildCharts();
    fillChartLegends();

    // reset playhead, paint after layout settles
    positionPlayheads(0);
    requestAnimationFrame(() => {
      recomputeGeom(); drawStrip(); update(0);
      requestAnimationFrame(() => { recomputeGeom(); drawStrip(); update(video.currentTime || 0); });
    });
  }

  function switchSequence(sid) {
    if (!SEQS[sid]) return;
    stopBboxLoop();
    video.pause();
    loadSequence(sid);
  }

  // ---- one-time setup ----
  function initOnce() {
    // dropdown options
    const sel = $("sequence-select");
    Object.keys(SEQS).forEach(sid => {
      const o = document.createElement("option");
      o.value = sid;
      const t = SEQS[sid].gt && SEQS[sid].gt.task_name;
      o.textContent = t ? (sid + " — " + t) : sid;
      sel.appendChild(o);
    });
    sel.addEventListener("change", () => switchSequence(sel.value));

    // badges + strip legend (RELCOL is constant)
    buildBadgesDOM();
    (function () {
      const order = ["near", "left_of", "above", "on_top_of", "inside"];
      const items = order.filter(r => RELCOL[r]).map(r => ({ c: RELCOL[r], t: r.replace(/_/g, " ") }));
      items.push({ c: "rgba(255,255,255,0.06)", t: "(none)" });
      $("strip-legend").innerHTML = items.map(
        it => `<span><i style="background:${it.c}"></i>${escapeHtml(it.t)}</span>`
      ).join("");
    })();

    // video listeners (persistent)
    video.addEventListener("timeupdate", () => update(video.currentTime));
    video.addEventListener("seeked", () => update(video.currentTime));
    video.addEventListener("seeking", () => update(video.currentTime));
    video.addEventListener("loadeddata", () => $("video-msg").classList.remove("show"));
    video.addEventListener("error", () => {
      $("video-msg").textContent = "Video file not found at video/" + curSid + "/color.mp4.";
      $("video-msg").classList.add("show");
    });

    // bbox overlay lifecycle: rAF while playing, single draws otherwise
    video.addEventListener("play", startBboxLoop);
    video.addEventListener("pause", stopBboxLoop);
    video.addEventListener("ended", stopBboxLoop);
    video.addEventListener("seeked", drawBboxes);
    video.addEventListener("loadedmetadata", () => { sizeBboxCanvas(); drawBboxes(); });

    // resize
    let rT = null;
    window.addEventListener("resize", () => {
      clearTimeout(rT);
      rT = setTimeout(() => {
        recomputeGeom(); drawStrip(); sizeBboxCanvas(); drawBboxes();
        update(video.currentTime || 0);
      }, 80);
    });

    // initial sequence
    const defSid = SEQS[PAYLOAD.default_sequence] ? PAYLOAD.default_sequence : Object.keys(SEQS)[0];
    loadSequence(defSid);
  }

  initOnce();
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Overview page template (separate surface; reuses index's visual language)
# --------------------------------------------------------------------------- #
OVERVIEW_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>REFLECT — sequence overview</title>
<style>
  :root {
    --bg: #0f1419;
    --panel: #161c24;
    --panel-2: #1c242e;
    --border: #2a3441;
    --text: #e6edf3;
    --muted: #8b98a5;
    --accent: #ffb000;
    --gt: #d62728;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px; min-height: 100vh; padding: 8px 14px;
  }
  code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.9em; background: var(--panel-2); padding: 1px 5px; border-radius: 4px;
  }

  /* ---------- Header (matches index.html) ---------- */
  .ov-header {
    display: flex; align-items: center; gap: 14px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 16px; margin-bottom: 0.6rem;
  }
  .ov-title { font-weight: 700; font-size: 18px; white-space: nowrap; }
  .spacer { flex: 1 1 auto; }
  .seq-switcher {
    display: flex; flex-direction: column; justify-content: center;
    gap: 2px; min-width: 0; flex: 0 0 auto; align-items: flex-end;
  }
  select#sequence-select {
    background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 4px 8px; font-size: 12px; font-family: inherit;
    max-width: 320px; cursor: pointer; font-weight: 600;
  }
  .showing {
    color: var(--muted); font-size: 10px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; max-width: 360px;
  }
  .showing b { color: var(--text); font-weight: 600; }

  /* ---------- Cards ---------- */
  .overview-body { display: grid; grid-template-columns: 1fr; gap: 0.6rem; }
  .overview-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 18px;
    opacity: 0; transform: translateY(8px);
    transition: opacity .18s ease-out, transform .18s ease-out;
  }
  .overview-card.in { opacity: 1; transform: none; }
  .overview-body.switching .overview-card.in { opacity: 0; transform: translateY(6px); }
  .card-title {
    font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
    color: var(--muted); margin: 0 0 8px; font-weight: 700;
  }

  /* key/value */
  .kv { display: grid; grid-template-columns: 170px 1fr; gap: 5px 16px; align-items: baseline; }
  .kv .k { color: var(--muted); font-size: 12px; }
  .kv .v { color: var(--text); font-size: 13px; }
  .kv .v.reason { font-style: italic; }
  .kv .v .sub { color: var(--muted); }
  .muted-italic { color: var(--muted); font-style: italic; }
  .subhead { color: var(--muted); font-size: 12px; margin: 9px 0 6px; }

  /* pills (match index event-badge styling) */
  .pill-row { display: flex; flex-wrap: wrap; gap: 6px; }
  .ov-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11.5px; font-weight: 600; padding: 3px 10px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--panel-2); color: var(--text);
  }
  .ov-pill .n { color: var(--muted); font-variant-numeric: tabular-nums; }
  .ov-pill.t-frame  { border-color: #3b6db5; color: #9ec3ff; background: rgba(59,109,181,0.12); }
  .ov-pill.t-amber  { border-color: #ff7f0e; color: #ffc98a; background: rgba(255,127,14,0.12); }
  .ov-pill.t-red    { border-color: #d62728; color: #ff9b9b; background: rgba(214,39,40,0.12); }
  .ov-pill.t-purple { border-color: #9467bd; color: #c9adea; background: rgba(148,103,189,0.12); }

  /* DINO big stat */
  .big-stat { display: flex; align-items: baseline; gap: 12px; }
  .big-stat .num { font-size: 38px; font-weight: 800; font-variant-numeric: tabular-nums; line-height: 1; color: var(--text); }
  .big-stat .lbl { color: var(--muted); font-size: 13px; }
  .caption { color: var(--muted); font-size: 11px; margin-top: 9px; line-height: 1.5; }

  /* inventory table */
  table.inv { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  table.inv th, table.inv td { border: 1px solid var(--border); padding: 6px 10px; text-align: center; }
  table.inv th { color: var(--muted); font-weight: 600; background: var(--panel-2); }
  table.inv td.label { text-align: left; font-family: ui-monospace, Menlo, monospace; color: var(--text); }
  table.inv td.c-ok { color: var(--text); }
  table.inv td.c-miss { background: rgba(214,39,40,0.14); color: #ffb3b3; }
  table.inv td.c-extra { background: rgba(255,127,14,0.16); color: #ffce9b; }

  /* LLM card */
  .llm-summary { font-size: 16px; line-height: 1.6; color: var(--text); margin: 2px 0 11px; }
  .highlights { display: flex; flex-direction: column; gap: 6px; }
  .hl {
    display: flex; align-items: flex-start; gap: 10px; padding: 8px 11px;
    border-radius: 8px; border: 1px solid var(--border);
    opacity: 0; transform: translateY(6px);
    transition: opacity .12s ease-out, transform .12s ease-out;
  }
  .hl.in { opacity: 1; transform: none; }
  .hl.sev-alert   { background: rgba(214,39,40,0.06); }
  .hl.sev-warning { background: rgba(255,127,14,0.06); }
  .hl.sev-info    { background: rgba(127,127,127,0.04); }
  .sev-pill {
    flex: 0 0 auto; font-size: 10.5px; font-weight: 700; text-transform: lowercase;
    padding: 2px 9px; border-radius: 999px; color: #fff; margin-top: 1px;
  }
  .sev-pill.alert   { background: #d62728; }
  .sev-pill.warning { background: #ff7f0e; color: #1a1a1a; }
  .sev-pill.info    { background: #7f7f7f; }
  .hl .hl-text { font-size: 13.5px; line-height: 1.5; }
  .llm-footer { color: var(--muted); font-size: 11px; margin-top: 10px; }
  .llm-unavailable { font-style: italic; color: var(--muted); line-height: 1.6; }
</style>
</head>
<body>

<div class="ov-header">
  <span class="ov-title">Sequence Overview</span>
  <span class="spacer"></span>
  <div class="seq-switcher">
    <select id="sequence-select"></select>
    <span class="showing" id="showing"></span>
  </div>
</div>

<div class="overview-body" id="overview-body">
  <div class="overview-card" id="card-general">
    <div class="card-title">General information</div>
    <div class="card-body" id="body-general"></div>
  </div>
  <div class="overview-card" id="card-dino">
    <div class="card-title">DINO detection stats</div>
    <div class="card-body" id="body-dino"></div>
  </div>
  <div class="overview-card" id="card-inv">
    <div class="card-title">Unique object IDs per module</div>
    <div class="card-body" id="body-inv"></div>
  </div>
  <div class="overview-card" id="card-llm">
    <div class="card-title">Pipeline overview</div>
    <div class="card-body" id="body-llm"></div>
  </div>
</div>

<script>
const PAYLOAD = /*__DATA_JSON__*/null;
</script>
<script>
(function () {
  "use strict";
  const SEQS = PAYLOAD.sequences;
  const LLM_PREFERRED = PAYLOAD.llm_preferred_model || "the configured model";
  const $ = id => document.getElementById(id);
  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g,
      c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  const pct = x => (x * 100).toFixed(1) + "%";
  function mmss(s) {
    const m = Math.floor(s / 60), sec = Math.round(s % 60);
    return String(m).padStart(2, "0") + ":" + String(sec).padStart(2, "0");
  }
  const MOD_TITLE = { detection: "Detection", tracking: "Tracking", depth: "Depth",
                      scene_graph: "Scene graph", validation: "Validation" };
  const TRIGGER_CLASS = {
    init: "", frame_counter_K: "t-frame", tracker_low_confidence: "t-amber",
    bbox_size_change: "t-amber", drift: "t-red", depth_jump: "t-purple",
  };

  function renderGeneral(d) {
    const gt = d.gt, ov = d.overview;
    let kv = "";
    kv += `<div class="k">Sequence ID</div><div class="v">${escapeHtml(d.sequence_id)}</div>`;
    if (gt && gt.task_name)
      kv += `<div class="k">Task name</div><div class="v">${escapeHtml(gt.task_name)}</div>`;
    kv += `<div class="k">Duration</div><div class="v">${ov.duration_s.toFixed(2)} s `
        + `<span class="sub">(${ov.n_frames} frames at ${(+d.fps).toFixed(1)} fps)</span></div>`;
    if (gt) {
      if (gt.failure_reason)
        kv += `<div class="k">GT failure reason</div><div class="v reason">${escapeHtml(gt.failure_reason)}</div>`;
      if (gt.failure_window_frames && gt.failure_window_s) {
        const [s0, s1] = gt.failure_window_s, [f0, f1] = gt.failure_window_frames;
        kv += `<div class="k">GT failure window</div><div class="v">${mmss(s0)} – ${mmss(s1)} `
            + `<span class="sub">(frames ${f0} – ${f1})</span></div>`;
      }
    } else {
      kv += `<div class="k">Ground truth</div><div class="v muted-italic">No ground truth available</div>`;
    }
    const mods = ov.modules_available.map(m => `<span class="ov-pill">${escapeHtml(MOD_TITLE[m] || m)}</span>`).join("");
    $("body-general").innerHTML =
      `<div class="kv">${kv}</div>`
      + `<div class="subhead">Modules available</div><div class="pill-row">${mods}</div>`;
  }

  function renderDino(d) {
    const ds = d.overview.dino_stats;
    const reasons = ds.trigger_reasons || {};
    const order = ["init", "frame_counter_K", "tracker_low_confidence", "bbox_size_change", "drift", "depth_jump"];
    const keys = order.filter(r => r in reasons).concat(Object.keys(reasons).filter(r => !order.includes(r)));
    const pills = keys.map(r =>
      `<span class="ov-pill ${TRIGGER_CLASS[r] || ""}">${escapeHtml(r)} <span class="n">${reasons[r]}</span></span>`
    ).join("");
    $("body-dino").innerHTML =
      `<div class="big-stat"><span class="num">${pct(ds.recovery_rate)}</span>`
      + `<span class="lbl">DINO recovery rate</span></div>`
      + `<div class="subhead">Total DINO runs: <b style="color:var(--text)">${ds.n_runs}</b></div>`
      + `<div class="subhead" style="margin-top:4px">Runs by trigger reason</div>`
      + `<div class="pill-row">${pills}</div>`
      + `<div class="caption">Recovery rate excludes <code>frame_counter_K</code> — the routine `
      + `30-frame timeout that fires whether or not the tracker is struggling.</div>`;
  }

  function renderInventory(d) {
    const inv = d.overview.object_inventory || {};
    const mods = (d.overview.modules_available || []).filter(m => m in inv);
    const labels = new Set();
    mods.forEach(m => Object.keys(inv[m] || {}).forEach(l => labels.add(l)));
    const labelList = [...labels].sort();
    const head = `<tr><th style="text-align:left">Label</th>`
      + mods.map(m => `<th>${escapeHtml(MOD_TITLE[m] || m)}</th>`).join("") + `</tr>`;
    const rows = labelList.map(lbl => {
      const cells = mods.map(m => {
        const v = (inv[m] || {})[lbl] || 0;
        const cls = v === 0 ? "c-miss" : (v > 1 ? "c-extra" : "c-ok");
        return `<td class="${cls}">${v}</td>`;
      }).join("");
      return `<tr><td class="label">${escapeHtml(lbl)}</td>${cells}</tr>`;
    }).join("");
    const table = labelList.length
      ? `<table class="inv">${head}${rows}</table>`
      : `<div class="muted-italic">No object identities found.</div>`;
    $("body-inv").innerHTML = table
      + `<div class="caption">Counts represent unique tracker-emitted object_ids. Multiple IDs `
      + `for the same label indicate identity instability (the tracker re-IDed the same physical object).</div>`;
  }

  function renderLlm(d) {
    const llm = d.overview.llm_overview || {};
    const meta = llm._meta || {};
    const unavailable = !llm.summary || meta.model === "unavailable";
    if (unavailable) {
      $("body-llm").innerHTML =
        `<div class="llm-unavailable">LLM commentary unavailable. Run <code>ollama serve</code>, `
        + `ensure <code>${escapeHtml(LLM_PREFERRED)}</code> is installed, and rerun `
        + `<code>python "dashboard 2/run.py"</code>.</div>`;
      return;
    }
    const hls = (llm.highlights || []).map(h => {
      const sev = (h.severity === "alert" || h.severity === "warning") ? h.severity : "info";
      return `<div class="hl sev-${sev}"><span class="sev-pill ${sev}">${escapeHtml(sev)}</span>`
        + `<span class="hl-text">${escapeHtml(h.text)}</span></div>`;
    }).join("");
    const when = String(meta.generated_at || "").replace("Z", "");
    $("body-llm").innerHTML =
      `<div class="llm-summary">${escapeHtml(llm.summary)}</div>`
      + `<div class="highlights" id="highlights">${hls}</div>`
      + `<div class="llm-footer">Generated by ${escapeHtml(meta.model || "?")} · prompt `
      + `${escapeHtml(meta.prompt_version || "?")} · ${escapeHtml(when)} UTC</div>`;
    requestAnimationFrame(() => {
      document.querySelectorAll("#highlights .hl").forEach((el, i) =>
        setTimeout(() => el.classList.add("in"), 30 * i));
    });
  }

  function renderAll(sid) {
    const d = SEQS[sid];
    renderGeneral(d);
    renderDino(d);
    renderInventory(d);
    renderLlm(d);
    $("showing").innerHTML = "Currently showing: <b>" + escapeHtml(d.sequence_id) + "</b> · "
      + d.n_frames + " frames · " + (d.total_seconds || 0).toFixed(1) + " s";
    $("sequence-select").value = sid;
  }

  function switchSequence(sid) {
    if (!SEQS[sid]) return;
    const body = $("overview-body");
    body.classList.add("switching");
    setTimeout(() => { renderAll(sid); body.classList.remove("switching"); }, 160);
  }

  function init() {
    const sel = $("sequence-select");
    Object.keys(SEQS).forEach(sid => {
      const o = document.createElement("option");
      o.value = sid;
      const t = SEQS[sid].gt && SEQS[sid].gt.task_name;
      o.textContent = t ? (sid + " — " + t) : sid;
      sel.appendChild(o);
    });
    sel.addEventListener("change", () => switchSequence(sel.value));

    const defSid = SEQS[PAYLOAD.default_sequence] ? PAYLOAD.default_sequence : Object.keys(SEQS)[0];
    renderAll(defSid);
    ["card-general", "card-dino", "card-inv", "card-llm"].forEach((id, i) =>
      setTimeout(() => $(id).classList.add("in"), 50 * i));
  }

  init();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
