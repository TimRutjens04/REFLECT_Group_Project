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
import re
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

# Legacy explicit color overrides per object_id, kept as a reference / hardcode
# escape hatch. Sequence colors are now derived from labels via
# assign_sequence_colors(); this map is only consulted if an entry exists for a
# specific object_id you want to pin to a non-default color.
OBJECT_COLORS = {
    "red apple_1": "#d62728",
    "dark blue bowl_2": "#1f3a93",
}

# Color name -> hex. Keys MUST be lowercase. Multi-word keys (e.g. "dark blue")
# are matched longest-first so "dark blue" wins over "blue" for "dark blue bowl".
COLOR_NAME_HEX = {
    "red":         "#d62728",
    "dark red":    "#8b1a1a",
    "light red":   "#f08080",
    "blue":        "#1f77b4",
    "dark blue":   "#1f3a93",
    "light blue":  "#7fb3d5",
    "green":       "#2ca02c",
    "dark green":  "#1b5e20",
    "light green": "#90ee90",
    "yellow":      "#f0b400",
    "orange":      "#ff7f0e",
    "purple":      "#9467bd",
    "pink":        "#e377c2",
    "brown":       "#8c564b",
    "black":       "#222222",
    "white":       "#dddddd",
    "gray":        "#7f7f7f",
    "grey":        "#7f7f7f",
}

# Fallback palette when an object's natural color is already taken (or the label
# has no recognizable color word). Picked from matplotlib tab10/tab20 leftovers
# so it visually meshes with the natural colors above.
FALLBACK_PALETTE = [
    "#17becf",  # teal
    "#bcbd22",  # olive
    "#aec7e8",  # pale blue
    "#ffbb78",  # pale orange
    "#98df8a",  # pale green
    "#ff9896",  # salmon
    "#c5b0d5",  # lavender
    "#c49c94",  # tan
    "#dbdb8d",  # mustard
    "#9edae5",  # pale cyan
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
METRICS_OUT = DASH_DIR / "metrics.html"
VIDEO_OUT_DIR = DASH_DIR / "video"

# ---- LLM (Ollama) configuration for the overview-page commentary ----------- #
OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.1:8b"
OLLAMA_FALLBACK = "llama3:latest"   # used if the preferred model isn't installed
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT_S   = 180
PROMPT_VERSION  = "overview_v2.5"
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


# Physically implausible scene-graph relations. Each rule is
# (from_label_substring, relation, to_label_substring); the sentinel "*fruit*"
# matches any fruit label. Used to flag scene-graph misclassifications in the
# LLM input so the model treats them as errors, not real task state.
_FRUITS = ("apple", "pear", "banana", "orange", "lemon", "peach", "mango", "grape", "fruit")
# Only relations the scene graph actually emits (near, left_of, above,
# on_top_of, inside) can ever fire here. Rules referencing relations the SG never
# produces (e.g. held_by_gripper) are dead code and were removed — they only led
# the LLM to hallucinate matching findings.
_IMPLAUSIBLE_RULES = [
    ("coffee machine", "inside",    "table"),
    ("coffee cup",     "inside",    "table"),
    ("drawer",         "inside",    "*fruit*"),
    ("bowl",           "inside",    "*fruit*"),
    ("fridge",         "inside",    "*fruit*"),
    ("pot",            "inside",    "stove burner"),
    ("fridge",         "inside",    "pot"),
    ("coffee machine", "on_top_of", "coffee cup"),
]


def _label_matches(label: str, pat: str) -> bool:
    if pat == "*fruit*":
        return any(f in label for f in _FRUITS)
    return pat in label


def is_implausible_relation(from_label: str, relation: str, to_label: str) -> bool:
    for fpat, rel, tpat in _IMPLAUSIBLE_RULES:
        if relation == rel and _label_matches(from_label, fpat) and _label_matches(to_label, tpat):
            return True
    return False


def _presence_flicker(frames) -> int:
    """Number of disappear→reappear cycles: count gaps (>1 frame) in a presence set."""
    fs = sorted(frames)
    return sum(1 for a, b in zip(fs, fs[1:]) if b - a > 1)


def _contiguous_ranges(frames, max_gap: int = 2) -> list:
    """[1,2,3, 10,11,12] -> [[1,3],[10,12]]. Gaps <= max_gap join one range."""
    if not frames:
        return []
    s = sorted(frames)
    out = [[s[0], s[0]]]
    for f in s[1:]:
        if f - out[-1][1] <= max_gap + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def _flag_bursts(fired_frames, gap: int = 4, min_span: int = 5) -> list:
    """Contiguous ranges where a flag fires on >half the frames. Firing frames
    within `gap` of each other join one cluster; clusters shorter than `min_span`
    (single-event firings) are dropped."""
    fs = sorted(set(fired_frames))
    if not fs:
        return []
    clusters = []
    start = prev = fs[0]
    cnt = 1
    for f in fs[1:]:
        if f - prev <= gap:
            prev = f
            cnt += 1
        else:
            clusters.append((start, prev, cnt))
            start = prev = f
            cnt = 1
    clusters.append((start, prev, cnt))
    out = []
    for s, e, c in clusters:
        span = e - s + 1
        if span >= min_span and c / span > 0.5:
            out.append([s, e])
    return out


def extract_color_word(label: str) -> str | None:
    """Return the recognized color word from a label, or None. Longest-key-first,
    lowercase, word-boundary aware. "dark blue bowl"->"dark blue", "purple cup"->
    "purple", "coffee machine"->None."""
    if not label:
        return None
    l = label.lower().strip()
    for key in sorted(COLOR_NAME_HEX.keys(), key=len, reverse=True):
        if l == key or l.startswith(key + " ") or l.endswith(" " + key) or \
           (" " + key + " ") in l:
            return key
    return None


def assign_sequence_colors(objects: list[dict]) -> dict:
    """Assign a chart color to every object_id in the sequence.

    Rules: same label -> same color; different labels sharing a natural color ->
    the longer-lived (more n_frames) claims it, others fall back; labels with no
    color word -> next FALLBACK_PALETTE slot; explicit OBJECT_COLORS pins win.
    Deterministic (no randomness) so builds are reproducible.
    """
    pinned = {oid: OBJECT_COLORS[oid] for o in objects
              if (oid := o.get("object_id")) and oid in OBJECT_COLORS}

    # Group by label; same label is one shared assignment.
    labels_in_order: list[str] = []
    label_frames: dict[str, int] = {}
    label_objects: dict[str, list[str]] = {}
    for o in objects:
        lbl = o.get("label") or ""
        oid = o.get("object_id") or ""
        nfr = int(o.get("n_frames") or 0)
        if lbl not in label_frames:
            labels_in_order.append(lbl)
            label_frames[lbl] = 0
            label_objects[lbl] = []
        label_frames[lbl] += nfr
        label_objects[lbl].append(oid)

    # Most-tracked label wins ties; discovery order breaks remaining ties.
    sorted_labels = sorted(
        labels_in_order,
        key=lambda l: (-label_frames[l], labels_in_order.index(l)),
    )

    taken: set[str] = set()
    label_color: dict[str, str] = {}

    def next_fallback() -> str:
        for c in FALLBACK_PALETTE:
            if c not in taken:
                return c
        return FALLBACK_PALETTE[len(taken) % len(FALLBACK_PALETTE)]

    for lbl in sorted_labels:
        # A pin on any of this label's objects takes priority over the natural
        # color word (and keeps same-label objects on the same color).
        pin = next((pinned[oid] for oid in label_objects[lbl] if oid in pinned), None)
        desired = pin or COLOR_NAME_HEX.get(extract_color_word(lbl) or "")
        chosen = desired if (desired and desired not in taken) else next_fallback()
        label_color[lbl] = chosen
        taken.add(chosen)

    result = {oid: label_color[lbl]
              for lbl, oids in label_objects.items() for oid in oids}
    result.update(pinned)   # exact per-object_id pins always win
    return result


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
    meaningful_triggers = (n_runs - reasons.get("frame_counter_K", 0)
                           - reasons.get("init", 0))
    recovery_rate = (round(meaningful_triggers / n_frames, 4)
                     if n_frames else 0.0)
    dino_stats = {
        "n_runs": n_runs,
        "trigger_reasons": dict(reasons),
        "meaningful_triggers": meaningful_triggers,
        "recovery_rate": recovery_rate,
        "recovery_rate_note": (
            "fraction of total frames where DINO ran due to a genuine recovery "
            "trigger (i.e. excluding routine frame_counter_K timeouts and the "
            "init detection); answers 'how often did the tracker need to recover.'"
        ),
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

    # ---- LLM input summary (GT-FREE; all floats rounded for a stable hash) ----
    # This dict is the ONLY thing the LLM sees. It deliberately contains no ground
    # truth — no failure_reason, no failure window, no success label — so the
    # commentary describes pipeline signals only. GT still reaches the page via the
    # separate top-level `gt` field (built in build_data and shown on the card).

    def conf_stats(oid):
        pts = tracker_conf.get(oid, [])
        if not pts:
            return None
        vals = sorted(p["y"] for p in pts)
        n = len(vals)
        lowest_5_frames = sorted(p["x"] for p in sorted(pts, key=lambda q: q["y"])[:5])
        return {
            "min": round(vals[0], 3),
            "p5": round(_percentile(vals, 5), 3),
            "mean": round(sum(vals) / n, 3),
            "p95": round(_percentile(vals, 95), 3),
            "max": round(vals[-1], 3),
            "frames_below_0_4": sum(1 for v in vals if v < 0.4),
            "frames_below_0_6": sum(1 for v in vals if v < 0.6),
            "lowest_5_frames": lowest_5_frames,
        }

    tracker_conf_stats = {}
    for oid in main_objs:
        s = conf_stats(oid)
        if s:
            tracker_conf_stats[oid] = s

    # Per-object status counts come from the VALIDATION module (tracking's
    # top-level status underreports drift/occlusion).
    status_keys = ["ok", "drifting", "occluded", "lost", "recovered"]
    val_status: dict[str, Counter] = {}
    for r in validation:
        for o in r.get("tracked_objects", []):
            oid = o.get("object_id")
            if oid:
                val_status.setdefault(oid, Counter())[obj_status(o) or "ok"] += 1
    tracker_status_summary = {
        oid: {k: c.get(k, 0) for k in status_keys} for oid, c in val_status.items()
    }

    # Contiguous flag bursts (flag fires on most frames of the range).
    drift_fired = [r["frame_id"] for r in tracking if r.get("flags", {}).get("drift_flag")]
    recov_fired = [r["frame_id"] for r in tracking if r.get("flags", {}).get("any_recovery_trigger")]
    depthjump_fired = [r["frame_id"] for r in depth
                       if any(o.get("depth_jump_flag") for o in r.get("per_object_depth", []))]
    flag_burst_ranges = {
        "drift_flag": _flag_bursts(drift_fired),
        "depth_jump_flag": _flag_bursts(depthjump_fired),
        "any_recovery_trigger": _flag_bursts(recov_fired),
    }

    # Scene-graph summary: relation counts, transitions, implausible relations.
    # Implausible edges are aggregated per specific (from_id, label, relation,
    # to_id, label) so the LLM can cite exact ids and frames, not just labels.
    relation_counts: Counter = Counter()
    imp_frames: dict[tuple, set] = {}
    for r in scene_graph:
        fid = r["frame_id"]
        nodes_by_id = {n.get("object_id"): n.get("label") for n in r.get("nodes", [])}
        for e in r.get("edges", []):
            rel = e.get("relation")
            if not rel:
                continue
            relation_counts[rel] += 1
            a, b = edge_endpoints(e)
            if not a or not b:
                continue
            la = nodes_by_id.get(a) or a.rsplit("_", 1)[0]
            lb = nodes_by_id.get(b) or b.rsplit("_", 1)[0]
            if is_implausible_relation(la.lower(), rel, lb.lower()):
                imp_frames.setdefault((a, la, rel, b, lb), set()).add(fid)
    trans_counts: Counter = Counter()
    for i in range(len(relation_strip) - 1):
        fr, to = relation_strip[i]["relation"], relation_strip[i + 1]["relation"]
        if fr and to:
            trans_counts[(fr, to)] += 1

    implausible_observed = []
    for key, fset in sorted(imp_frames.items(), key=lambda kv: -len(kv[1])):
        fids = sorted(fset)
        ranges = _contiguous_ranges(fids, max_gap=2)
        ranges_show = sorted(ranges, key=lambda rg: -(rg[1] - rg[0]))[:6]
        entry = {
            "from_id": key[0],
            "from_label": key[1],
            "relation": key[2],
            "to_id": key[3],
            "to_label": key[4],
            "n_frames": len(fids),
            "first_frame": fids[0],
            "last_frame": fids[-1],
            "frame_ranges": ranges_show,
            "example_frames": fids[:5],
        }
        if len(ranges) > len(ranges_show):
            entry["n_extra_ranges"] = len(ranges) - len(ranges_show)
        implausible_observed.append(entry)

    scene_graph_summary = {
        "n_unique_relations": sorted(relation_counts.keys()),
        "relation_counts": dict(relation_counts),
        "relation_transitions": [
            {"from": k[0], "to": k[1], "count": v}
            for k, v in sorted(trans_counts.items(), key=lambda kv: -kv[1])
        ],
        "implausible_relations_observed": implausible_observed,
    }
    # Make the ABSENCE of implausible relations explicit, and tell the LLM which
    # relation names actually exist in this sequence so it can't invent others.
    scene_graph_summary["n_implausible_relations_observed"] = len(implausible_observed)
    scene_graph_summary["relation_types_emitted_by_sg"] = sorted({
        e.get("relation") for r in scene_graph for e in r.get("edges", []) if e.get("relation")
    })

    # Flicker: object presence gaps (tracking) and main-pair relation gaps.
    track_presence: dict[str, set] = {}
    for r in tracking:
        fid = r["frame_id"]
        for o in r.get("tracked_objects", []):
            oid = o.get("object_id")
            if oid:
                track_presence.setdefault(oid, set()).add(fid)
    object_flicker = {oid: _presence_flicker(fr) for oid, fr in track_presence.items()}
    main_rel_frames: dict[str, set] = {}
    for r in scene_graph:
        fid = r["frame_id"]
        for e in r.get("edges", []):
            a, b = edge_endpoints(e)
            rel = e.get("relation")
            if rel and a in main_set and b in main_set and {a, b} == main_set:
                main_rel_frames.setdefault(rel, set()).add(fid)
    relation_flicker = {}
    if len(main_objs) >= 2:
        oa, ob = main_objs[0], main_objs[1]
        for rel, fr in main_rel_frames.items():
            relation_flicker[f"{oa}|{ob}|{rel}"] = _presence_flicker(fr)
    flicker_summary = {"objects": object_flicker, "relations": relation_flicker}

    # Cross-module mismatches over the tracker-namespace modules only (detection
    # uses its own detection-id namespace, so comparing it here would be noise).
    ns_mods = {m: _ids_set(rows_by_mod[m], extractors[m])
               for m in ("tracking", "depth", "scene_graph", "validation") if rows_by_mod[m]}
    all_ns_ids = set().union(*ns_mods.values()) if ns_mods else set()
    mismatches = []
    for oid in sorted(all_ns_ids):
        missing = [m for m in ns_mods if oid not in ns_mods[m]]
        if missing:
            mismatches.append({
                "oid": oid,
                "in_modules": [m for m in ns_mods if oid in ns_mods[m]],
                "missing_from": missing,
            })

    blackout = [f for f in range(n_frames) if f not in bbox_tracker]
    bo_ranges: list[list[int]] = []
    for f in blackout:
        if bo_ranges and f == bo_ranges[-1][1] + 1:
            bo_ranges[-1][1] = f
        else:
            bo_ranges.append([f, f])
    longest_burst = max(bo_ranges, key=lambda r: r[1] - r[0], default=None)

    llm_dino = {
        "n_runs": n_runs,
        "trigger_reasons": dict(reasons),
        "meaningful_recovery_rate": recovery_rate,
        "n_routine_timeouts": reasons.get("frame_counter_K", 0),
    }
    llm_inventory = {m: object_inventory[m]
                     for m in ("detection", "tracking", "scene_graph", "validation")
                     if m in object_inventory}

    llm_input = {
        "sequence_id": seq_id,
        "task_name": gt.get("task_name") if gt else None,   # task description only — no failure info
        "n_frames": n_frames,
        "fps": fps,
        "duration_s": total_seconds,
        "dino": llm_dino,
        "object_inventory": llm_inventory,
        "tracker_confidence_stats": tracker_conf_stats,
        "tracker_status_summary": tracker_status_summary,
        "flag_burst_ranges": flag_burst_ranges,
        "scene_graph_summary": scene_graph_summary,
        "flicker_summary": flicker_summary,
        "cross_module_mismatches": mismatches,
        "blackout_summary": {
            "total_empty_frames": len(blackout),
            "longest_burst_frames": longest_burst if longest_burst else [],
        },
    }

    # ---- Object-presence ranges for the timeline Gantt (honest segments) -----
    # Reuse track_presence (oid -> set of frames). max_gap=1 keeps presence
    # exact-ish (bridges only single dropped frames). Ordered by n_frames desc so
    # the front-end can take the first 6 keys without re-sorting.
    object_presence_ranges = {}
    for oid, fset in sorted(track_presence.items(), key=lambda kv: -len(kv[1])):
        fids = sorted(fset)
        object_presence_ranges[oid] = {
            "label": oid.rsplit("_", 1)[0],
            "n_frames": len(fids),
            "ranges": _contiguous_ranges(fids, max_gap=1),
        }

    # ---- Per-object colors derived from label color-words (no AI) -------------
    # Covers EVERY tracked object_id (charts, Gantt, chips, bbox overlay all read
    # from this single map). Same label -> same color; collisions resolved by
    # n_frames; OBJECT_COLORS pins win. Deterministic.
    object_colors = assign_sequence_colors([
        {"object_id": oid, "label": v["label"], "n_frames": v["n_frames"]}
        for oid, v in object_presence_ranges.items()
    ])
    for oid in per_object:
        if oid in object_colors:
            per_object[oid]["color"] = object_colors[oid]

    # ---- Per-frame scene-graph edges for the live "Relations" chip row -------
    edges_by_frame: dict[int, list] = {}
    for r in scene_graph:
        fid = r["frame_id"]
        items = []
        for e in r.get("edges", []):
            a, b = edge_endpoints(e)
            rel = e.get("relation")
            if a and b and rel:
                items.append({"from_id": a, "relation": rel, "to_id": b})
        if items:
            edges_by_frame[fid] = items

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
        "object_presence_ranges": object_presence_ranges,
        "object_colors": object_colors,
        "edges_by_frame": edges_by_frame,
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
## Role

You are reviewing a single robot perception sequence. The pipeline runs Grounding DINO
for object detection, a single-object tracker propagating between detections, a depth
consistency module, a scene-graph builder, and a validation module. Your job is to
produce a concise overview describing what the dashboard observed in this sequence.

IMPORTANT: The dashboard's purpose is to DETECT potential pipeline or task failures
on its own, from the signals available in the pipeline logs. You will not be given
ground-truth task-failure labels. Your commentary must describe what the pipeline
signals indicate, not what "actually happened" externally. Do not claim a task
succeeded or failed. Describe what the data shows; let the reader judge.

## Glossary

- tracker_confidence: 0 to 1 score per (frame, object); lower values mean less
  reliable tracking.
- tracker_status: ok | drifting | occluded | lost | recovered. The validation
  module reports per-object status that can disagree with the tracker's own
  top-level reporting.
- bbox_size_change_flag: fires when the tracked box area changes sharply versus
  its initialization size — sometimes a sign that the tracker locked onto a
  different region.
- drift_flag: fires when pixel displacement exceeds a threshold over consecutive
  frames.
- recovery_trigger: the tracker asked the pipeline to re-run detection.
- frame_counter_K_flag: a ROUTINE 30-frame timeout that fires regardless of
  tracker state. It is NOT a sign of trouble.
- depth_jump_flag, depth_coherence_flag, depth_validity_flag, any_depth_trigger:
  per-frame, per-object depth-module signals.
- Scene graph: builds a per-frame graph of which objects are near/inside/on top
  of/left of/above each other. The relations are derived from 3D positions and
  bbox geometry.
- Cross-module mismatch: an object_id present in one module's log but missing
  from another's.
- Flicker: an object or relation that disappears and reappears across frames.

## Domain knowledge — apply this when forming highlights

### Grounding DINO cadence

DINO runs automatically every 30 frames via the routine timeout
(frame_counter_K_flag). These routine calls do NOT indicate that the tracker was
struggling. The meaningful re-detections are those triggered by
tracker_low_confidence, bbox_size_change, drift, or depth_jump.

The input field `dino_stats.recovery_rate` is the **fraction of total frames
where DINO ran due to a genuine recovery trigger** (i.e. excluding the routine
frame_counter_K timeouts and the init detection). It answers "how often did the
tracker need to recover?" — a small percentage (e.g. 5-10%) is typical for a
healthy run; a much higher value signals frequent instability. Interpret the
metric per-frame, not per-DINO-call.

### Tracker confidence drops

A confidence drop is NOT automatic evidence of pipeline failure. Common benign
causes in this dataset include:
- The object is moving and the robot's gripper is interacting with it.
- The object is partially or fully occluded by the robot's hand.

Use cautious language. Say "this could indicate" or "this might suggest occlusion
or interaction" rather than "the tracker failed."

### Scene-graph behavior

The scene graph is the only module that can structurally signal what is happening
in the task. For example, in a "put apple in bowl" task, edges like
"apple on_top_of bowl" or "apple inside bowl" indicate the placement state.

HOWEVER, the scene graph can produce implausible relations. Treat the following
patterns as scene-graph errors, not as real task state. Flag them in the
commentary as "the scene graph reports an implausible relation" rather than as
a real event:

- Coffee Machine is inside the table.
- Coffee cup is inside the table.
- Drawer is inside any fruit.
- Bowl is inside any fruit.
- Fridge is inside any fruit.
- Pot is inside the stove burner.
- Fridge is inside the pot.
- Coffee machine is on top of the coffee cup.

These are physically implausible and indicate a scene-graph misclassification.

HARD RULE — DO NOT FABRICATE IMPLAUSIBLE-RELATION FINDINGS.

You will ONLY raise an implausible-relation highlight if the input's
`scene_graph_summary.implausible_relations_observed` list contains a matching
entry for that exact (from_id, relation, to_id) combination. The build script
has already pre-checked every scene-graph edge against the implausible-pattern
list above; if a relation is not in `implausible_relations_observed`, then by
definition it is NOT implausible in this sequence, regardless of how the
labels sound to you.

If `scene_graph_summary.n_implausible_relations_observed` is 0, do not produce
ANY implausible-relation highlight, do not paraphrase the implausible-pattern
list, and do not flag any relation as a scene-graph error. The patterns above
exist so you know what to call out IF the input contains them; they are not a
list of findings to manufacture.

The scene graph emits these spatial relation types: near, left_of, above,
on_top_of, inside. It may also emit `held_by_gripper` (the robot gripper
holding an object) — this is a NORMAL relation and is NEVER implausible by
itself; the robot routinely holds the object it is manipulating. Use only
relation names that appear in the input's `relation_types_emitted_by_sg`
list; do not invent others.

Normal relations like `near`, `left_of`, `above`, and `held_by_gripper`
between sensibly-paired objects are NOT implausible. Only the specific
patterns listed above count as implausible, and only when present in the
input's implausible_relations_observed list.

When you raise a highlight about an implausible scene-graph relation, you MUST
cite all of the following from the input data:
- The exact relation type ("inside", "on_top_of", "above", "near", "left_of").
- The specific object_ids involved (use `from_id` and `to_id` from the input,
  not just labels — IDs are more precise and tell the reader which instance
  was affected).
- A specific frame reference: either a single frame range like "frames 100-105"
  taken from `frame_ranges`, or "frames 100, 102, 310" taken from
  `example_frames`, or the inclusive interval `first_frame`-`last_frame`.
- The total `n_frames` the relation persisted for.

Generic phrasing like "at certain points" or "between the drawer and the
gripper" is not acceptable when the input contains the specific values.

Use this template for implausible-relation highlights:
"The scene graph reports {from_id} as '{relation}' {to_id} across frames
{range_text} ({n_frames} frames total). This is physically implausible and
likely indicates a scene-graph misclassification rather than a real event."

If multiple implausible relations are present, raise each as its own
highlight rather than rolling them into one. Severity for these is "alert".

### Flicker thresholds

- 2-3 flickers of an object or relation across the sequence is usually
  occlusion (robot hand briefly blocks the view). Generally benign — do not
  raise to alert severity.
- MORE than 3 flickers of a single object or relation is worth commenting on
  as possible instability — phrase it as "this could indicate" rather than
  asserting a failure.

### Tone

Use cautious, exploratory language throughout: "this could suggest," "this
might indicate," "worth investigating," "the data shows." Avoid definitive
verdicts: "the task failed," "the tracker is broken," "the pipeline did not
work." The dashboard surfaces possible issues; it does not declare verdicts.

## Reference frames — calibration for healthy vs concerning state

When evaluating per-object statistics in the input, use these two reference
states as calibration points.

### Healthy state (what stable mid-task tracking looks like)

- Multiple objects tracked simultaneously, each with tracker_confidence well
  above 0.9.
- All tracker_status values are "ok"; no flags fired this frame.
- Depth medians per object are stable across recent frames (small frame-to-frame
  change).
- Scene graph has an edge between the relevant objects (e.g., "near"), and that
  edge has been consistent for many frames.
- bbox_area_ratio_to_init is close to 1.0 for each tracked object.

### Concerning state (what degradation looks like)

- Only one object tracked when two or more were tracked moments ago — the
  pipeline lost an object.
- tracker_confidence has dropped below 0.4 for at least one object.
- tracker_status is "drifting" while bbox_size_change_flag and recovery_trigger
  are both true.
- bbox_area_ratio_to_init has dropped well below 0.8 (the bbox shrank
  significantly).
- Depth median has shifted sharply (more than ~0.2m) from its recent value.
- Scene graph has lost the relation between the relevant objects, or no edges
  exist at all.

You will not see raw per-frame data — you will see aggregated statistics. Use
the reference states as the standard for interpreting whether the aggregates
look healthy or concerning.

## Output schema

Output ONLY valid JSON in this exact shape. No text outside the JSON.

{
  "summary": "2-3 sentence headline overview, plain English, cautious tone.",
  "highlights": [
    {
      "text": "Specific finding referencing concrete values or counts from the input.",
      "severity": "info" | "warning" | "alert"
    }
  ]
}

Maximum 6 highlights. Prefer 4-5 substantive ones over a long list of generic
ones. Every highlight must reference at least one of: a specific object_id, a
specific numeric value from the input, or a specific frame_id (or frame
range). Each cited value must appear LITERALLY in the input data — do not
infer, paraphrase, or round values that aren't present. For
implausible-relation highlights, all of these are required: object_ids,
relation type from the five allowed names, and frame range — AND the entry
must exist in `scene_graph_summary.implausible_relations_observed`.

If you cannot find a value in the input that supports a claim, do not include
the highlight. It is better to produce fewer highlights than to invent
specifics.

## Severity rules

Severity is based ENTIRELY on what the pipeline signals indicate. Do not
reference any GT, failure-window, or task-success information. You will not
receive any such information in the input.

- "alert" = a clear pipeline anomaly, including:
    * a tracked object's minimum confidence below 0.4;
    * a sustained drifting/occluded/lost status reported by validation;
    * an implausible scene-graph relation from the list above;
    * a cross-module identity mismatch (object_id in one module, missing from
      another);
    * extended periods (more than ~15 frames) where the pipeline emitted zero
      tracked objects.
- "warning" = unusual but possibly normal, including:
    * minimum confidence between 0.4 and 0.6 (could indicate occlusion);
    * 4-10 flickers of an object or relation;
    * depth_jump_flag firing more than a handful of times;
    * meaningful bbox_size_change events.
- "info" = stable, descriptive observations the reader should know but that
  do not indicate any concern.
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
    # Wrap the data in an explicit instruction. Without this, weaker models (e.g.
    # the llama3:latest fallback) tend to just echo the input JSON back verbatim
    # under format:"json" instead of producing the {summary, highlights} review.
    user_msg = (
        "Analyze the following robot-perception pipeline summary and write your review.\n"
        "Respond with ONLY a JSON object of this exact shape:\n"
        '{"summary": "2-3 sentence overview", '
        '"highlights": [{"text": "...", "severity": "info|warning|alert"}]}\n'
        "Do NOT repeat, echo, or copy the input fields shown below; produce your OWN "
        "analysis of them.\n\n"
        "PIPELINE_SUMMARY_INPUT:\n" + json.dumps(summary, indent=2)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
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

    # Deterministic backstop: the build already pre-checked every edge, so if the
    # input flagged ZERO implausible relations, any highlight that nonetheless
    # asserts implausibility is a hallucination — drop it. (When real implausible
    # relations exist, highlights are kept and the prompt template governs them.)
    imp = (summary.get("scene_graph_summary") or {}).get("implausible_relations_observed") or []
    if not imp:
        fabricated = re.compile(r"implausib|misclassif|physically impossible|scene[- ]?graph error", re.I)
        kept = [h for h in highlights if not fabricated.search(h["text"])]
        if len(kept) != len(highlights):
            print(f"      (dropped {len(highlights) - len(kept)} fabricated "
                  f"implausible-relation highlight(s) — input flagged none)")
        highlights = kept

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
    t0 = datetime.now()
    result = _call_ollama(model, summary)
    elapsed = (datetime.now() - t0).total_seconds()
    if result is None:
        print(f"      → call failed after {elapsed:.1f}s; rendering 'unavailable' for {seq_id}")
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
    print(f"      → generated in {elapsed:.1f}s; wrote to cache ({cache_path.name})")
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
def _slice(s: str, start: str, end: str) -> str:
    return s.split(start, 1)[1].split(end, 1)[0]


def _extract_index_parts() -> tuple[str, str, str]:
    """(css, content-markup, iife-js) from the single-page timeline template.
    content-markup is everything after </header> up to the first <script>; the
    IIFE is the second <script> (the first only declares PAYLOAD)."""
    css = _slice(HTML_TEMPLATE, "<style>", "</style>")
    after_head = HTML_TEMPLATE.split("</head>", 1)[1]
    content = after_head.split("</header>", 1)[1].split("<script>", 1)[0]
    js = (HTML_TEMPLATE.split("const PAYLOAD = /*__DATA_JSON__*/null;", 1)[1]
          .split("<script>", 1)[1].rsplit("</script>", 1)[0])
    return css, content, js


def _extract_overview_parts() -> tuple[str, str, str]:
    """(css, cards-markup, iife-js) from the overview template; the ov-header
    (its own dropdown) is dropped — only the four cards are kept."""
    css = _slice(OVERVIEW_TEMPLATE, "<style>", "</style>")
    cards = ('<div class="overview-body"'
             + OVERVIEW_TEMPLATE.split('<div class="overview-body"', 1)[1].split("<script>", 1)[0])
    js = (OVERVIEW_TEMPLATE.split("const PAYLOAD = /*__DATA_JSON__*/null;", 1)[1]
          .split("<script>", 1)[1].rsplit("</script>", 1)[0])
    return css, cards, js


def build_merged_template() -> str:
    """Assemble ONE tabbed page from the two templates: shared CSS (index +
    overview + chrome), one global header, Overview/Timeline panels, and the two
    IIFEs plus a tab coordinator. The /*__DATA_JSON__*/null token appears once."""
    idx_css, idx_content, idx_js = _extract_index_parts()
    ov_css, ov_cards, ov_js = _extract_overview_parts()
    return (
        MERGED_HEAD
        + idx_css + "\n" + ov_css + "\n" + MERGE_CSS
        + "</style>\n</head>\n<body class=\"tab-overview\">\n"
        + GLOBAL_HEADER
        + '<div class="tab-panels">\n'
        + '<div class="tab-panel" id="overview-panel">\n' + ov_cards + "</div>\n"
        + '<div class="tab-panel" id="timeline-panel">\n' + idx_content + "</div>\n"
        + "</div>\n"
        + "<script>\nconst PAYLOAD = /*__DATA_JSON__*/null;\n</script>\n"
        + "<script>\n" + idx_js + "\n</script>\n"
        + "<script>\n" + ov_js + "\n</script>\n"
        + "<script>\n" + COORDINATOR_JS + "\n</script>\n"
        + "</body>\n</html>\n"
    )


def render_merged(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return build_merged_template().replace("/*__DATA_JSON__*/null", payload_json)


# --------------------------------------------------------------------------- #
# LLM evaluation metrics (no human annotation — output vs. structured input)
# --------------------------------------------------------------------------- #
_NUMBER_PAT = re.compile(r"(?<![A-Za-z_])(\d+(?:\.\d+)?)(?![A-Za-z_])")
_FRAME_PAT = re.compile(r"\bframes?\s+(\d+)(?:\s*[-–]\s*(\d+))?", re.IGNORECASE)


def check_schema(output: dict) -> dict:
    """Schema validity: is the LLM output well-formed?
    Returns {"passed": bool, "issues": [str, ...]}."""
    issues = []
    if not isinstance(output, dict):
        return {"passed": False, "issues": ["output is not a dict"]}
    if "summary" not in output or not isinstance(output["summary"], str):
        issues.append("missing or non-string 'summary'")
    hl = output.get("highlights")
    if not isinstance(hl, list):
        issues.append("'highlights' is not a list")
    else:
        for i, h in enumerate(hl):
            if not isinstance(h, dict):
                issues.append(f"highlight {i} is not a dict")
                continue
            if "text" not in h or not isinstance(h.get("text"), str):
                issues.append(f"highlight {i} missing 'text'")
            if h.get("severity") not in {"info", "warning", "alert"}:
                issues.append(f"highlight {i} has invalid severity")
    return {"passed": not issues, "issues": issues}


def check_references(summary_input: dict, output: dict) -> dict:
    """Reference validity: every cited frame_id and object_id exists in input.
    Returns {"total", "valid", "rate", "invalid_examples"}."""
    n_frames = int(summary_input.get("n_frames", 0))
    valid_oids = set()
    for oid in (summary_input.get("tracker_confidence_stats") or {}):
        valid_oids.add(oid)
    for oid in (summary_input.get("tracker_status_summary") or {}):
        valid_oids.add(oid)
    for m in (summary_input.get("cross_module_mismatches") or []):
        if isinstance(m, dict) and m.get("oid"):
            valid_oids.add(m["oid"])

    total = valid = 0
    bad = []
    for h in (output.get("highlights") or []):
        text = h.get("text") or ""
        for m in _FRAME_PAT.finditer(text):
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            for f in (a, b):
                total += 1
                if 0 <= f < n_frames:
                    valid += 1
                else:
                    bad.append(f"frame {f} outside [0,{n_frames - 1}]")
        for oid in valid_oids:
            if oid in text:
                total += 1
                valid += 1
        valid_lower = {oid.lower() for oid in valid_oids}
        for token in re.findall(r"\b[a-z][a-z _]*_\d+\b", text.lower()):
            if token not in valid_lower:
                total += 1
                bad.append(f"object_id '{token}' not in input")

    rate = (valid / total) if total else 1.0
    return {"total": total, "valid": valid, "rate": round(rate, 4),
            "invalid_examples": bad[:6]}


def check_grounding(summary_input: dict, output: dict) -> dict:
    """Numeric grounding: every cited number appears in the input data.
    Returns {"total", "grounded", "rate", "ungrounded_examples"}."""
    input_text = json.dumps(summary_input, sort_keys=True)

    def appears(num_str: str) -> bool:
        if num_str in input_text:
            return True
        try:
            v = float(num_str)
        except ValueError:
            return False
        for prec in (3, 2, 1, 0):
            r = round(v, prec)
            if f"{r}" in input_text or f"{r:.{prec}f}" in input_text:
                return True
        return False

    total = grounded = 0
    bad = []
    for h in (output.get("highlights") or []):
        text = h.get("text") or ""
        text_without_frames = _FRAME_PAT.sub("", text)
        for m in _NUMBER_PAT.finditer(text_without_frames):
            num = m.group(1)
            try:
                if float(num) < 5 and "." not in num:
                    continue
            except ValueError:
                continue
            total += 1
            if appears(num):
                grounded += 1
            else:
                bad.append(num)

    rate = (grounded / total) if total else 1.0
    return {"total": total, "grounded": grounded, "rate": round(rate, 4),
            "ungrounded_examples": bad[:6]}


def _metric_tier(rate) -> str:
    if rate is None:
        return "na"
    if rate >= 0.95:
        return "good"
    if rate >= 0.80:
        return "warn"
    return "bad"


def _mesc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


_METRICS_CSS = """
  /* the reused overview-card CSS starts at opacity:0 (it fades in via JS on the
     dashboard); this static page has no such JS, so force the cards visible. */
  .metrics-card { opacity: 1; transform: none; }
  .metrics-wrap { display: flex; flex-direction: column; gap: 1rem; max-width: 1080px; margin: 0 auto; }
  .metrics-header h1 { font-size: 20px; font-weight: 800; margin: 2px 0 5px; }
  .metrics-header .cap { color: var(--muted); font-size: 12.5px; line-height: 1.55; max-width: 900px; }
  .card-explainer { color: var(--muted); font-size: 12.5px; line-height: 1.6; margin: 4px 0 14px; }
  .metrics-table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin-top: 8px; }
  .metrics-table th, .metrics-table td { border: 1px solid var(--border); padding: 6px 10px; text-align: left; vertical-align: top; }
  .metrics-table th { color: var(--muted); font-weight: 600; background: var(--panel-2); }
  .metrics-table td.num-col { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .metrics-table .ok { color: #2ca02c; font-weight: 700; }
  .metrics-table .fail { color: #c62828; font-weight: 700; }
  .metrics-table td.rate { font-variant-numeric: tabular-nums; font-weight: 700; }
  .metrics-table td.rate[data-tier="good"] { color: #2ca02c; }
  .metrics-table td.rate[data-tier="warn"] { color: #bf7900; }
  .metrics-table td.rate[data-tier="bad"]  { color: #c62828; }
  .metrics-table td.rate[data-tier="na"]   { color: var(--muted); font-weight: 400; }
  .metrics-table td.examples { color: var(--muted); font-family: ui-monospace, Menlo, monospace; font-size: 11px; }
  .metrics-foot { color: var(--muted); font-size: 11.5px; line-height: 1.6; margin: 4px 2px 24px; max-width: 980px; }
  .metrics-foot .gen { font-size: 11px; opacity: .8; margin-bottom: 6px; }
"""

_EXPLAIN_SCHEMA = (
    "Did each LLM output parse as valid JSON with the expected structure "
    "(a summary string plus a list of highlights, each with text and severity)? "
    "The build's validator drops malformed entries before rendering, so 100% "
    "confirms the validator is working — anything lower would mean the model "
    "produced output the dashboard couldn't render."
)
_EXPLAIN_REFERENCES = (
    "Every highlight references concrete things from the input — specific frame "
    "IDs and object IDs. This metric checks: for each citation, does the cited "
    "entity actually exist in the structured summary that was sent to the LLM? "
    "Invalid references would mean the model invented frames or objects that "
    "don't exist in the data. The build's validator already filters references "
    "on the way out, so a high rate confirms the safety net is working."
)
_EXPLAIN_GROUNDING = (
    "When the LLM cites a specific number — a tracker confidence value, a frame "
    "count, a percentage — does that number actually appear in the input data? "
    "This catches hallucinated specifics: numbers that sound plausible but the "
    "model made up. The check rounds floats to common precisions before matching, "
    "so a value of 0.31 still counts as grounded if the input has 0.314. A low "
    "score here means the model is inventing numerical details."
)
_METRICS_LIMITATIONS = (
    "Limitations: these three metrics measure well-formedness and grounding only. "
    "They do not measure whether the LLM caught the most important findings "
    "(coverage) or whether the commentary was useful to a reader — those require "
    "human review. For a higher-stakes evaluation, complement these automated "
    "metrics with a small Likert-rated user study."
)


def _fmt_pct(rate) -> str:
    return "n/a" if rate is None else f"{rate * 100:.0f}%"


def render_metrics_html(metrics_per_seq: dict) -> str:
    ov_css, _, _ = _extract_overview_parts()
    seqs = list(metrics_per_seq.keys())
    avail = [s for s in seqs if metrics_per_seq[s]["llm_available"]]

    # ---- aggregates over AVAILABLE sequences only ----
    sch_rate = (sum(1 for s in avail if metrics_per_seq[s]["schema"]["passed"]) / len(avail)
                if avail else None)

    def agg(metric_key, valid_key):
        tot = sum(metrics_per_seq[s][metric_key]["total"] for s in avail)
        val = sum(metrics_per_seq[s][metric_key][valid_key] for s in avail)
        return (val / tot) if tot else (1.0 if avail else None)

    ref_rate = agg("references", "valid")
    grd_rate = agg("grounding", "grounded")

    # ---- per-sequence rows ----
    def schema_rows():
        out = []
        for s in seqs:
            m = metrics_per_seq[s]
            if not m["llm_available"]:
                out.append(f'<tr><td>{_mesc(s)}</td>'
                           f'<td class="rate" data-tier="na">n/a</td>'
                           f'<td class="examples">LLM unavailable</td></tr>')
                continue
            sch = m["schema"]
            res = ('<span class="ok">PASS</span>' if sch["passed"]
                   else '<span class="fail">FAIL</span>')
            issues = "—" if not sch["issues"] else _mesc("; ".join(sch["issues"][:4]))
            out.append(f'<tr><td>{_mesc(s)}</td><td>{res}</td>'
                       f'<td class="examples">{issues}</td></tr>')
        return "\n".join(out)

    def rate_rows(metric_key, valid_key, examples_key):
        out = []
        for s in seqs:
            m = metrics_per_seq[s]
            if not m["llm_available"]:
                out.append(f'<tr><td>{_mesc(s)}</td>'
                           f'<td class="rate" data-tier="na">n/a</td>'
                           f'<td class="num-col">—</td>'
                           f'<td class="examples">LLM unavailable</td></tr>')
                continue
            mm = m[metric_key]
            tier = _metric_tier(mm["rate"])
            ex = mm.get(examples_key) or []
            ex_str = "—" if not ex else _mesc("; ".join(str(e) for e in ex))
            out.append(f'<tr><td>{_mesc(s)}</td>'
                       f'<td class="rate" data-tier="{tier}">{_fmt_pct(mm["rate"])}</td>'
                       f'<td class="num-col">{mm[valid_key]} / {mm["total"]}</td>'
                       f'<td class="examples">{ex_str}</td></tr>')
        return "\n".join(out)

    schema_card = f"""
  <div class="overview-card metrics-card">
    <div class="card-title">Schema Validity</div>
    <p class="card-explainer">{_EXPLAIN_SCHEMA}</p>
    <div class="big-stat"><span class="num">{_fmt_pct(sch_rate)}</span>
      <span class="lbl">aggregate pass rate</span></div>
    <table class="metrics-table">
      <tr><th>Sequence</th><th>Result</th><th>Issues</th></tr>
      {schema_rows()}
    </table>
  </div>"""

    ref_card = f"""
  <div class="overview-card metrics-card">
    <div class="card-title">Reference Validity</div>
    <p class="card-explainer">{_EXPLAIN_REFERENCES}</p>
    <div class="big-stat"><span class="num">{_fmt_pct(ref_rate)}</span>
      <span class="lbl">aggregate valid-reference rate</span></div>
    <table class="metrics-table">
      <tr><th>Sequence</th><th>Rate</th><th>Valid / Total</th><th>Failing examples</th></tr>
      {rate_rows("references", "valid", "invalid_examples")}
    </table>
  </div>"""

    grd_card = f"""
  <div class="overview-card metrics-card">
    <div class="card-title">Numeric Grounding</div>
    <p class="card-explainer">{_EXPLAIN_GROUNDING}</p>
    <div class="big-stat"><span class="num">{_fmt_pct(grd_rate)}</span>
      <span class="lbl">aggregate grounded-number rate</span></div>
    <table class="metrics-table">
      <tr><th>Sequence</th><th>Rate</th><th>Grounded / Total</th><th>Failing examples</th></tr>
      {rate_rows("grounding", "grounded", "ungrounded_examples")}
    </table>
  </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>REFLECT — LLM evaluation metrics</title>
<style>
{ov_css}
{_METRICS_CSS}
</style>
</head>
<body>
<div class="metrics-wrap">
  <div class="metrics-header">
    <h1>LLM Evaluation Metrics</h1>
    <div class="cap">Automated checks of the Overview-tab LLM commentary across all
    sequences. No human review required — every check compares the model's output
    against the structured summary that was sent to it.</div>
  </div>
{schema_card}
{ref_card}
{grd_card}
  <div class="metrics-foot">
    <div class="gen">Generated {_utc_now()} · prompt {_mesc(PROMPT_VERSION)}</div>
    {_METRICS_LIMITATIONS}
  </div>
</div>
</body>
</html>
"""


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

    metrics_per_seq = {}
    for seq in sequences:
        d = build_data(repo, seq)
        summary = d.pop("_llm_input", {})
        d["overview"]["llm_overview"] = get_llm_overview(model, seq, summary)
        payload["sequences"][seq] = d
        # Automated LLM-quality metrics: output validated against the structured
        # input that was sent to the model (no human annotation).
        llm_overview = d["overview"]["llm_overview"]
        metrics_per_seq[seq] = {
            "schema":        check_schema(llm_overview),
            "references":    check_references(summary, llm_overview),
            "grounding":     check_grounding(summary, llm_overview),
            "llm_available": bool(llm_overview.get("summary")),
        }
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

    # One combined tabbed page (Overview + Timeline). The single PAYLOAD carries
    # everything both tabs need, so no per-file stripping.
    HTML_OUT.write_text(render_merged(payload), encoding="utf-8")
    print(f"  wrote {HTML_OUT}")
    if OVERVIEW_OUT.exists():
        OVERVIEW_OUT.unlink()
        print(f"  removed stale {OVERVIEW_OUT}")

    METRICS_OUT.write_text(render_metrics_html(metrics_per_seq), encoding="utf-8")
    print(f"  wrote {METRICS_OUT}")

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
    grid-template-rows: 50px minmax(240px, 2.3fr) 1.15fr 1.15fr 108px;
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

  /* relation chip-chains: [object] (relation) [object], stacked one per row */
  .rel-chain {
    display: inline-flex; align-items: center; gap: 6px; margin: 2px 0; flex: 0 0 100%;
    transition: opacity 180ms ease, transform 180ms ease;
  }
  .rel-chain.entering, .rel-chain.exiting { opacity: 0; transform: scale(0.96); }
  .rel-chain .obj-chip { margin: 0; transition: none; }
  .rel-chip {
    padding: 1px 7px; border-radius: 10px; font-size: 11px; font-style: italic;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    white-space: nowrap; line-height: 1.4;
  }
  .rel-more { flex: 0 0 100%; color: var(--muted); font-size: 10.5px; padding: 2px 4px; }

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

  /* object-presence Gantt */
  .gantt-host { position: absolute; left: 0; right: 0; top: 18px; bottom: 14px; }
  .gantt-caption {
    position: absolute; bottom: 2px; left: 12px; right: 10px; z-index: 3;
    color: var(--muted); font-size: 10px; pointer-events: none;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
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
        <div class="obj-row">
          <span class="obj-row-label">Relations (<b id="rel-count">0</b>)</span>
          <span class="obj-chips" id="rel-chips"></span>
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

<div class="panel" id="panel-gantt">
  <div class="panel-ttl">Object presence</div>
  <div class="gantt-host"><canvas id="gantt-canvas"></canvas></div>
  <div class="gantt-caption" id="gantt-caption"></div>
  <div class="panel-msg" id="gantt-msg">No tracked objects.</div>
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
  // Single source of truth for object colors: the per-sequence map built at
  // build time from label color-words (CURRENT.object_colors). Falls back to
  // label-based resolution for ids not in the map (e.g. "gripper").
  function colorOf(oid) {
    const oc = CURRENT && CURRENT.object_colors;
    if (oc && oc[oid]) return oc[oid];
    return chipColor(oid);
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
  const ganttCanvas = $("gantt-canvas");
  const ganttCtx = ganttCanvas.getContext("2d");
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

  // ---- object-presence Gantt (drawn once per sequence; playhead sweeps it) ----
  function shortId(oid) {
    const m = String(oid).match(/^(.*)_(\d+)$/);
    return m ? (m[1].split(/\s+/).pop() + "_" + m[2]) : String(oid);
  }
  function drawGantt() {
    const host = ganttCanvas.parentElement;
    const cssW = host.clientWidth, cssH = host.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    ganttCanvas.width = cssW * dpr;
    ganttCanvas.height = cssH * dpr;
    ganttCanvas.style.width = cssW + "px";
    ganttCanvas.style.height = cssH + "px";
    const ctx = ganttCtx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    if (!CURRENT) return;

    const opr = CURRENT.object_presence_ranges || {};
    const allIds = Object.keys(opr);
    $("gantt-msg").classList.toggle("show", allIds.length === 0);
    const ids = allIds.slice(0, 6);

    // Align the plot area with the line charts so the shared playhead lines up.
    const left = geom.left || Y_AXIS_W;
    const width = geom.width > 0 ? geom.width : (cssW - left - 14);
    const axisH = 13;
    const areaH = cssH - axisH;
    if (width <= 0 || areaH <= 0) return;
    const rowH = areaH / Math.max(ids.length, 1);

    for (let i = 0; i < ids.length; i++) {
      const obj = opr[ids[i]];
      const color = colorOf(ids[i]);
      const y = i * rowH + 1.5;
      const h = Math.max(4, rowH - 3);
      // id label in the left gutter (same width as the charts' y-axis)
      ctx.fillStyle = color;
      ctx.font = "9px ui-monospace, Menlo, monospace";
      ctx.textBaseline = "middle"; ctx.textAlign = "left";
      ctx.fillText(shortId(ids[i]), 3, y + h / 2);
      // faint full-width track, then honest presence segments (gaps = unseen)
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      ctx.fillRect(left, y, width, h);
      ctx.fillStyle = color;
      for (const seg of obj.ranges) {
        const x0 = left + (seg[0] / MAXF) * width;
        const x1 = left + (seg[1] / MAXF) * width;
        ctx.fillRect(x0, y, Math.max(1, x1 - x0), h);
      }
    }

    // x-axis frame labels at 0/25/50/75/100%
    ctx.fillStyle = "#8b98a5";
    ctx.font = "9px -apple-system, sans-serif";
    ctx.textBaseline = "bottom"; ctx.textAlign = "center";
    for (let p = 0; p <= 1.0; p += 0.25) {
      ctx.fillText(String(Math.round(p * MAXF)), left + p * width, cssH - 2);
    }

    let cap = "Object presence over time (top 6 by total frames tracked). " +
      "Gaps = frames where the tracker did not see this object.";
    if (allIds.length > ids.length) cap += " · " + (allIds.length - ids.length) + " more not shown.";
    $("gantt-caption").textContent = cap;
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
  // Tracker box stroke: the object's own color (same as its chart line/Gantt/chip).
  // A non-ok validation status still overrides it (amber/red) as a degradation cue.
  function boxColorFor(oid, status) {
    if (status === "drifting" || status === "occluded" || status === "lost") return statusColor(status);
    return colorOf(oid);
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
    trk.forEach(b => drawBox(b.bbox, boxColorFor(b.oid, b.status), "TRK " + b.oid, s));
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
    const c = colorOf(oid);
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
    $("trk-chips").innerHTML = ""; $("sg-chips").innerHTML = ""; $("rel-chips").innerHTML = "";
    $("trk-count").textContent = "0"; $("sg-count").textContent = "0"; $("rel-count").textContent = "0";
  }
  function updateObjChips(f) {
    if (!CURRENT) return;
    const bbf = CURRENT.bbox_by_frame || {};
    const trkList = (bbf.tracker && bbf.tracker[f]) ? bbf.tracker[f].map(o => o.oid) : [];
    const sgList = (CURRENT.sg_by_frame && CURRENT.sg_by_frame[f]) ? CURRENT.sg_by_frame[f] : [];
    diffChips("trk-chips", "trk-count", trkList);
    diffChips("sg-chips", "sg-count", sgList);
  }

  // ---- Relations chip-chains: [from] (relation) [to], animated per chain ----
  const REL_CHIP_COLORS = {
    near:      { bg: "rgba(148,103,189,0.12)", fg: "#5b3b8c" },
    left_of:   { bg: "rgba(127,127,127,0.12)", fg: "#444"    },
    above:     { bg: "rgba(188,189,34,0.16)",  fg: "#7a7c12" },
    on_top_of: { bg: "rgba(227,119,194,0.14)", fg: "#9a3a7e" },
    inside:    { bg: "rgba(255,127,14,0.14)",  fg: "#9a4b07" },
  };
  const REL_MAX_CHAINS = 4;
  function makeRelObjChip(oid) {
    const c = colorOf(oid);
    const el = document.createElement("div");
    el.className = "obj-chip" + (isPhantom(oid) ? " phantom" : "");
    el.textContent = oid;
    el.style.color = c;
    el.style.background = hexToRgba(c, 0.12);
    return el;
  }
  function makeRelChip(relation) {
    const el = document.createElement("div");
    el.className = "rel-chip";
    el.textContent = relation.replace(/_/g, " ");
    const col = REL_CHIP_COLORS[relation] || { bg: "rgba(127,127,127,0.12)", fg: "#8b98a5" };
    el.style.background = col.bg;
    el.style.color = col.fg;
    return el;
  }
  function updateRelationChains(f) {
    const cont = $("rel-chips");
    const edges = (CURRENT && CURRENT.edges_by_frame && CURRENT.edges_by_frame[f]) || [];
    $("rel-count").textContent = edges.length;
    const shown = edges.slice(0, REL_MAX_CHAINS);
    const desired = new Map(shown.map(e => [e.from_id + "|" + e.relation + "|" + e.to_id, e]));

    const existing = new Map();
    cont.querySelectorAll(".rel-chain").forEach(el => existing.set(el.dataset.key, el));
    // exits: fade the whole chain out, then remove (timeout fallback)
    existing.forEach((el, key) => {
      if (!desired.has(key) && !el.classList.contains("exiting")) {
        el.classList.add("exiting");
        el.addEventListener("transitionend", () => { if (el.classList.contains("exiting")) el.remove(); }, { once: true });
        setTimeout(() => { if (el.isConnected && el.classList.contains("exiting")) el.remove(); }, 260);
      }
    });
    // entries: build [from](rel)[to] and fade the whole chain in
    desired.forEach((e, key) => {
      const ex = existing.get(key);
      if (ex) { ex.classList.remove("exiting"); return; }
      const chain = document.createElement("div");
      chain.className = "rel-chain entering";
      chain.dataset.key = key;
      chain.appendChild(makeRelObjChip(e.from_id));
      chain.appendChild(makeRelChip(e.relation));
      chain.appendChild(makeRelObjChip(e.to_id));
      cont.appendChild(chain);
      requestAnimationFrame(() => requestAnimationFrame(() => chain.classList.remove("entering")));
    });
    // "+N more" muted caption, kept as the last child
    let moreEl = cont.querySelector(".rel-more");
    const extra = edges.length - shown.length;
    if (extra > 0) {
      if (!moreEl) { moreEl = document.createElement("div"); moreEl.className = "rel-more"; }
      moreEl.textContent = "+" + extra + " more";
      cont.appendChild(moreEl);
    } else if (moreEl) {
      moreEl.remove();
    }
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
    updateRelationChains(f);
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
    colA = colorOf(OA);
    colB = colorOf(OB);

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
      reasonEl.className = "reason tl-only";
      reasonEl.innerHTML = "<b>GT failure:</b> " + escapeHtml(gt.failure_reason);
    } else {
      reasonEl.className = "reason tl-only empty";
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

    // caption + dropdown sync
    $("showing").innerHTML = "Currently showing: <b>" + escapeHtml(CURRENT.sequence_id) +
      "</b> · " + NF + " frames · " + (CURRENT.total_seconds || 0).toFixed(1) + "s";
    $("sequence-select").value = sid;

    // availability messages
    const avail = CURRENT.available || {};
    $("depth-msg").classList.toggle("show", !avail.depth);

    // video
    setVideoSrc(sid);

    // charts + strip
    destroyCharts();
    buildCharts();
    fillChartLegends();

    // reset playhead, paint after layout settles
    positionPlayheads(0);
    requestAnimationFrame(() => {
      recomputeGeom(); drawGantt(); update(0);
      requestAnimationFrame(() => { recomputeGeom(); drawGantt(); update(video.currentTime || 0); });
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
    // (the shared dropdown is populated + wired by the tab coordinator)
    // badges
    buildBadgesDOM();

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
        recomputeGeom(); drawGantt(); sizeBboxCanvas(); drawBboxes();
        update(video.currentTime || 0);
      }, 80);
    });
  }

  // Called by the tab coordinator when the Timeline tab becomes visible. Charts
  // built while the panel was hidden have zero size, so resize + redraw against
  // the now-live area; also (re)size the bbox canvas which needs the video laid out.
  function onShow() {
    recomputeGeom();
    if (confChart) confChart.resize();
    if (depChart) depChart.resize();
    recomputeGeom();
    drawGantt();
    sizeBboxCanvas();
    drawBboxes();
    update(video.currentTime || 0);
  }
  function onHide() {
    video.pause();
    stopBboxLoop();
  }

  initOnce();
  window.__timeline = { loadSequence, switchSequence, onShow, onHide };
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
      + `<div class="caption">Fraction of total frames where DINO ran due to a genuine `
      + `recovery trigger. Excludes <code>frame_counter_K</code> (the routine 30-frame `
      + `timeout) and the <code>init</code> detection. Answers: <em>how often did the `
      + `tracker need to recover?</em></div>`;
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

  // First paint on page load: render + staggered fade-in of the four cards.
  // (the shared dropdown is populated + wired by the tab coordinator)
  function firstPaint(sid) {
    renderAll(sid);
    ["card-general", "card-dino", "card-inv", "card-llm"].forEach((id, i) =>
      setTimeout(() => { const el = $(id); if (el) el.classList.add("in"); }, 50 * i));
  }

  window.__overview = { switchSequence, renderAll, firstPaint };
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Merged-page pieces (combined Overview + Timeline tabs)
# --------------------------------------------------------------------------- #
MERGED_HEAD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>REFLECT — pipeline dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
<style>
"""

# One shared header for both tabs. seqId/task/reason/readout are timeline-only
# (hidden on the Overview tab via body.tab-overview .tl-only). The dropdown +
# "showing" caption are the single shared instance.
GLOBAL_HEADER = r"""<header class="global-header">
  <span class="app-title">Pipeline Dashboard</span>
  <div class="seq-switcher"><select id="sequence-select"></select><span class="showing" id="showing"></span></div>
  <span class="seq tl-only" id="seqId"></span>
  <span class="task tl-only" id="taskName"></span>
  <span class="reason tl-only" id="reason"></span>
  <span class="spacer"></span>
  <span class="readout tl-only">Frame <span id="frameNo">0</span> / <span id="frameTot">0</span> &middot; t = <span class="t" id="timeNo">0.00</span> s</span>
  <div class="tabstrip">
    <button class="tab" data-tab="overview">Overview</button>
    <button class="tab" data-tab="timeline">Timeline</button>
  </div>
</header>
"""

# Page chrome — appended LAST so it overrides the index/overview body rules.
MERGE_CSS = r"""
  /* ===== merged page chrome (tabs + global header) — overrides above ===== */
  html, body { margin: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px; height: 100vh; overflow: hidden;
    display: flex; flex-direction: column; gap: 7px; padding: 7px 9px;
  }
  .global-header {
    flex: 0 0 auto; min-height: 44px;
    display: flex; align-items: center; gap: 12px;
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 0 12px; min-width: 0;
  }
  .app-title { font-weight: 700; font-size: 14px; white-space: nowrap; flex: 0 0 auto; }
  .global-header .seq-switcher { flex: 0 0 auto; flex-direction: column; align-items: flex-start; }
  .global-header .reason { flex: 0 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .global-header .spacer { flex: 1 1 auto; }
  .tl-only { display: inline-flex; }
  body.tab-overview .tl-only { display: none !important; }
  .tabstrip { display: flex; align-items: center; gap: 2px; flex: 0 0 auto; align-self: stretch; }
  .tab {
    appearance: none; -webkit-appearance: none; background: transparent; border: 0;
    cursor: pointer; color: var(--muted); font: inherit; font-size: 13px;
    padding: 6px 12px; border-bottom: 2px solid transparent; line-height: 1.2;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--text); border-bottom-color: #5b8def; }
  .tab-panels { flex: 1 1 auto; min-height: 0; position: relative; }
  .tab-panel { position: absolute; inset: 0; transition: opacity 150ms ease; opacity: 1; }
  .tab-panel.fading { opacity: 0; }
  #overview-panel { overflow-y: auto; padding: 2px 6px; }
  #timeline-panel {
    display: grid; grid-template-rows: minmax(220px, 2.3fr) 1.15fr 1.15fr 108px;
    gap: 7px; overflow: hidden;
  }
"""

# Tab coordinator — populates the single dropdown, drives both panels, handles
# the tab strip + localStorage. Runs after both IIFEs (which expose __timeline/__overview).
COORDINATOR_JS = r"""(function () {
  "use strict";
  const SEQS = PAYLOAD.sequences;
  const sel = document.getElementById("sequence-select");
  const T = window.__timeline || {};
  const O = window.__overview || {};

  // single shared dropdown, populated once
  Object.keys(SEQS).forEach(sid => {
    const o = document.createElement("option");
    o.value = sid;
    const t = SEQS[sid].gt && SEQS[sid].gt.task_name;
    o.textContent = t ? (sid + " — " + t) : sid;
    sel.appendChild(o);
  });
  const defSid = SEQS[PAYLOAD.default_sequence] ? PAYLOAD.default_sequence : Object.keys(SEQS)[0];
  sel.value = defSid;

  // Build both panels while both are still visible (Chart.js needs a sized canvas).
  if (T.loadSequence) T.loadSequence(defSid);
  if (O.firstPaint) O.firstPaint(defSid);

  // dropdown drives both panels on every change
  sel.addEventListener("change", () => {
    const sid = sel.value;
    if (T.switchSequence) T.switchSequence(sid);
    if (O.switchSequence) O.switchSequence(sid);
  });

  // ---- tabs ----
  const tabs = document.querySelectorAll(".tab");
  const ovPanel = document.getElementById("overview-panel");
  const tlPanel = document.getElementById("timeline-panel");
  function readTab() {
    try {
      const s = localStorage.getItem("dash2_active_tab");
      if (s === "overview" || s === "timeline") return s;
    } catch (e) {}
    return "overview";
  }
  function switchTab(name, animate) {
    document.body.classList.toggle("tab-overview", name === "overview");
    document.body.classList.toggle("tab-timeline", name === "timeline");
    tabs.forEach(t => t.classList.toggle("active", t.dataset.tab === name));
    const show = name === "overview" ? ovPanel : tlPanel;
    const hide = name === "overview" ? tlPanel : ovPanel;
    if (name !== "timeline" && T.onHide) T.onHide();   // pause video + bbox loop
    try { localStorage.setItem("dash2_active_tab", name); } catch (e) {}
    if (!animate) {
      hide.style.display = "none";
      show.style.display = "";
      if (name === "timeline" && T.onShow) T.onShow();
      return;
    }
    hide.classList.add("fading");
    setTimeout(() => {
      hide.style.display = "none";
      hide.classList.remove("fading");
      show.style.display = "";
      show.classList.add("fading");
      requestAnimationFrame(() => {
        show.classList.remove("fading");
        if (name === "timeline" && T.onShow) T.onShow();
      });
    }, 150);
  }
  tabs.forEach(t => t.addEventListener("click", () => switchTab(t.dataset.tab, true)));
  switchTab(readTab(), false);
})();
"""


if __name__ == "__main__":
    main()
