"""Utility helpers: path discovery, mm:ss parsing, frame lookup, mismatch detection."""
import os
from pathlib import Path

# Only detection and tracking are required for sequence discovery.
# Depth and scene-graph files are optional — controlled by SHOW_* flags in dashboard.py.
REQUIRED_SUFFIXES = ["__detection.jsonl", "__tracking.jsonl"]


def discover_sequences(outputs_root: str) -> list[str]:
    """
    Scan subdirectories of outputs_root for sequences that have at minimum
    {seq}__detection.jsonl and {seq}__tracking.jsonl.
    """
    root = Path(outputs_root)
    if not root.exists():
        return []
    seqs = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        sid = folder.name
        if all((folder / f"{sid}{s}").exists() for s in REQUIRED_SUFFIXES):
            seqs.append(sid)
    return seqs


def mmss_to_seconds(s: str) -> float:
    """Convert 'mm:ss' string to total seconds."""
    parts = s.strip().split(":")
    return int(parts[0]) * 60 + float(parts[1])


def find_frame_path(frames_dir: str, frame_id: int) -> str | None:
    """
    Try multiple naming conventions; return first path that exists, else None.

    Priority order (first wins):
      1. {id:06d}.jpg      ← output of pipeline/scripts/extract_frames.py
      2. {id:06d}.png
      3. frame_{id:04d}.jpg  ← legacy mock data convention
      4. frame_{id:04d}.png
      5. frame_{id}.jpg
      6. frame_{id}.png
    """
    d = Path(frames_dir)
    candidates = [
        d / f"{frame_id:06d}.jpg",
        d / f"{frame_id:06d}.png",
        d / f"frame_{frame_id:04d}.jpg",
        d / f"frame_{frame_id:04d}.png",
        d / f"frame_{frame_id}.jpg",
        d / f"frame_{frame_id}.png",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def detect_object_mismatches(det_ids: set, track_ids: set, sg_ids: set) -> dict:
    """Return per-object presence flags across detection, tracking, scene graph."""
    all_ids = det_ids | track_ids | sg_ids
    return {
        oid: {
            "in_detection":   oid in det_ids,
            "in_tracking":    oid in track_ids,
            "in_scene_graph": oid in sg_ids,
        }
        for oid in sorted(all_ids)
    }
