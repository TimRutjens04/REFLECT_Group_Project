"""Headless end-to-end: real models on synthetic frames, validated via JSONL.

This is the agent-side acceptance test (the live webcam is the human's).
Skips if model weights cannot be loaded in this environment.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def jsonl_rows(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("headless")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "livescene.app",
            "--source",
            "synthetic",
            "--headless",
            "--max-frames",
            "60",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        tail = proc.stderr[-2000:]
        if any(s in tail.lower() for s in ("download", "connection", "urlopen", "http")):
            pytest.skip(f"model weights unavailable: {tail}")
        pytest.fail(f"headless run failed (rc={proc.returncode}):\n{tail}")

    jsonl = out_dir / "scene_graph.jsonl"
    assert jsonl.exists(), "scene_graph.jsonl not written"
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    assert (out_dir / "annotated.mp4").exists()
    return rows


def test_jsonl_has_frames_with_nodes(jsonl_rows):
    assert len(jsonl_rows) == 60
    frames_with_nodes = [r for r in jsonl_rows if r["nodes"]]
    assert frames_with_nodes, "no frame contained any nodes"
    row = frames_with_nodes[0]
    assert {"sequence_id", "frame_id", "timestamp", "nodes", "edges", "localization_flag"} <= set(row)


def test_node_row_shape(jsonl_rows):
    node = next(n for r in jsonl_rows for n in r["nodes"])
    assert {"object_id", "label", "pixel_center", "depth_used_m", "position_3d", "bbox_xyxy", "status"} <= set(node)
    label, tid = node["object_id"].rsplit("_", 1)
    assert label == node["label"]
    assert tid.isdigit()


def test_inside_or_near_edge_emitted(jsonl_rows):
    relations = {e["relation"] for r in jsonl_rows for e in r["edges"]}
    assert relations & {"inside", "near"}, f"no inside/near edge in 60 frames, got {relations}"


def test_drifting_square_ends_inside_rectangle(jsonl_rows):
    # The red square drifts into the blue rectangle; by the end of the clip
    # the dominant relation between them must be inside(square -> rectangle).
    late = jsonl_rows[45:]
    hits = [
        e
        for r in late
        for e in r["edges"]
        if e["relation"] == "inside"
        and e["from_object_id"].startswith("red square")
        and e["to_object_id"].startswith("blue rectangle")
    ]
    assert len(hits) >= len(late) // 2, f"square-in-rectangle only in {len(hits)}/{len(late)} late frames"


def test_depth_values_metric(jsonl_rows):
    depths = [n["depth_used_m"] for r in jsonl_rows for n in r["nodes"] if n["depth_used_m"] > 0]
    assert depths
    assert all(0.05 < d < 20.0 for d in depths), "depths not plausible metres"
