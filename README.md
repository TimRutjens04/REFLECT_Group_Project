# ARGUS - Adaptive Re-detection with Grounding for Unconstrained Robot Scenes

> A real-world object tracking pipeline for robot failure analysis, built on top of the [REFLECT](https://robot-reflect.github.io/) framework.

---

## Why this exists

The [REFLECT paper](https://robot-reflect.github.io/) proposes a multimodal framework for analysing robot execution failures using video, audio, and scene graphs. Their original pipeline is designed for the **RoboFail** dataset, this is a curated set of simulated and real robot trajectories and relies on data structures and pre-processing steps.

When we tried to apply REFLECT to **real-world recordings from a UR5e robot arm**, the original pipeline did not fit: different data layout, no guaranteed static scene, and real tracking noise (motion blur, occlusion, lighting changes). So we built ARGUS: a new detection–tracking pipeline designed from scratch.

---

## What it does

ARGUS takes a robot task description and a paired RGB-D video, and produces:

- A **per-frame annotated tracking video** showing bounding boxes and track IDs
- A **JSONL detection log** with confidence scores and trigger reasons for every detection event
- A **JSONL tracking log** with bounding box geometry, centroid drift, and area-change flags per frame
- **Per-frame JSON state snapshots** capturing the full detection context

The pipeline is self-healing: when YOLOE tracking degrades (bbox drift, area explosion, or object loss), it automatically falls back to Grounding DINO to re-detect the object and re-prime the tracker.

---

## Pipeline logic

here should be full architecture image

**Models used:**

| Role | Model |
|---|---|
| Zero-shot object detection | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) (`grounding-dino-tiny`) |
| Instance tracking | [YOLOE-11L-SEG](https://github.com/ultralytics/ultralytics) |
| Multi-object tracker | BoTSORT (via Ultralytics) |
| Alternative tracker | SAM2 (`sam2_b.pt`) |

---

## Sample output

**Detection overlay - frame 0 (`putAppleBowl1`)**

Generated at runtime: `outputs/<run_id>/images/detection_step_0.png`

Both the red apple (conf 0.93) and dark blue bowl (conf 0.85) are detected by Grounding DINO on the first frame. YOLOE is then seeded with these bounding boxes as visual prompts and tracks through the rest of the episode.

**Tracked video**

`pipeline/outputs/<run_id>/videos/tracked_putAppleBowl1.mp4` - YOLOE bounding boxes rendered green; Grounding DINO re-detection frames rendered in orange (`GDINO: <label>`).

**Detection JSONL snippet**

```json
{
  "sequence_id": "putAppleBowl1",
  "frame_id": 117,
  "trigger_reason": "tracker_low_confidence",
  "detections": [
    { "label": "red apple","confidence": 0.930, "bbox_xyxy": [...] },
    { "label": "dark blue bowl","confidence": 0.847, "bbox_xyxy": [...] }
  ],
  "detection_success": true,
  "runtime_ms": 450.0
}
```

---

## Quick start

```bash
# 1. Enter the pipeline directory
cd pipeline

# 2. Install dependencies (Python 3.11+, uv)
uv sync

# 3. Place your data under example_data/ (see Data format below)

# 4. Run the pipeline
uv run python main.py
```

Results land in `pipeline/outputs/<run_id>/`, where `<run_id>` is a timestamped folder per run, e.g. `outputs/20260610_141523_putAppleBowl1/` (see Output layout below).

---

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (for dependency management)
- A CUDA-capable GPU is strongly recommended (CPU works but is very slow)
- RGB-D recordings from a UR5e (or compatible robot) - see Data format

---

## Pipeline commands

All entry points live in `pipeline/`. Run them with `uv run python <script>`.

### Full pipeline (main entry point)

```bash
uv run python main.py
```

Runs detection on frame 0 of task 1, then tracks through the full episode with YOLOE + re-detection every 30 frames.

### Configurable options (edit `main.py`)

| Parameter | Default | Description |
|---|---|---|
| `redetect_every_n_frames` | `30` | Force a Grounding DINO re-detect every N frames |
| `redetect_on_lost` | `False` | Trigger re-detect when a label goes missing |
| `redetect_on_invalid` | `True` | Trigger re-detect on bbox drift / area change |
| `validate_with_depth` | `True` | Use depth channel for area-normalised validation |
| `frame_step` | `1` | Process every Nth frame (1 = all frames) |

### Switch tasks

```python
# In main.py, change the task ID:
task = loader.get(2)   # Task 2: putAppleBowl2
```

### Use SAM2 instead of YOLOE

Uncomment the `track_video_with_sam2(...)` block in `main.py` and comment out the YOLOE call.

---

## Data format

### Input layout

```
example_data/
├── tasks_real_world.json          # task definitions (see below)
└── real_data/
    └── <general_folder_name>/     # e.g. putAppleBowl1
        └── videos/
            ├── color.mp4          # RGB video from robot camera
            └── depth/             # zarr array OR depth.mp4
                └── .zarray        # (zarr format)
```

### Task definition (`tasks_real_world.json`)

```json
"Task 1": {
  "name": "put apple in bowl",
  "general_folder_name": "putAppleBowl1",
  "object_list": ["red apple", "dark blue bowl"],
  "actions": ["Pick up apple", "Put apple inside bowl"],
  "success_condition": "apple is inside bowl.",
  "gt_failure_reason": "Apple is placed on top of the bowl instead of inside..."
}
```

`object_list` drives the Grounding DINO prompts automatically — no manual prompting needed.

### Output layout

Every pipeline run gets its own timestamped folder under `pipeline/outputs/`, named `<YYYYMMDD_HHMMSS>_<task_folder_name>`:

```
pipeline/outputs/
└── 20260610_141523_putAppleBowl1/    # one folder per run
    ├── run_metadata.json             # run ID, task, config, git commit
    ├── images/
    │   └── detection_step_0.png      # detection overlay for frame 0
    ├── jsonl/
    │   ├── detections.jsonl          # one JSON object per detection event
    │   ├── tracking.jsonl            # one JSON object per tracked frame
    │   └── validation.jsonl          # validator handoff consumed by the depth stage
    ├── state_summary/
    │   └── detection/
    │       ├── frame_0000.json       # full detection state at frame 0
    │       ├── frame_0030.json       # ... and every re-detect trigger
    │       └── ...
    ├── videos/
    │   └── tracked_<task_name>.mp4   # annotated output video
    └── depth/                        # depth + scene graph stage outputs
        ├── <seq>__depth.jsonl
        ├── <seq>__scene_graph.jsonl
        ├── <seq>__keyframes.json
        ├── plots/
        ├── visualizations/
        └── graph_visualizations/
```

#### `detections.jsonl` fields

| Field | Type | Description |
|---|---|---|
| `sequence_id` | str | Task folder name |
| `frame_id` | int | Frame index |
| `trigger_reason` | str | `init` / `frame_counter_K` / `tracker_low_confidence` |
| `prompts_used` | list[str] | Exact prompts sent to Grounding DINO |
| `detections` | list | `label`, `bbox_xyxy`, `confidence`, `is_selected` |
| `detection_success` | bool | Whether any object was found |
| `runtime_ms` | float | Detector wall-clock time |

#### `tracking.jsonl` fields

| Field | Type | Description |
|---|---|---|
| `sequence_id` | str | Task folder name |
| `frame_id` | int | Frame index |
| `tracked_objects` | list | Per-object: `bbox_xyxy`, `center_xy`, `displacement_px`, `bbox_area_ratio_to_init`, `tracker_confidence` |
| `flags.bbox_size_change_flag` | bool | Bbox area changed > 67% from initial |
| `flags.drift_flag` | bool | Centroid moved > 50 px in one step |

---

## Troubleshooting

**FileNotFoundError: Color video missing**


Check that `example_data/real_data/<folder_name>/videos/color.mp4` exists and that the folder name in `tasks_real_world.json` matches exactly.

**RuntimeError: No depth source available for frame N**


The loader tries zarr, depth video, and zarr-chunked formats in order. Ensure at least one depth source is present. If you only have RGB, set `validate_with_depth=False` in `main.py`.

**Cannot prime YOLOE without detections**

 Grounding DINO returned nothing on frame 0. Try lowering `box_threshold` in `DetectorConfig` (default 0.4), or adjust the `object_list` labels in your task JSON to match the visual appearance more closely. If that does not work you may have to edit the data to begin at the point where GDino can see the objects.

**YOLOE re-detects every frame (high trigger rate)**


Reduce sensitivity: increase `drift_thresh_px` and `area_change_thresh` in `CompositeTrackingValidator`, or increase `validator_settle_frames` to give the tracker more time to stabilise after a re-prime.