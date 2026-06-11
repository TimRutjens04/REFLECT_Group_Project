# Pipeline Analysis Dashboard

Interactive Streamlit dashboard for inspecting REFLECT RoboFail pipeline outputs
(detection, tracking, depth, scene graph) on a per-sequence basis.

---

## Requirements

- Python 3.10+
- Streamlit ≥ 1.35 (required for clickable chart points)

Install dependencies:

```bash
pip install -r dashboard/requirements.txt
```

---

## Quick Start

```bash
# From the repo root:
streamlit run dashboard/dashboard.py
```

The dashboard auto-discovers sequences under `pipeline/real_world/jsonl/` and
ground-truth data from `example_data/tasks_real_world.json`. Override paths in
the sidebar at runtime.

---

## Extracting Frames (Frame Viewer)

The Frame Viewer tab and thumbnail previews require JPEG frames extracted from
each sequence's `color.mp4`:

```bash
# By folder name:
python pipeline/scripts/extract_frames.py --folder-name putAppleBowl1

# By task ID (looked up in tasks_real_world.json):
python pipeline/scripts/extract_frames.py --task-id 1

# Overwrite existing frames:
python pipeline/scripts/extract_frames.py --folder-name putAppleBowl1 --force
```

Frames are saved to `example_data/real_data/<folder_name>/frames/<frame_id:06d>.jpg`
and are excluded from git (see `.gitignore`).

---

## Feature Flags

Two flags at the top of `dashboard/dashboard.py` control optional tabs:

| Flag                   | Default | Enable when …                                    |
|------------------------|---------|--------------------------------------------------|
| `SHOW_DEPTH_TAB`       | `False` | `__depth.jsonl` files are available per sequence |
| `SHOW_SCENE_GRAPH_TAB` | `False` | `__scene_graph.jsonl` files are available        |

When `False`, the tab is still shown but displays an info message instead of charts.

---

## LLM commentary

The dashboard works with any Ollama model you have installed. The sidebar shows a
dropdown of installed models; pick whichever you want to try. For best output quality,
install `llama3.1:8b`:

```
ollama serve     # in one terminal
ollama pull llama3.1:8b
```

The dashboard will fall back to whatever is installed if `llama3.1:8b` is not present,
with a small warning in the sidebar noting quality may be lower.

The Overview tab includes an optional LLM commentary layer that produces a plain-English
summary and up to 6 findings with severity chips. Commentary is not generated
automatically; click **Regenerate** once per sequence to trigger it. Results are cached
by `(model, prompt_version, summary_hash)` so re-running the page does not re-call
Ollama unless you click Regenerate again — and each model's output is cached
independently, so switching models and regenerating compares cleanly.

## File Layout

```
dashboard/
├── dashboard.py       # Main Streamlit entry point
├── utils.py           # Path discovery, frame lookup, mismatch helpers
├── loaders.py         # Cached JSONL + GT parsers
├── metrics.py         # Scene-graph and reference-stats calculations
├── plots.py           # All Plotly chart builders
├── interactive.py     # Inline frame viewer, stats table, absent-frame diagnosis
├── frame_viewer.py    # BBox overlay renderer
└── requirements.txt   # Python dependencies
```
