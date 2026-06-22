# Video + time-series sync dashboard

A synchronized dashboard for the perception pipeline. Plays the robot video
with live time-series charts, metric cards, event badges, scene-graph chips, and an
object-presence Gantt — everything updates in real time as the video plays or as you
scrub the seek bar.

Two tabs in one page:

- **Overview** — sequence-level summary: general info, DINO recovery stats, object-ID
  inventory per module, and an LLM-generated commentary.
- **Timeline** — the live video dashboard with the bbox overlay, live metric cards,
  event badges, chip rows for tracker / scene graph / relations, and three timeline
  panels (tracker confidence, depth median, object-presence Gantt).

Everything fits in one viewport.

> **What this README covers.** Only the dashboard-specific setup. The perception
> pipeline that produces the JSONLs is documented elsewhere by the team, generate
> your sequences first, then come back here.

---

## Setup

### 1. Python dependencies

From the repo root:

```bash
pip install pandas numpy opencv-python requests
```

`opencv-python` is used to probe each video's natural pixel dimensions for the bbox
overlay. `requests` is used to call Ollama for the LLM commentary. The build
degrades gracefully if either is missing.

### 2. Source video files

The build looks for each sequence's video at

```
example_data/real_data/{sequence_id}/videos/color.mp4
```

and stages it into `dashboard 2/video/{sequence_id}/color.mp4` (the `dashboard 2/video/`
folder is gitignored, so a fresh clone has none of these).

`example_data/real_data/putAppleBowl1/` is whitelisted in the repo's `.gitignore`,
so `putAppleBowl1`'s video and zarr come along with a fresh clone, that sequence
works out of the box. For any other sequence (e.g. `putPearDrawer1`), copy its
`color.mp4` into `example_data/real_data/{sequence_id}/videos/color.mp4` before
building.

If a sequence has JSONLs but no video, the page still renders all the charts and
data — only the video element shows a "video not available" placeholder.

### 3. Ollama (optional, for the LLM commentary on the Overview tab)

The build calls a local Ollama server during the build and embeds the commentary
into the page. If Ollama isn't running, the commentary card shows an "unavailable"
message and the rest of the page works fine, you can skip this step entirely.

If you want the LLM commentary:

```bash
# Install Ollama: https://ollama.ai/download

ollama serve            # in one terminal — leave running
ollama pull llama3.1:8b # in another terminal
```

`llama3.1:8b` is preferred; the build also accepts `llama3:latest` as a fallback.
Outputs are cached on disk at `dashboard 2/.llm_cache/`, so subsequent builds are
fast.

---

## Run

From the repo root:

```bash
python "dashboard 2/run.py"
```

This auto-discovers every sequence with valid JSONLs under
`pipeline/real_world/jsonl/`, stages each one's video into `dashboard 2/video/`,
writes `dashboard 2/index.html` with all data embedded inline, starts a local HTTP
server with video-seek support on `http://127.0.0.1:8765/`, and opens the dashboard
in your default browser. Press **Ctrl-C** to stop.

If you edit `build.py` or generate fresh JSONLs, Ctrl-C, rerun `run.py`, and refresh
the browser to see the changes.

---

## Add a new sequence

The dashboard auto-discovers sequences. To add one:

1. Have the upstream pipeline generate its JSONLs at
   `pipeline/real_world/jsonl/{sequence_id}/{sequence_id}__detection.jsonl`
   (and the other module JSONLs with the same `{seq}__module.jsonl` naming, for example "putAppleBowl1__depth.jsonl", "putAppleBowl1__scene_graph.jsonl").
2. Place its `color.mp4` at
   `example_data/real_data/{sequence_id}/videos/color.mp4`.
3. Rerun `run.py`. The new sequence appears in the header dropdown.

A sequence is included as long as it has the detection and tracking JSONLs;
depth / scene_graph / validation are optional.

To change which sequence is selected on first load, edit `DEFAULT_SEQUENCE` at the
top of `build.py` and rerun.

---

## Troubleshooting

**The sequence dropdown is empty.** No sequences have the required JSONL pair.
Confirm JSONLs are at `pipeline/real_world/jsonl/{sid}/{sid}__detection.jsonl` and
`{sid}__tracking.jsonl` (exact naming, double underscore). Rerun `run.py`.

**The dashboard loads but the video doesn't play.** Source video missing. Place
`color.mp4` at `example_data/real_data/{sid}/videos/color.mp4` and rebuild.

**The LLM commentary card shows "unavailable".** Ollama isn't running, or the model
isn't pulled. Run `ollama serve` and `ollama pull llama3.1:8b`, then rerun `run.py`.
After switching models or editing the prompt, bump `PROMPT_VERSION` at the top of
`build.py` to invalidate the on-disk cache.

**Changes don't show up after refresh.** Stop the running `run.py` (Ctrl-C) and
restart. The HTML is generated at build time, not per request, so a browser refresh
alone won't see code changes.

---

## Files

- `run.py`        — build + serve + open browser (range-capable static server).
- `build.py`      — discovers all sequences, reads their JSONLs, calls Ollama for
                    LLM commentary, writes the combined `index.html`.
- `index.html`    — generated artifact with both tabs and every sequence's data
                    embedded inline as `const PAYLOAD = {...}`.
- `video/{seq}/color.mp4` — each sequence's video symlinked or copied here so the
                    HTTP server can serve it. Gitignored.
- `.llm_cache/`   — on-disk cache of LLM outputs, keyed by sequence + prompt version
                    + input hash. Gitignored.
