# live-scene-graph

Point a webcam at your own objects, type what to look for, and watch a live
scene graph of spatial relations build itself.

```
┌──────────────────────────┬──────────────────────────┐
│  webcam + tracked boxes  │  live scene graph        │
│  mug 0.91 0.62m          │   (apple)──inside──(bowl)│
│  bowl 0.85 0.65m         │      │near               │
│                          │   (mug)                  │
└──────────────────────────┴──────────────────────────┘
```

Pipeline: **YOLOE** (open-vocabulary text prompts) → **BoTSORT** tracking →
**Depth Anything V2** (metric-indoor) → REFLECT **scene-graph builder** →
OpenCV split-view render. Relations: `inside`, `on_top_of`, `near`,
`left_of`, `above`.

The scene-graph builder (`src/livescene/scene_graph/`) is copied from the
REFLECT pipeline; its robot/gripper code paths are present but dormant
(always called with `gripper_closed=False`).

## Install

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
cd live-scene-graph
uv sync
```

`uv` resolves the default PyTorch wheels: on Linux the default wheel includes
CUDA, on macOS it includes MPS — verified working on Linux + RTX 4080 with no
extra index configuration. Check your install with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
```

Model weights (YOLOE ~30 MB + MobileCLIP text encoder ~570 MB + Depth
Anything V2 Small ~100 MB) download automatically on first run.

## Run (live)

```bash
uv run python -m livescene.app --source 0
```

- A window opens with the split view. Press **q** in the window to quit.
- **Type a new comma-separated prompt list in the terminal + Enter** to
  re-prompt the detector at any time, e.g.:

  ```
  apple, bowl, mug, book
  ```

  Re-prompting resets tracking ids and starts a fresh graph. Multi-word
  prompts ("red square", "coffee mug") work.

Useful flags:

```bash
uv run python -m livescene.app --source 0 \
    --prompts "apple, bowl, mug" \   # initial prompts
    --conf 0.3 \                     # raise to suppress spurious boxes
    --hfov 70 \                      # your camera's horizontal FOV (deg)
    --depth-every-k 3 \              # run depth less often (faster)
    --width 1280 --height 720
```

`--source` also accepts a video path, an image path (repeated forever), or
`synthetic` (a generated test scene of coloured shapes — prompt it with
`red square, blue rectangle, green circle`).

## Run (headless)

No window; writes `scene_graph.jsonl`, `annotated.mp4`, and `last_frame.png`
to `--out`:

```bash
uv run python -m livescene.app --source synthetic --headless --max-frames 60 --out out/
```

`scripts/make_test_clip.py` generates the synthetic scene as an mp4 for
file-based runs.

## Config knobs

| flag / `AppConfig` field | default | note |
|---|---|---|
| `--yoloe-model` | `yoloe-11s-seg.pt` | `-11m`/`-11l` for quality over speed |
| `--depth-model` | `Depth-Anything-V2-Metric-Indoor-Small-hf` | Base/Large for quality |
| `--conf` | `0.25` | YOLOE confidence floor |
| `--hfov` | `60.0` | horizontal FOV → intrinsics; set to your camera's |
| `--depth-every-k` | `2` | depth is the bottleneck; reuse the map between runs |
| `--depth-input-long-side` | off | downscale depth input (helps on weak GPUs/MPS) |
| `--device` | auto | `cuda` > `mps` > `cpu` |
| `AppConfig.scene_graph` | webcam-tuned | relation thresholds (`near_threshold_m=0.45` etc.) |

Scene-graph relation thresholds live in `SceneGraphConfig`
(`src/livescene/scene_graph/builder.py`); `AppConfig` overrides a few of them
for desk-scale webcam scenes (see `config.py`).

## Performance

Measured on an RTX 4080 (CUDA, fp16 depth), full pipeline including render:

- 640×480 synthetic: ~14 FPS
- 1280×720: ~13 FPS (depth ~25 ms every 2nd frame, YOLOE ~50 ms every frame)

The live webcam is read by a capture thread that keeps only the newest frame,
so inference latency never builds up a lag backlog.

## Known limitations

- **Intrinsics are approximate.** They are derived from resolution + an
  assumed horizontal FOV (default 60°), not calibration. 3D positions are
  coarse; distance-based relations (`near`) are approximate. If relations
  trigger too eagerly/lazily, set `--hfov` to your camera's real FOV and/or
  tune `SceneGraphConfig.near_threshold_m`.
- **Monocular metric depth is approximate.** Depth Anything V2 metric-indoor
  outputs plausible metres indoors but absolute scale can be off by tens of
  percent, and it changes scale with scene content. Depth-gated relations
  (`on_top_of`'s support check, depth-jump occlusion) are tuned loosely.
- **Open-vocab detection flickers.** Near the confidence floor, YOLOE can
  produce occasional spurious boxes or BoTSORT id switches; vanished ids
  linger for 10 frames as decaying "occluded" ghost nodes (REFLECT's
  occlusion buffer) before a `Missing` flag fires.
- Edges are computed per frame with no temporal smoothing (a stretch goal).

## Tests

```bash
uv run pytest tests/ -q
```

- `test_relations.py` — deterministic scene-graph correctness (no camera, no
  models): hand-built tracked objects + synthetic depth assert `inside`,
  `on_top_of`, `near`, `left_of`, `above`, ghosting, `Missing`, depth-jump
  occlusion.
- `test_intrinsics.py` — K from FOV.
- `test_detector_smoke.py` / `test_depth_smoke.py` — models load and satisfy
  their I/O contracts on the detected device (skip if weights can't
  download).
- `test_pipeline_headless.py` — real models on the synthetic scene end to
  end; parses the JSONL and asserts the drifting square ends up `inside` the
  rectangle.
- `test_render.py` — split-view geometry from a synthetic `SceneGraphFrame`.
