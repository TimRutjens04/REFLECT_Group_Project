# REFLECT - Multimodal Perception & State Representation

Research pipeline for the **RoboFail / REFLECT** dataset (Columbia University).
Encodes robot episode video and audio into compact state representations using pretrained CLIP and WAV2CLIP embeddings, then analyzes normal vs. failure separation.

---

## Project structure

```
code/               pipeline scripts (align → encode → analyze)
notebooks/          exploration.ipynb — narrative analysis with all figures
docs/               research plan and project deliverables
data/               raw dataset (not in version control — see below)
aligned/            output of align.py  — .npz per episode
encoded/            output of encode.py — .npz per episode
analysis/           output of analyze.py — figures + metrics.json
```

---

## Setup

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- [just](https://just.systems/man/en/) — command runner

#### macOS

```bash
brew install just
```

#### Windows

PowerShell, requires [Scoop](https://scoop.sh) or [Winget](https://learn.microsoft.com/en-us/windows/package-manager/):

```powershell
scoop install just
# or
winget install Casey.Just
```

> On Windows, `just` runs commands via `sh` by default. Install [Git for Windows](https://git-scm.com/download/win) to get a compatible shell, or prefix commands with `poetry run python` manually if `just` is unavailable.

### Install dependencies

```bash
poetry install
```

> Some packages (torch, torchvision, wav2clip) are pinned for Apple Silicon / MPS compatibility.
> On Windows or Linux, torch will fall back to CPU automatically — no code changes needed, but encoding will be slower.
> If you hit version conflicts on Windows, check `pyproject.toml` and pin torch to the latest stable CPU wheel.

### Dataset

The `data/` folder is excluded from version control (large binary files).

```
data/
  Archive/
    boilWater-1/
    makeSalad-5/
    ...
  real_world/
    putFruitsBowl2/
```

---

## Running the pipeline

All commands go through `just`. Run `just` with no arguments to see the full list.

### Full pipeline

```bash
just all        # align → encode → analyze in sequence
```

### Step by step

| Command | Input | Output | Description |
|---|---|---|---|
| `just align` | `data/` | `aligned/*.npz` | Extract frames + audio windows, assign failure labels |
| `just align-one <episode>` | `data/<episode>` | `aligned/<episode>.npz` | Align a single episode |
| `just encode` | `aligned/` | `encoded/*.npz` | Run CLIP (vision) + WAV2CLIP (audio), produce 512-dim embeddings |
| `just analyze` | `encoded/` | `analysis/` | UMAP plots, silhouette scores, PCA trajectories, metrics.json |
| `just notebook` | `encoded/` | `notebooks/exploration.ipynb` | Re-execute the full analysis notebook |

### Inspection & validation

```bash
just inspect    # print raw data structure (episodes, fps, audio, failure timestamps)
just sanity     # run 5-check validation on all aligned .npz files
```

### Code quality

```bash
just lint       # ruff: check for errors and style issues
just lint-fix   # ruff: auto-fix fixable issues
just fmt        # ruff: format all code
just check      # lint + format in one pass
```

---

## Pipeline outputs

Each `.npz` file follows a consistent schema:

### aligned/

```
timestamps        (N,)          float64 — seconds from episode start
frames            (N, H, W, 3)  uint8   — RGB frames
audio_windows     (N, n_samples) float32 — 0.5s mono audio centered on each frame
failure_labels    (N,)          bool    — True within 0.5s of annotated failure
fps_base, audio_sr
```

### encoded/

```
timestamps           (N,)       float64
visual_embeddings    (N, 512)   float32 — CLIP ViT-B/32, L2-normalized
audio_embeddings     (N, 512)   float32 — WAV2CLIP, L2-normalized
failure_labels       (N,)       bool
fps_base, audio_sr
```

---

Key metric results are written to `analysis/metrics.json` after each `just analyze` run.
