# REFLECT — RoboFail multimodal pipeline
# Usage: just <recipe>

set dotenv-load := true

python := "poetry run python3"

# List available recipes
default:
    @just --list

# ── Setup ─────────────────────────────────────────────────────────────────────

# Install all dependencies via Poetry
install:
    poetry install

# ── Pipeline ──────────────────────────────────────────────────────────────────

# Run full pipeline: align → encode → owl → analyze
all: align encode owl analyze

# Align all episodes (raw data → aligned/*.npz)
align:
    {{python}} code/align.py

# Align a single episode by name, e.g: just align-one makeSalad-5
align-one episode:
    {{python}} code/align.py {{episode}}

# Encode all aligned episodes (aligned/*.npz → encoded/*.npz)
encode:
    {{python}} code/encode.py aligned/

# Run OWL-ViT object detection on all episodes (→ owl/*.npz)
owl:
    {{python}} code/owl_detect.py

# Run OWL-ViT for a single episode by name, e.g: just owl-one appleInFridge1
owl-one episode:
    {{python}} code/owl_detect.py {{episode}}

# Force-rerun OWL-ViT on all episodes, overwriting existing results
owl-force:
    {{python}} code/owl_detect.py --force

# Force-rerun OWL-ViT for a single episode, e.g: just owl-force-one appleInFridge1
owl-force-one episode:
    {{python}} code/owl_detect.py {{episode}} --force

# Analyze all encoded episodes (encoded/*.npz → analysis/)
analyze:
    {{python}} code/analyze.py encoded/

# ── Perception pipeline (Stages 1–5) ─────────────────────────────────────────

# Stage 1 — Grounding DINO open-vocab detection (aligned/*.npz → detect/*.npz)
stage1:
    {{python}} code/pipeline/detect.py

# Stage 1 for a single episode, e.g: just stage1-one boilWater-1
stage1-one episode:
    {{python}} code/pipeline/detect.py {{episode}}

# Stage 2 — SAM 2 instance segmentation (detect/*.npz → segment/*.npz)
stage2:
    {{python}} code/pipeline/segment.py

# Stage 2 for a single episode
stage2-one episode:
    {{python}} code/pipeline/segment.py {{episode}}

# Stage 3 — Depth Anything V2 + SigLIP 2 (segment/*.npz → depth_state/*.npz)
stage3:
    {{python}} code/pipeline/depth_state.py

# Stage 3 for a single episode
stage3-one episode:
    {{python}} code/pipeline/depth_state.py {{episode}}

# Stage 4 — IoU-based temporal tracking (detect+segment → track/*.npz)
stage4:
    {{python}} code/pipeline/track.py

# Stage 4 for a single episode
stage4-one episode:
    {{python}} code/pipeline/track.py {{episode}}

# Stage 5 — Rule-based scene graph assembly (all stages → scene_graphs_pipeline/*.json)
stage5:
    {{python}} code/pipeline/sg_assemble.py

# Stage 5 for a single episode
stage5-one episode:
    {{python}} code/pipeline/sg_assemble.py {{episode}}

# Run full perception pipeline: Stage 1 → 2 → 3 → 4 → 5
full-pipeline: stage1 stage2 stage3 stage4 stage5

# Run full perception pipeline for a single episode, e.g: just full-pipeline-one boilWater-1
full-pipeline-one episode:
    {{python}} code/pipeline/detect.py {{episode}}
    {{python}} code/pipeline/segment.py {{episode}}
    {{python}} code/pipeline/depth_state.py {{episode}}
    {{python}} code/pipeline/track.py {{episode}}
    {{python}} code/pipeline/sg_assemble.py {{episode}}

# ── Ground-truth scene graphs ─────────────────────────────────────────────────

# Generate GT scene graph videos for all sim episodes (→ scene_graphs/*.mp4)
scene-graph:
    {{python}} code/scene_graph.py --all

# Generate GT scene graph video for a single sim episode, e.g: just scene-graph-one boilWater-1
scene-graph-one episode:
    {{python}} code/scene_graph.py {{episode}}

# Execute the exploration notebook in-place
notebook:
    poetry run jupyter nbconvert --to notebook --execute --inplace \
        notebooks/exploration.ipynb \
        --ExecutePreprocessor.timeout=300 \
        --ExecutePreprocessor.kernel_name=python3

# Launch the embedding visualizer GUI
gui:
    poetry run streamlit run code/gui.py

# ── Inspection ────────────────────────────────────────────────────────────────

# Inspect raw data layout
inspect:
    {{python}} code/inspect_data.py

# Run sanity checks on aligned output
sanity:
    {{python}} code/sanity_check.py

# ── Code quality ──────────────────────────────────────────────────────────────

# Lint all code
lint:
    poetry run ruff check code/

# Auto-fix lint issues
lint-fix:
    poetry run ruff check --fix code/

# Format all code
fmt:
    poetry run ruff format code/

# Lint + format in one pass
check: lint fmt
