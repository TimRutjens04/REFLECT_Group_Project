# REFLECT — RoboFail multimodal pipeline
# Usage: just <recipe>

set dotenv-load := true

python := "uv run python3"

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

# Run OWL-ViT object detection on all real-world episodes (→ owl/*.npz)
owl:
    {{python}} code/owl_detect.py

# Run OWL-ViT for a single episode by name, e.g: just owl-one appleInFridge1
owl-one episode:
    {{python}} code/owl_detect.py {{episode}}

# Analyze all encoded episodes (encoded/*.npz → analysis/)
analyze:
    {{python}} code/analyze.py encoded/

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
