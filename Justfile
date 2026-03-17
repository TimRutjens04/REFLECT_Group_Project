# REFLECT — RoboFail multimodal pipeline
# Usage: just <recipe>

set dotenv-load := false

python := "poetry run python3"

# List available recipes
default:
    @just --list

# ── Pipeline ──────────────────────────────────────────────────────────────────

# Run full pipeline: align → encode → analyze
all: align encode analyze

# Align all episodes (raw data → aligned/*.npz)
align:
    {{python}} code/align.py

# Align a single episode by name, e.g: just align-one makeSalad-5
align-one episode:
    {{python}} code/align.py {{episode}}

# Encode all aligned episodes (aligned/*.npz → encoded/*.npz)
encode:
    {{python}} code/encode.py aligned/

# Analyze all encoded episodes (encoded/*.npz → analysis/)
analyze:
    {{python}} code/analyze.py encoded/

# Execute the exploration notebook in-place
notebook:
    poetry run jupyter nbconvert --to notebook --execute --inplace \
        notebooks/exploration.ipynb \
        --ExecutePreprocessor.timeout=300 \
        --ExecutePreprocessor.kernel_name=python3

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
    poetry run ruff check code/ notebooks/

# Auto-fix lint issues
lint-fix:
    poetry run ruff check --fix code/ notebooks/

# Format all code
fmt:
    poetry run ruff format code/ notebooks/

# Lint + format in one pass
check: lint fmt
