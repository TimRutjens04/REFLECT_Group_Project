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

# Run CSRT tracking on all episodes (detect/ → track/)
track:
    {{python}} code/track.py

# Run CSRT tracking on a single episode, e.g: just track-one boilWater-1
track-one episode:
    {{python}} code/track.py {{episode}}

# Force-rerun CSRT tracking on a single episode
track-force-one episode:
    {{python}} code/track.py {{episode}} --force

# Render tracked bbox overlay video for all episodes (→ visuals/tracked/)
visualize-track:
    {{python}} code/visualize_track.py

# Render tracked bbox overlay video for a single episode
visualize-track-one episode:
    {{python}} code/visualize_track.py {{episode}}

# Annotate EEF tip in frames to compute camera-robot extrinsics (T_cam_robot.npy)
annotate-eef:
    {{python}} pipeline/annotations/annotate_eef.py

# Re-solve T_cam_robot from existing annotations without re-annotating
annotate-eef-solve:
    {{python}} pipeline/annotations/annotate_eef.py --solve-only

# Render scene-graph keyframe strip PNG for an episode, e.g:
#   just visualize-sg putAppleBowl1
#   just visualize-sg putAppleBowl1 /path/to/custom.jsonl
visualize-sg episode sg="":
    {{python}} pipeline/scene_graph/visualize_scene_graph.py \
        --sg "{{ if sg == "" { "scene_graphs/" + episode + "/scene_graph.jsonl" } else { sg } }}" \
        --video example_data/real_data/{{episode}}/videos/color.mp4 \
        --out visuals/scene_graphs/{{episode}}

# Render scene-graph keyframe MP4 (left=bbox, right=graph) for an episode, e.g:
#   just visualize-sg-mp4 putAppleBowl1
#   just visualize-sg-mp4 putAppleBowl1 /path/to/custom.jsonl 8
visualize-sg-mp4 episode sg="" fps="5":
    {{python}} pipeline/scene_graph/visualize_scene_graph.py \
        --sg "{{ if sg == "" { "scene_graphs/" + episode + "/scene_graph.jsonl" } else { sg } }}" \
        --video example_data/real_data/{{episode}}/videos/color.mp4 \
        --out visuals/scene_graphs/{{episode}} \
        --mp4 --fps {{fps}} --keyframes-only

# Benchmark CSRT vs ByteTrack vs ReID+Kalman on a single object (GDINO-driven), e.g:
#   just benchmark occluded.mp4 "cooking pot"
#   just benchmark occluded.mp4 "cooking pot" clean.mp4   (with GT companion video)
benchmark video prompt gt_video="":
    {{python}} code/benchmark/harness.py \
        --video "{{video}}" \
        --prompt "{{prompt}}" \
        $([ -n "{{gt_video}}" ] && echo "--gt-video {{gt_video}}")

# Benchmark N objects with manual bbox seed (no GDINO re-detection), e.g:
#   just benchmark-multi occluded.mp4 "mug 1. mug 2"
#   just benchmark-multi occluded.mp4 "mug 1. mug 2" clean.mp4 "coffee mug. coffee mug"
benchmark-multi video labels gt_video="" prompts="":
    {{python}} code/benchmark/harness_multi.py \
        --video "{{video}}" \
        --labels "{{labels}}" \
        $([ -n "{{gt_video}}" ] && echo "--gt-video {{gt_video}}") \
        $([ -n "{{prompts}}" ] && echo "--prompts {{prompts}}")

# Benchmark with GDINO pseudo-GT for numerical metrics (MOTA, MOTP, FPS, etc.), e.g:
#   just benchmark-num data/IMG_0081.MOV "coffee mug"
benchmark-num video label:
    {{python}} code/benchmark/harness_multi.py \
        --video "{{video}}" \
        --labels "{{label}}" \
        --detect-prompts "{{label}}" \
        --pseudo-gt "{{label}}"

# Draw a bbox on frame 0, benchmark all 3 trackers with GDINO re-detection, play live, e.g:
#   just benchmark-draw mug_video.mp4 "coffee mug"
benchmark-draw video label:
    {{python}} code/benchmark/harness_multi.py \
        --video "{{video}}" \
        --labels "{{label}}" \
        --detect-prompts "{{label}}" \
        --play

# Manually seed CSRT from a drawn bbox and track through a video (no GDINO), e.g:
#   just demo-track IMG_0081.MOV "coffee mug"
demo-track video label:
    {{python}} code/demo_track.py "{{video}}" --label "{{label}}"

# Track and visualize a single named object, e.g: just track-object boilWater-1 "apple"
track-object episode object:
    {{python}} code/track.py {{episode}} --object "{{object}}" --force
    {{python}} code/visualize_track.py {{episode}} --object "{{object}}"

# Track and visualize multiple named objects (GDINO-style period-separated), e.g:
#   just track-objects boilWater-1 "cooking pot. fridge. stove burner."
track-objects episode objects:
    {{python}} code/track.py {{episode}} --object "{{objects}}" --force
    {{python}} code/visualize_track.py {{episode}} --object "{{objects}}"

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

# Execute the exploration notebook in-place
notebook:
    poetry run jupyter nbconvert --to notebook --execute --inplace \
        notebooks/exploration.ipynb \
        --ExecutePreprocessor.timeout=300 \
        --ExecutePreprocessor.kernel_name=python3

# Launch the embedding visualizer GUI
gui:
    poetry run streamlit run code/gui.py

# ── Depth estimation ──────────────────────────────────────────────────────────

# Run Depth Anything V2 Metric Indoor (small) on a video, e.g:
#   just depth-video visuals/reid_can_demo.mp4
depth-video video:
    {{python}} code/depth_video.py "{{video}}"

# Same but with the large model for higher accuracy
depth-video-large video:
    {{python}} code/depth_video.py "{{video}}" --model large

# ── Inspection ────────────────────────────────────────────────────────────────

# Inspect raw data layout
inspect:
    {{python}} code/inspect_data.py

# Run sanity checks on aligned output
sanity:
    {{python}} code/sanity_check.py

# ── Tests ─────────────────────────────────────────────────────────────────────

# Run all unit tests
test:
    poetry run pytest tests/ -v

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
