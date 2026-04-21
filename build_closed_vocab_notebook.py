import json
from pathlib import Path


WORKDIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = WORKDIR / "ClosedVocabComparison.ipynb"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


notebook = {
    "cells": [
        md_cell(
            "# Grounding DINO vs closed-vocabulary detectors\n\n"
            "This notebook compares the open-vocabulary reference model `Grounding DINO Base` against three efficient closed-vocabulary detectors on the 24 images in `data/`.\n\n"
            "Shared evaluation uses manual frame-level presence labels for `apple`, `cup`, and `person`."
        ),
        md_cell(
            "## Model choices\n\n"
            "- `YOLO11n`: latest lightweight Ultralytics detection model on COCO-80.\n"
            "- `RT-DETRv2-R18`: compact real-time transformer detector.\n"
            "- `Faster R-CNN ResNet50 FPN V2`: strong Torchvision closed-vocabulary baseline.\n"
            "- `Grounding DINO Base`: open-vocabulary reference.\n\n"
            "All choices were kept lightweight enough to stay practical on CPU-only hardware."
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import Markdown, display, Image\n"
            "\n"
            "from closed_vocab_eval import RESULTS_DIR, run_experiment\n"
        ),
        code_cell(
            "summary = run_experiment()\n"
            "summary"
        ),
        code_cell(
            "summary_df = pd.read_csv(RESULTS_DIR / 'summary_metrics.csv')\n"
            "shared = summary_df[summary_df['class'].isin(['apple', 'cup', 'person'])].copy()\n"
            "shared.groupby('model')[['frame_ap', 'precision', 'recall', 'f1', 'latency_per_image_s']].mean().sort_values('frame_ap', ascending=False)"
        ),
        code_cell(
            "display(Markdown((RESULTS_DIR / 'report.md').read_text(encoding='utf-8')))"
        ),
        code_cell(
            "for name in ['summary_bars.png', 'cup_confidence_trace.png', 'person_confidence_trace.png', 'apple_confidence_trace.png']:\n"
            "    path = RESULTS_DIR / name\n"
            "    if path.exists():\n"
            "        display(Markdown(f'### {name}'))\n"
            "        display(Image(filename=str(path)))"
        ),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(NOTEBOOK_PATH)
