import json
from pathlib import Path


WORKDIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = WORKDIR / "Stakeholder_ClosedVocab_Justification.ipynb"


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
            "# Open-vocabulary vs closed-vocabulary object detection\n\n"
            "This notebook is the stakeholder-facing justification for the model choice.\n\n"
            "The earlier notebook already compared the open-vocabulary models and selected **Grounding DINO** as the strongest open-vocabulary candidate. The purpose of this notebook is narrower and more critical: **does that choice still make sense when compared with strong closed-vocabulary detectors?**"
        ),
        md_cell(
            "## Reproducibility\n\n"
            "This notebook is designed to run from the same folder that contains:\n"
            "- `data/`\n"
            "- `stakeholder_eval.py`\n"
            "- `.hf_cache/` and `.torch_cache/` if you want the first run to avoid re-downloading weights\n\n"
            "If local caches are present, the code will use them. Otherwise the first run will download the pretrained weights from the official model repositories."
        ),
        code_cell(
            "import sys\n"
            "import subprocess\n"
            "packages = ['numpy', 'pandas', 'matplotlib', 'pillow', 'torch', 'torchvision', 'transformers', 'ultralytics']\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], check=False)\n"
        ),
        md_cell(
            "## Evaluation design\n\n"
            "The new data introduces two different questions, so the benchmark is split into two parts.\n\n"
            "### 1. Shared-class benchmark\n\n"
            "This is the fair comparison. It uses only classes that all models can detect with their native label space: **`apple`** and **`cup`**.\n\n"
            "### 2. Novel-class stress test\n\n"
            "This is the deployment relevance check. It uses **`pear`**, **`bucket`**, and **`egg`**, which appeared in the new images but are **not** part of the COCO label space used by the closed-vocabulary baselines.\n\n"
            "These two sections should not be mixed. The first tests raw detection quality on shared vocabulary. The second tests task flexibility when the requested object list changes."
        ),
        md_cell(
            "## Why these metrics\n\n"
            "- **Frame-level AP** is the primary metric because the dataset does not contain bounding-box annotations. It measures whether frames with the target object are ranked above frames without it.\n"
            "- **Precision / recall / F1** at a fixed threshold (`0.25`) show the practical operating trade-off between false positives and missed detections.\n"
            "- **Latency per image** matters because a deployment decision should consider runtime, not just accuracy.\n"
            "- **Exact label-space support** is reported separately because a closed-vocabulary detector cannot produce an exact class prediction if that class is absent from its pretrained taxonomy."
        ),
        md_cell(
            "## Critical scope note\n\n"
            "Bounding-box mAP was **not** used here. That was a deliberate choice, not an omission. The folder does not contain box annotations, so computing box mAP would first require building a new labeled dataset. That would change the task from model comparison to annotation production. The evaluation below instead focuses on the decision the stakeholder actually cares about: **which model is better suited to this evolving object-detection requirement?**"
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import Markdown, display, Image\n"
            "\n"
            "from stakeholder_eval import RESULTS_DIR, run_experiment\n"
        ),
        code_cell(
            "force_run = not (RESULTS_DIR / 'summary_metrics.csv').exists()\n"
            "summary = run_experiment(force=force_run)\n"
            "summary"
        ),
        md_cell("## Dataset overview"),
        code_cell(
            "display(Image(filename='data_contact_sheet_numbered.jpg'))"
        ),
        md_cell(
            "The sequence now contains:\n"
            "- the original office occlusion sequence,\n"
            "- new tabletop scenes with `pear`, `bucket`, and `apple`,\n"
            "- white-table scenes with `cup` and `egg`,\n"
            "- bright and low-light office scenes with changed viewpoint and lighting.\n\n"
            "This matters because it lets us separate **known-category robustness** from **vocabulary flexibility**."
        ),
        md_cell("## Exact label-space support"),
        code_cell(
            "support_df = pd.read_csv(RESULTS_DIR / 'class_support.csv')\n"
            "support_df"
        ),
        code_cell(
            "display(Image(filename=str(RESULTS_DIR / 'class_support.png')))"
        ),
        md_cell(
            "This table is important for interpretation. If a model does not contain an exact class such as `pear`, `bucket`, or `egg`, then any failure on that class is a **taxonomy limitation**, not only a detector-quality issue."
        ),
        md_cell("## Shared-class benchmark"),
        code_cell(
            "summary_df = pd.read_csv(RESULTS_DIR / 'summary_metrics.csv')\n"
            "shared_df = summary_df[summary_df['class'].isin(['apple', 'cup'])].copy()\n"
            "shared_df[['model', 'class', 'frame_ap', 'precision', 'recall', 'f1', 'latency_per_image_s', 'class_supported']].sort_values(['class', 'frame_ap', 'f1'], ascending=[True, False, False])"
        ),
        code_cell(
            "display(Image(filename=str(RESULTS_DIR / 'shared_summary.png')))\n"
            "display(Image(filename=str(RESULTS_DIR / 'shared_apple_trace.png')))\n"
            "display(Image(filename=str(RESULTS_DIR / 'shared_cup_trace.png')))"
        ),
        md_cell(
            "### Interpretation of the shared benchmark\n\n"
            "- This is the fairest accuracy comparison because all compared models are allowed to detect these classes.\n"
            "- If a closed-vocabulary model wins here, that does **not** automatically invalidate Grounding DINO. It only means that for a **fixed known taxonomy**, a specialized detector can be more efficient.\n"
            "- This section should be used to answer: *If the object list never changes, what is the best detector?*"
        ),
        md_cell("## Novel-class stress test"),
        code_cell(
            "novel_df = summary_df[summary_df['class'].isin(['pear', 'bucket', 'egg'])].copy()\n"
            "novel_df[['model', 'class', 'frame_ap', 'precision', 'recall', 'f1', 'latency_per_image_s', 'class_supported']].sort_values(['class', 'frame_ap', 'f1'], ascending=[True, False, False])"
        ),
        code_cell(
            "display(Image(filename=str(RESULTS_DIR / 'novel_summary.png')))\n"
            "display(Image(filename=str(RESULTS_DIR / 'novel_pear_trace.png')))\n"
            "display(Image(filename=str(RESULTS_DIR / 'novel_bucket_trace.png')))\n"
            "display(Image(filename=str(RESULTS_DIR / 'novel_egg_trace.png')))"
        ),
        md_cell(
            "### Interpretation of the novel-class stress test\n\n"
            "- This is **not** a fair closed-vocabulary benchmark. It is a task-fit benchmark.\n"
            "- The purpose is to test the exact stakeholder risk: what happens when new objects appear without retraining.\n"
            "- Using proxy labels such as `sports ball`, `bowl`, or `orange` would artificially improve closed-vocabulary models while changing the semantic task, so proxies were intentionally not used."
        ),
        md_cell("## Final conclusion"),
        code_cell(
            "report = Path(RESULTS_DIR / 'report.md').read_text(encoding='utf-8')\n"
            "display(Markdown(report))"
        ),
        md_cell(
            "## Recommendation wording for the stakeholder\n\n"
            "A careful conclusion is:\n\n"
            "**Closed-vocabulary baselines are competitive and in some cases better on shared known classes, especially when speed matters. However, the expanded dataset shows that the deployment requirement is not a fixed closed taxonomy. Because the object list changes across scenes, the final system still benefits from an open-vocabulary detector, and Grounding DINO remains the safer choice despite the higher runtime cost.**"
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
