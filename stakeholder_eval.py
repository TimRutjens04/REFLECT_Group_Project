import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
)
from ultralytics import YOLO


WORKDIR = Path(__file__).resolve().parent
DATA_DIR = WORKDIR / "data"
RESULTS_DIR = WORKDIR / "stakeholder_results"
HF_CACHE = WORKDIR / ".hf_cache"
TORCH_CACHE = WORKDIR / ".torch_cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE))

DEVICE = "cpu"
MAX_IMAGE_EDGE = 1024
PRESENCE_THRESHOLD = 0.25

SHARED_CLASSES = ["apple", "cup"]
NOVEL_CLASSES = ["pear", "bucket", "egg"]
ALL_CLASSES = SHARED_CLASSES + NOVEL_CLASSES

PROMPTS = {
    "apple": "apple",
    "cup": "cup",
    "pear": "pear",
    "bucket": "metal bucket",
    "egg": "egg",
}

MODEL_SOURCES = {
    "Grounding DINO Base": "https://huggingface.co/IDEA-Research/grounding-dino-base",
    "YOLO11n": "https://docs.ultralytics.com/models/yolo11/",
    "RT-DETRv2-R18": "https://huggingface.co/PekingU/rtdetr_v2_r18vd",
    "Faster R-CNN ResNet50 FPN V2": "https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html",
}

MODEL_NOTES = {
    "Grounding DINO Base": "Open-vocabulary reference selected from the earlier notebook because it was the strongest among the tested open-vocabulary models.",
    "YOLO11n": "Lightweight one-stage closed-vocabulary baseline chosen for speed and practical deployment cost.",
    "RT-DETRv2-R18": "Compact transformer-based closed-vocabulary baseline to avoid comparing GDINO only against CNN detectors.",
    "Faster R-CNN ResNet50 FPN V2": "Strong two-stage closed-vocabulary baseline included because it is a widely used accuracy-oriented reference.",
}

SCENE_GROUPS = {
    "legacy_occlusion_sequence": list(range(1, 25)),
    "tabletop_novel_objects": list(range(25, 31)),
    "white_table_sequence": list(range(31, 38)),
    "bright_office_sequence": list(range(38, 49)),
    "low_light_sequence": list(range(49, 57)),
}

ANNOTATION_INDEXES = {
    "apple": [
        1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 50,
        52, 53, 54, 55, 56,
    ],
    "cup": [
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 31, 33, 34, 35,
        37, 38, 40, 41, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54, 55, 56,
    ],
    "pear": [25, 26, 27, 28, 30, 32],
    "bucket": [27, 28, 29, 30, 32],
    "egg": [31, 33, 36, 41],
}

YOLO_ID = "yolo11n.pt"
RTDETR_ID = "PekingU/rtdetr_v2_r18vd"
GDINO_ID = "IDEA-Research/grounding-dino-base"


def list_images() -> List[Path]:
    return sorted(
        [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )


IMAGE_PATHS = list_images()
IMAGE_INDEX_TO_NAME = {idx: path.name for idx, path in enumerate(IMAGE_PATHS, 1)}
FRAME_LABELS = {
    cls: {IMAGE_INDEX_TO_NAME[idx] for idx in indexes}
    for cls, indexes in ANNOTATION_INDEXES.items()
}


def hf_snapshot_dir(model_stub: str) -> Path | None:
    for cache_root in [HF_CACHE / "hub", Path.home() / ".cache" / "huggingface" / "hub"]:
        cache_dir = cache_root / model_stub
        ref_path = cache_dir / "refs" / "main"
        if ref_path.exists():
            return cache_dir / "snapshots" / ref_path.read_text(encoding="utf-8").strip()
    return None


def resolve_gdino_source() -> str:
    local = hf_snapshot_dir("models--IDEA-Research--grounding-dino-base")
    return str(local) if local and local.exists() else GDINO_ID


def resolve_rtdetr_source() -> str:
    local = hf_snapshot_dir("models--PekingU--rtdetr_v2_r18vd")
    return str(local) if local and local.exists() else RTDETR_ID


def load_image(path: Path) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= MAX_IMAGE_EDGE:
        return image
    scale = MAX_IMAGE_EDGE / longest_edge
    return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)


def exact_class_support() -> pd.DataFrame:
    yolo_model = YOLO(YOLO_ID)
    yolo_names = set(yolo_model.names.values())

    frcnn_categories = set(FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"])

    rtdetr_source = resolve_rtdetr_source()
    rtdetr_model = RTDetrV2ForObjectDetection.from_pretrained(
        rtdetr_source,
        local_files_only=rtdetr_source != RTDETR_ID,
    )
    rtdetr_categories = set(rtdetr_model.config.id2label.values())

    rows = []
    for cls in ALL_CLASSES:
        rows.append(
            {
                "class": cls,
                "Grounding DINO Base": 1,
                "YOLO11n": int(cls in yolo_names),
                "RT-DETRv2-R18": int(cls in rtdetr_categories),
                "Faster R-CNN ResNet50 FPN V2": int(cls in frcnn_categories),
            }
        )
    return pd.DataFrame(rows)


def safe_average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    positives = int(y_true_arr.sum())
    if positives == 0:
        return float("nan")
    if np.max(y_score_arr) <= 0:
        return 0.0

    order = np.argsort(-y_score_arr, kind="mergesort")
    y_sorted = y_true_arr[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives

    ap = 0.0
    previous_recall = 0.0
    for precision_at_k, recall_at_k, truth in zip(precision, recall, y_sorted):
        if truth == 1:
            ap += float(precision_at_k) * float(recall_at_k - previous_recall)
            previous_recall = float(recall_at_k)
    return ap


def classification_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = (np.asarray(y_score, dtype=float) >= threshold).astype(int)

    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@dataclass
class ModelResult:
    frame_scores: Dict[str, Dict[str, float]]
    latency_per_image_s: float


class GroundingDinoRunner:
    def __init__(self) -> None:
        self.source = resolve_gdino_source()
        self.processor = AutoProcessor.from_pretrained(
            self.source,
            local_files_only=self.source != GDINO_ID,
            use_fast=False,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.source,
            local_files_only=self.source != GDINO_ID,
        ).to(DEVICE)
        self.model.eval()

    def run(self, image_paths: Sequence[Path], classes: Sequence[str]) -> ModelResult:
        prompt_text = " . ".join(PROMPTS[cls] for cls in classes) + " ."
        rows: Dict[str, Dict[str, float]] = {}
        start = time.perf_counter()

        for path in image_paths:
            image = load_image(path)
            inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.05,
                text_threshold=0.05,
                target_sizes=[image.size[::-1]],
            )[0]
            row = {cls: 0.0 for cls in classes}
            label_list = processed.get("text_labels", processed.get("labels", []))
            score_list = processed["scores"].cpu().tolist() if len(processed["scores"]) else []
            for raw_label, score in zip(label_list, score_list):
                label = str(raw_label).lower().strip().strip(".")
                for cls in classes:
                    prompt_label = PROMPTS[cls].lower()
                    if cls == "bucket":
                        matched = ("bucket" in label) or ("pail" in label)
                    else:
                        matched = (cls in label) or (prompt_label in label)
                    if matched:
                        row[cls] = max(row[cls], float(score))
            rows[path.name] = row

        total = time.perf_counter() - start
        return ModelResult(frame_scores=rows, latency_per_image_s=total / max(len(image_paths), 1))


class YoloRunner:
    def __init__(self) -> None:
        self.model = YOLO(YOLO_ID)
        self.name_to_id = {name: idx for idx, name in self.model.names.items()}

    def run(self, image_paths: Sequence[Path], classes: Sequence[str]) -> ModelResult:
        rows: Dict[str, Dict[str, float]] = {}
        start = time.perf_counter()

        for path in image_paths:
            result = self.model.predict(str(path), conf=0.01, verbose=False)[0]
            row = {cls: 0.0 for cls in classes}
            if result.boxes is not None and len(result.boxes) > 0:
                labels = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()
                for cls in classes:
                    cls_id = self.name_to_id.get(cls)
                    if cls_id is None:
                        continue
                    cls_scores = scores[labels == cls_id]
                    row[cls] = float(cls_scores.max()) if len(cls_scores) else 0.0
            rows[path.name] = row

        total = time.perf_counter() - start
        return ModelResult(frame_scores=rows, latency_per_image_s=total / max(len(image_paths), 1))


class FasterRcnnRunner:
    def __init__(self) -> None:
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights).to(DEVICE)
        self.model.eval()
        self.transforms = self.weights.transforms()
        self.name_to_id = {name: idx for idx, name in enumerate(self.weights.meta["categories"])}

    def run(self, image_paths: Sequence[Path], classes: Sequence[str]) -> ModelResult:
        rows: Dict[str, Dict[str, float]] = {}
        start = time.perf_counter()

        for path in image_paths:
            image = load_image(path)
            tensor = self.transforms(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = self.model(tensor)[0]
            labels = output["labels"].cpu().numpy().astype(int)
            scores = output["scores"].cpu().numpy()
            row = {cls: 0.0 for cls in classes}
            for cls in classes:
                cls_id = self.name_to_id.get(cls)
                if cls_id is None:
                    continue
                cls_scores = scores[labels == cls_id]
                row[cls] = float(cls_scores.max()) if len(cls_scores) else 0.0
            rows[path.name] = row

        total = time.perf_counter() - start
        return ModelResult(frame_scores=rows, latency_per_image_s=total / max(len(image_paths), 1))


class RtDetrV2Runner:
    def __init__(self) -> None:
        self.source = resolve_rtdetr_source()
        local_only = self.source != RTDETR_ID
        self.processor = RTDetrImageProcessor.from_pretrained(self.source, local_files_only=local_only)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(
            self.source,
            local_files_only=local_only,
        ).to(DEVICE)
        self.model.eval()
        self.name_to_id = {label: int(idx) for idx, label in self.model.config.id2label.items()}

    def run(self, image_paths: Sequence[Path], classes: Sequence[str]) -> ModelResult:
        rows: Dict[str, Dict[str, float]] = {}
        start = time.perf_counter()

        for path in image_paths:
            image = load_image(path)
            inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
            processed = self.processor.post_process_object_detection(
                outputs,
                threshold=0.01,
                target_sizes=torch.tensor([image.size[::-1]], device=DEVICE),
            )[0]
            labels = processed["labels"].cpu().numpy().astype(int)
            scores = processed["scores"].cpu().numpy()
            row = {cls: 0.0 for cls in classes}
            for cls in classes:
                cls_id = self.name_to_id.get(cls)
                if cls_id is None:
                    continue
                cls_scores = scores[labels == cls_id]
                row[cls] = float(cls_scores.max()) if len(cls_scores) else 0.0
            rows[path.name] = row

        total = time.perf_counter() - start
        return ModelResult(frame_scores=rows, latency_per_image_s=total / max(len(image_paths), 1))


def group_name(frame_name: str) -> str:
    frame_index = next(idx for idx, name in IMAGE_INDEX_TO_NAME.items() if name == frame_name)
    for scene, indices in SCENE_GROUPS.items():
        if frame_index in indices:
            return scene
    return "unassigned"


def evaluate_model(
    model_name: str,
    result: ModelResult,
    image_paths: Sequence[Path],
    class_support: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support_lookup = {
        (row["class"], col): int(row[col])
        for _, row in class_support.iterrows()
        for col in ["Grounding DINO Base", "YOLO11n", "RT-DETRv2-R18", "Faster R-CNN ResNet50 FPN V2"]
    }

    frame_rows = []
    summary_rows = []

    for cls in ALL_CLASSES:
        scores = [result.frame_scores[path.name].get(cls, 0.0) for path in image_paths]
        truths = [int(path.name in FRAME_LABELS.get(cls, set())) for path in image_paths]
        support = support_lookup[(cls, model_name)]

        for path, score, truth in zip(image_paths, scores, truths):
            frame_rows.append(
                {
                    "model": model_name,
                    "frame": path.name,
                    "scene_group": group_name(path.name),
                    "class": cls,
                    "score": score,
                    "truth": truth,
                    "predicted_present": int(score >= PRESENCE_THRESHOLD),
                    "class_supported": support,
                }
            )

        metrics = classification_metrics(truths, scores, PRESENCE_THRESHOLD)
        metrics["frame_ap"] = safe_average_precision(truths, scores)
        metrics["latency_per_image_s"] = result.latency_per_image_s
        metrics["mean_score"] = float(np.mean(scores))
        metrics["model"] = model_name
        metrics["class"] = cls
        metrics["class_supported"] = support
        metrics["source"] = MODEL_SOURCES[model_name]
        summary_rows.append(metrics)

    return pd.DataFrame(frame_rows), pd.DataFrame(summary_rows)


def aggregate_metrics(summary_df: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    subset = summary_df[summary_df["class"].isin(classes)].copy()
    return (
        subset.groupby("model")[["frame_ap", "precision", "recall", "f1", "latency_per_image_s"]]
        .mean()
        .sort_values(["frame_ap", "f1"], ascending=False)
    )


def plot_metric_bars(aggregate_df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    aggregate_df["frame_ap"].plot(kind="bar", ax=axes[0], color="#2b6cb0")
    axes[0].set_title(f"{title}: frame-level AP")
    axes[0].set_ylabel("AP")
    axes[0].set_ylim(0, 1.05)

    aggregate_df["latency_per_image_s"].plot(kind="bar", ax=axes[1], color="#2f855a")
    axes[1].set_title(f"{title}: latency per image")
    axes[1].set_ylabel("Seconds")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_support_matrix(support_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    matrix = support_df.set_index("class")
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Exact label-space support")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, int(matrix.iloc[y, x]), va="center", ha="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_confidence_traces(frame_scores_df: pd.DataFrame, classes: Sequence[str], prefix: str) -> None:
    ordered_frames = [path.name for path in IMAGE_PATHS]
    x_positions = np.arange(len(ordered_frames))
    frame_numbers = np.arange(1, len(ordered_frames) + 1)

    for cls in classes:
        subset = frame_scores_df[frame_scores_df["class"] == cls].copy()
        plt.figure(figsize=(15, 4.8))
        for model_name, model_df in subset.groupby("model"):
            ordered = model_df.set_index("frame").loc[ordered_frames]
            plt.plot(x_positions, ordered["score"].to_numpy(), marker="o", linewidth=1.5, label=model_name)

        truth = subset[subset["model"] == subset["model"].iloc[0]].set_index("frame").loc[ordered_frames]["truth"].to_numpy()
        plt.fill_between(x_positions, 0, truth, color="#e6f0ff", alpha=0.35, label="Ground-truth presence")
        plt.axhline(PRESENCE_THRESHOLD, color="black", linestyle="--", linewidth=1.0, label=f"Threshold {PRESENCE_THRESHOLD:.2f}")
        tick_step = max(1, len(frame_numbers) // 12)
        tick_positions = x_positions[::tick_step]
        tick_labels = [str(n) for n in frame_numbers[::tick_step]]
        plt.xticks(tick_positions, tick_labels)
        plt.ylim(0, 1.05)
        plt.title(f"{cls.title()} confidence across the sequence")
        plt.ylabel("Max confidence")
        plt.xlabel("Frame index")
        plt.tight_layout()
        plt.legend()
        plt.savefig(RESULTS_DIR / f"{prefix}_{cls}_trace.png", dpi=180, bbox_inches="tight")
        plt.close()


def build_manifest() -> pd.DataFrame:
    rows = []
    for idx, path in enumerate(IMAGE_PATHS, 1):
        rows.append(
            {
                "index": idx,
                "filename": path.name,
                "scene_group": group_name(path.name),
                **{cls: int(path.name in FRAME_LABELS.get(cls, set())) for cls in ALL_CLASSES},
            }
        )
    return pd.DataFrame(rows)


def summary_report(
    shared_agg: pd.DataFrame,
    novel_agg: pd.DataFrame,
    support_df: pd.DataFrame,
) -> str:
    winner_shared = shared_agg.index[0]
    fastest_shared = shared_agg["latency_per_image_s"].idxmin()

    lines = [
        "# Stakeholder comparison: open-vocabulary vs closed-vocabulary detectors",
        "",
        "## What this notebook tests",
        "",
        "- A fair shared-class benchmark on `apple` and `cup`, because those classes are supported by all compared models.",
        "- A novel-class stress test on `pear`, `bucket`, and `egg`, because these appeared in the updated data but are outside the standard COCO label space used by the closed-vocabulary baselines.",
        "- A practical deployment trade-off: ranking quality, operational precision/recall at one threshold, and inference latency.",
        "",
        "## Why these metrics were used",
        "",
        "- `Frame-level AP` is the primary ranking metric because we do not have bounding-box annotations. It tests whether frames containing the target object are ranked above frames that do not.",
        "- `Precision`, `recall`, and `F1` at threshold 0.25 show the operational trade-off between false alarms and misses.",
        "- `Latency per image` matters because a detector that wins slightly on accuracy but is much slower may be a poor deployment choice.",
        "- Exact label-space support is reported separately, because a closed-vocabulary detector can only detect classes that exist in its pretrained taxonomy.",
        "",
        "## Critical interpretation",
        "",
        f"- Best closed-vocabulary model on the shared benchmark: **{winner_shared}**.",
        f"- Fastest model on the shared benchmark: **{fastest_shared}**.",
        "- The shared benchmark tells us how well each model handles known categories under occlusion, viewpoint change, and lighting change.",
        "- The novel-class benchmark answers a different question: what happens when the task vocabulary changes without retraining.",
        "- A closed-vocabulary model scoring zero on `pear`, `bucket`, or `egg` is not a bug in the metric. It reflects a real deployment limitation: the requested class does not exist in the model label set.",
        "",
        "## Shared benchmark aggregate",
        "",
        "```csv",
        shared_agg.round(4).reset_index().to_csv(index=False).strip(),
        "```",
        "",
        "## Novel-class aggregate",
        "",
        "```csv",
        novel_agg.round(4).reset_index().to_csv(index=False).strip(),
        "```",
        "",
        "## Exact label-space support",
        "",
        "```csv",
        support_df.to_csv(index=False).strip(),
        "```",
        "",
        "## Outcome",
        "",
        "- If the vocabulary is fixed to a known set such as `apple` and `cup`, a closed-vocabulary detector is competitive and may even be preferable for speed.",
        "- If the deployment requires adapting to new requested objects without retraining, the open-vocabulary reference remains the safer choice.",
        "- This means the final recommendation depends on the real system requirement: fixed taxonomy favors a closed-vocabulary baseline, evolving task vocabulary favors Grounding DINO.",
        "",
        "## Sources",
        "",
    ]
    for model_name, url in MODEL_SOURCES.items():
        lines.append(f"- {model_name}: {url}")
    return "\n".join(lines) + "\n"


def run_experiment(force: bool = True) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)

    manifest_df = build_manifest()
    manifest_df.to_csv(RESULTS_DIR / "dataset_manifest.csv", index=False)

    support_df = exact_class_support()
    support_df.to_csv(RESULTS_DIR / "class_support.csv", index=False)
    plot_support_matrix(support_df, RESULTS_DIR / "class_support.png")

    if not force:
        frame_scores_path = RESULTS_DIR / "frame_scores.csv"
        summary_path = RESULTS_DIR / "summary_metrics.csv"
        if frame_scores_path.exists() and summary_path.exists():
            frame_scores_df = pd.read_csv(frame_scores_path)
            summary_df = pd.read_csv(summary_path)
        else:
            force = True

    if force:
        runners = {
            "Grounding DINO Base": GroundingDinoRunner(),
            "YOLO11n": YoloRunner(),
            "RT-DETRv2-R18": RtDetrV2Runner(),
            "Faster R-CNN ResNet50 FPN V2": FasterRcnnRunner(),
        }

        frame_parts = []
        summary_parts = []
        for model_name, runner in runners.items():
            result = runner.run(IMAGE_PATHS, ALL_CLASSES)
            frame_df, summary_df = evaluate_model(model_name, result, IMAGE_PATHS, support_df)
            frame_parts.append(frame_df)
            summary_parts.append(summary_df)

        frame_scores_df = pd.concat(frame_parts, ignore_index=True)
        summary_df = pd.concat(summary_parts, ignore_index=True)
        frame_scores_df.to_csv(RESULTS_DIR / "frame_scores.csv", index=False)
        summary_df.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

    plot_confidence_traces(frame_scores_df, SHARED_CLASSES, "shared")
    plot_confidence_traces(frame_scores_df, NOVEL_CLASSES, "novel")

    shared_agg = aggregate_metrics(summary_df, SHARED_CLASSES)
    novel_agg = aggregate_metrics(summary_df, NOVEL_CLASSES)
    shared_agg.reset_index().to_csv(RESULTS_DIR / "shared_aggregate.csv", index=False)
    novel_agg.reset_index().to_csv(RESULTS_DIR / "novel_aggregate.csv", index=False)

    plot_metric_bars(shared_agg, "Shared classes", RESULTS_DIR / "shared_summary.png")
    plot_metric_bars(novel_agg, "Novel classes", RESULTS_DIR / "novel_summary.png")

    report = summary_report(shared_agg, novel_agg, support_df)
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")

    payload = {
        "shared_classes": SHARED_CLASSES,
        "novel_classes": NOVEL_CLASSES,
        "presence_threshold": PRESENCE_THRESHOLD,
        "shared_winner": shared_agg.index[0],
        "shared_fastest": shared_agg["latency_per_image_s"].idxmin(),
        "shared_ranking": shared_agg.reset_index().to_dict(orient="records"),
        "novel_ranking": novel_agg.reset_index().to_dict(orient="records"),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    summary = run_experiment(force=True)
    print(json.dumps(summary, indent=2))
