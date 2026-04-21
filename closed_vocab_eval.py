import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

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
RESULTS_DIR = WORKDIR / "results"
HF_CACHE = WORKDIR / ".hf_cache"
TORCH_CACHE = WORKDIR / ".torch_cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE))

DEVICE = "cpu"
PRESENCE_THRESHOLD = 0.25
MAX_IMAGE_EDGE = 1280
SHARED_CLASSES = ["apple", "cup", "person"]
OPEN_VOCAB_ONLY_CLASSES: List[str] = []
BACKGROUND_CLASSES: List[str] = []
ALL_TRACKED_CLASSES = SHARED_CLASSES + OPEN_VOCAB_ONLY_CLASSES + BACKGROUND_CLASSES

YOLO_ID = "yolo11n.pt"
RTDETR_ID = "PekingU/rtdetr_v2_r18vd"
GDINO_ID = "IDEA-Research/grounding-dino-base"

MODEL_SOURCES = {
    "Grounding DINO Base": "https://huggingface.co/IDEA-Research/grounding-dino-base",
    "YOLO11n": "https://docs.ultralytics.com/models/yolo11/",
    "RT-DETRv2-R18": "https://huggingface.co/PekingU/rtdetr_v2_r18vd",
    "Faster R-CNN ResNet50 FPN V2": "https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html",
}


FRAME_LABELS = {
    "cup": {
        "IMG_0540.jpeg",
        "IMG_0541.jpeg",
        "IMG_0542.jpeg",
        "IMG_0543.jpeg",
        "IMG_0544.jpeg",
        "IMG_0545.jpeg",
        "IMG_0546.jpeg",
        "IMG_0547.jpeg",
        "IMG_0548.jpeg",
        "IMG_0549.jpeg",
        "IMG_0550.jpeg",
        "IMG_0551.jpeg",
        "IMG_0552.jpeg",
    },
    "person": {
        "IMG_0540.jpeg",
        "IMG_0541.jpeg",
        "IMG_0542.jpeg",
        "IMG_0543.jpeg",
        "IMG_0544.jpeg",
        "IMG_0545.jpeg",
        "IMG_0546.jpeg",
        "IMG_0547.jpeg",
        "IMG_0548.jpeg",
        "IMG_0549.jpeg",
        "IMG_0550.jpeg",
        "IMG_0551.jpeg",
        "IMG_0552.jpeg",
    },
    "apple": {
        "IMG_0528.jpeg",
        "IMG_0529.jpeg",
        "IMG_0530.jpeg",
        "IMG_0531.jpeg",
        "IMG_0532.jpeg",
        "IMG_0533.jpeg",
        "IMG_0534.jpeg",
        "IMG_0535.jpeg",
        "IMG_0538.jpeg",
        "IMG_0540.jpeg",
        "IMG_0541.jpeg",
        "IMG_0542.jpeg",
        "IMG_0543.jpeg",
        "IMG_0544.jpeg",
        "IMG_0545.jpeg",
        "IMG_0546.jpeg",
        "IMG_0547.jpeg",
        "IMG_0548.jpeg",
        "IMG_0550.jpeg",
        "IMG_0551.jpeg",
        "IMG_0552.jpeg",
    },
    "chair": {f"IMG_{i:04d}.jpeg" for i in list(range(528, 536)) + list(range(537, 553))},
    "dining table": {f"IMG_{i:04d}.jpeg" for i in list(range(528, 536)) + list(range(537, 553))},
    "potted plant": {f"IMG_{i:04d}.jpeg" for i in list(range(528, 536)) + list(range(537, 553))},
    "tv": {f"IMG_{i:04d}.jpeg" for i in list(range(528, 536)) + list(range(537, 553))},
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


def frame_sort_key(path: Path) -> int:
    return int(path.stem.split("_")[1])


def list_images() -> List[Path]:
    return sorted(DATA_DIR.glob("*.jpeg"), key=frame_sort_key)


def load_image(path: Path) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= MAX_IMAGE_EDGE:
        return image
    scale = MAX_IMAGE_EDGE / longest_edge
    resized = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    return resized


def average_precision_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    positives = int(y_true_arr.sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-y_score_arr, kind="mergesort")
    y_sorted = y_true_arr[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives

    ap = 0.0
    previous_recall = 0.0
    for p, r, truth in zip(precision, recall, y_sorted):
        if truth == 1:
            ap += float(p) * float(r - previous_recall)
            previous_recall = float(r)
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

    def score_image(self, image: Image.Image, classes: Sequence[str]) -> Dict[str, float]:
        prompt = " . ".join(classes) + " ."
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.05,
            text_threshold=0.05,
            target_sizes=[image.size[::-1]],
        )[0]
        row = {cls: 0.0 for cls in classes}
        if len(results["scores"]) == 0:
            return row

        labels = results["labels"]
        scores = results["scores"].cpu().tolist()
        for label, score in zip(labels, scores):
            normalized = str(label).lower().strip().strip(".")
            for cls in classes:
                if cls == normalized or cls in normalized:
                    row[cls] = max(row[cls], float(score))
        return row

    def run(self, image_paths: Sequence[Path], classes: Sequence[str]) -> ModelResult:
        rows: Dict[str, Dict[str, float]] = {}
        start = time.perf_counter()
        for path in image_paths:
            image = load_image(path)
            rows[path.name] = self.score_image(image, classes)
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
        self.categories = self.weights.meta["categories"]
        self.name_to_id = {name: idx for idx, name in enumerate(self.categories)}

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
        self.model = RTDetrV2ForObjectDetection.from_pretrained(self.source, local_files_only=local_only).to(DEVICE)
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


def evaluate_model(model_name: str, result: ModelResult, image_paths: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_rows = []
    summary_rows = []

    for cls in ALL_TRACKED_CLASSES:
        scores = [result.frame_scores[path.name].get(cls, 0.0) for path in image_paths]
        y_true = [int(path.name in FRAME_LABELS.get(cls, set())) for path in image_paths]

        for path, score, truth in zip(image_paths, scores, y_true):
            frame_rows.append(
                {
                    "model": model_name,
                    "frame": path.name,
                    "class": cls,
                    "score": score,
                    "truth": truth,
                    "predicted_present": int(score >= PRESENCE_THRESHOLD),
                }
            )

        base = classification_metrics(y_true, scores, PRESENCE_THRESHOLD)
        base["frame_ap"] = average_precision_binary(y_true, scores)
        base["latency_per_image_s"] = result.latency_per_image_s
        base["model"] = model_name
        base["class"] = cls
        base["source"] = MODEL_SOURCES[model_name]
        base["mean_score"] = float(np.mean(scores))
        summary_rows.append(base)

    return pd.DataFrame(frame_rows), pd.DataFrame(summary_rows)


def plot_shared_classes(frame_scores: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    x_positions = np.arange(len(sorted(frame_scores["frame"].unique())))

    for cls in SHARED_CLASSES:
        subset = frame_scores[frame_scores["class"] == cls].copy()
        plt.figure(figsize=(14, 5))
        for model_name, model_df in subset.groupby("model"):
            ordered = model_df.sort_values("frame", key=lambda s: s.map(lambda value: frame_sort_key(Path(value))))
            plt.plot(x_positions, ordered["score"].to_numpy(), marker="o", linewidth=1.8, label=model_name)

        truth_ordered = (
            subset[subset["model"] == subset["model"].iloc[0]]
            .sort_values("frame", key=lambda s: s.map(lambda value: frame_sort_key(Path(value))))
        )
        truth_values = truth_ordered["truth"].to_numpy()
        plt.fill_between(x_positions, 0, truth_values, color="#d8e8ff", alpha=0.35, label="Ground truth presence")
        plt.axhline(PRESENCE_THRESHOLD, color="black", linestyle="--", linewidth=1.0, label=f"Threshold {PRESENCE_THRESHOLD:.2f}")
        plt.xticks(x_positions, truth_ordered["frame"], rotation=90)
        plt.ylim(0, 1.05)
        plt.title(f"{cls.title()} confidence over frames")
        plt.ylabel("Max detection confidence")
        plt.tight_layout()
        plt.legend()
        plt.savefig(RESULTS_DIR / f"{cls}_confidence_trace.png", dpi=180, bbox_inches="tight")
        plt.close()


def plot_summary_bars(summary_df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    shared = summary_df[summary_df["class"].isin(SHARED_CLASSES)].copy()
    aggregate = (
        shared.groupby("model")[["frame_ap", "f1", "precision", "recall", "latency_per_image_s"]]
        .mean()
        .sort_values("frame_ap", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    aggregate["frame_ap"].plot(kind="bar", ax=axes[0], color="#4c72b0")
    axes[0].set_title("Mean frame-level AP on shared classes")
    axes[0].set_ylabel("AP")
    axes[0].set_ylim(0, 1.05)

    aggregate["latency_per_image_s"].plot(kind="bar", ax=axes[1], color="#55a868")
    axes[1].set_title("Mean latency per image")
    axes[1].set_ylabel("Seconds")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "summary_bars.png", dpi=180, bbox_inches="tight")
    plt.close()


def create_report(summary_df: pd.DataFrame) -> str:
    shared = summary_df[summary_df["class"].isin(SHARED_CLASSES)].copy()
    aggregate = (
        shared.groupby("model")[["frame_ap", "precision", "recall", "f1", "latency_per_image_s"]]
        .mean()
        .sort_values("frame_ap", ascending=False)
    )
    winner = aggregate.index[0]

    aggregate_table = aggregate.round(4).reset_index().to_csv(index=False).strip()

    lines = [
        "# Closed-vocabulary comparison on the occlusion sequence",
        "",
        "This report compares Grounding DINO against three efficient closed-vocabulary detectors on the 24 images in `data/`.",
        "Shared evaluation uses manual frame-level presence labels for `apple`, `cup`, and `person`, because the folder does not contain bounding-box ground truth.",
        "This sequence is a fair closed-vocabulary test bed because all three foreground classes are covered by the selected pretrained label spaces.",
        "",
        "## Winner on the shared closed-vocabulary task",
        "",
        f"Best overall closed-vocabulary model on this sequence: **{winner}**",
        "",
        "## Mean metrics on shared classes (`apple`, `cup`, `person`)",
        "",
        "```csv",
        aggregate_table,
        "```",
        "",
        "## Model rationale",
        "",
        "- Grounding DINO Base: open-vocabulary reference model.",
        "- YOLO11n: latest lightweight YOLO detect model with COCO-80 classes and low compute.",
        "- RT-DETRv2-R18: recent real-time transformer detector with a compact 20M-parameter footprint.",
        "- Faster R-CNN ResNet50 FPN V2: strong standard closed-vocabulary baseline from Torchvision.",
        "",
        "## Sources",
        "",
    ]

    for model_name, url in MODEL_SOURCES.items():
        lines.append(f"- {model_name}: {url}")

    return "\n".join(lines) + "\n"


def run_experiment() -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    image_paths = list_images()
    runners = {
        "Grounding DINO Base": GroundingDinoRunner(),
        "YOLO11n": YoloRunner(),
        "RT-DETRv2-R18": RtDetrV2Runner(),
        "Faster R-CNN ResNet50 FPN V2": FasterRcnnRunner(),
    }

    all_frame_scores = []
    all_summary = []

    for model_name, runner in runners.items():
        result = runner.run(image_paths, ALL_TRACKED_CLASSES)
        frame_df, summary_df = evaluate_model(model_name, result, image_paths)
        all_frame_scores.append(frame_df)
        all_summary.append(summary_df)

    frame_scores_df = pd.concat(all_frame_scores, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)

    frame_scores_df.to_csv(RESULTS_DIR / "frame_scores.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    plot_shared_classes(frame_scores_df)
    plot_summary_bars(summary_df)

    report = create_report(summary_df)
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")

    aggregate = (
        summary_df[summary_df["class"].isin(SHARED_CLASSES)]
        .groupby("model")[["frame_ap", "precision", "recall", "f1", "latency_per_image_s"]]
        .mean()
        .sort_values("frame_ap", ascending=False)
    )
    summary_payload = {
        "winner_shared_classes": aggregate.index[0],
        "shared_class_ranking": aggregate.reset_index().to_dict(orient="records"),
        "presence_threshold": PRESENCE_THRESHOLD,
        "shared_classes": SHARED_CLASSES,
        "open_vocab_only_classes": OPEN_VOCAB_ONLY_CLASSES,
        "background_classes": BACKGROUND_CLASSES,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_payload


if __name__ == "__main__":
    payload = run_experiment()
    print(json.dumps(payload, indent=2))
