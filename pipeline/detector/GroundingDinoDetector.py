from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from interfaces.IDetection import DetectedObject, DetectionResult, ObjectDetector
from interfaces.IFrameInput import RgbdFrame


@dataclass
class DetectorConfig:
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    device: str | None = None  # auto-pick if None
    box_threshold: float = 0.4  # minimum score to keep a box
    accept_threshold: float = 0.1  # below this → flagged low-confidence
    text_threshold: float = 0.3
    max_alternatives: int = 3  # ranked alternatives per label


class GroundingDinoDetector(ObjectDetector):
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.processor = None
        self.model = None
        self.device = None

    def load(self) -> None:
        if self.model is not None:
            return
        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.model_id
        ).to(self.device)
        self.model.eval()

    def detect(self, frame: RgbdFrame, prompt: str) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("Detector not loaded. Call load() first.")

        image = Image.fromarray(frame.rgb)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        if len(results["scores"]) == 0:
            return DetectionResult(
                detections=[],
                success=False,
                failure_reason=f"no detections above box_threshold={self.config.box_threshold}",
                prompt_used=prompt,
            )

        detections = self._select_best_per_label(results, prompt)
        if not detections:
            return DetectionResult(
                detections=[],
                success=False,
                failure_reason="all detections filtered out during label grouping",
                prompt_used=prompt,
            )

        return DetectionResult(detections=detections, success=True, prompt_used=prompt)

    def _select_best_per_label(
        self, results: dict, prompt: str
    ) -> list[DetectedObject]:
        scores = results["scores"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()
        labels = (
            results.get("text_labels") or results.get("labels") or [""] * len(scores)
        )
        labels = [str(l) for l in labels]

        grouped: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
        for score, box, label in zip(scores, boxes, labels):
            grouped[label].append((float(score), box))

        out: list[DetectedObject] = []
        for label, items in grouped.items():
            items.sort(key=lambda x: x[0], reverse=True)
            best_score, best_box = items[0]
            alternatives = [
                DetectedObject(label=label, score=s, bbox_2d=b, prompt=prompt)
                for s, b in items[1 : 1 + self.config.max_alternatives]
            ]
            out.append(
                DetectedObject(
                    label=label,
                    score=best_score,
                    bbox_2d=best_box,
                    prompt=prompt,
                    alternatives=alternatives or None,
                )
            )
        out.sort(key=lambda d: d.score, reverse=True)
        return out

    def is_confident(self, detection: DetectedObject) -> bool:
        return detection.score >= self.config.accept_threshold
