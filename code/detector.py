from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from interfaces import DetectedObject, DetectionResult, ObjectDetector, RgbdFrame

# MPS lacks aten::_cummax_helper used by Grounding DINO's attention masking
GDINO_DEVICE = "cpu"


class GroundingDinoDetector(ObjectDetector):
    """Grounding DINO via HuggingFace transformers."""

    MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(
        self,
        score_thresh: float = 0.30,
        device: str = GDINO_DEVICE,
    ) -> None:
        self._score_thresh = score_thresh
        self._device       = device
        self._model        = None
        self._processor    = None

    def load(self) -> None:
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.MODEL_ID
        ).to(self._device)
        self._model.eval()

    @staticmethod
    def _normalize(label: str) -> str:
        return label.strip().rstrip(".").lower()

    def detect(
        self,
        frame: RgbdFrame,
        prompt: str,
        context_labels: list[str] | None = None,
    ) -> DetectionResult:
        if self._model is None:
            raise RuntimeError("Call load() before detect()")

        # Build multi-category prompt: "pot. stove burner. fridge."
        # Passing all scene categories lets GDINO disambiguate between them,
        # greatly reducing false positives (e.g. fridge scored as pot).
        primary_norm = self._normalize(prompt)
        seen: dict[str, None] = {primary_norm: None}
        parts = [primary_norm]
        for lbl in (context_labels or []):
            n = self._normalize(lbl)
            if n and n not in seen:
                seen[n] = None
                parts.append(n)
        text = ". ".join(parts) + "."

        pil    = Image.fromarray(frame.rgb)
        inputs = self._processor(
            images=pil, text=text, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self._score_thresh,
            text_threshold=self._score_thresh,
            target_sizes=[pil.size[::-1]],
        )[0]

        # Keep only detections whose matched text span is the primary object.
        detections = [
            DetectedObject(
                label=label,
                score=float(score),
                bbox_2d=box.cpu().numpy().astype(np.float32),
            )
            for box, score, label in zip(
                results["boxes"], results["scores"], results["text_labels"]
            )
            if self._normalize(label) == primary_norm
        ]
        detections.sort(key=lambda d: d.score, reverse=True)
        return DetectionResult(detections=detections)
