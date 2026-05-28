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

    def detect_with_embeddings(
        self,
        frame: RgbdFrame,
        prompt: str,
        context_labels: list[str] | None = None,
    ) -> tuple[DetectionResult, list[np.ndarray]]:
        """Like detect() but also returns a per-detection L2-normalized query embedding."""
        if self._model is None:
            raise RuntimeError("Call load() before detect_with_embeddings()")

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

        embeds_store: dict[str, torch.Tensor] = {}

        def _hook(_module: object, _inp: object, out: object) -> None:
            tensor = out[0] if isinstance(out, tuple) else out
            embeds_store["q"] = tensor.detach().cpu()  # (1, num_queries, hidden_dim)

        hook = self._model.model.decoder.layers[-1].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                outputs = self._model(**inputs)
        finally:
            hook.remove()

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self._score_thresh,
            text_threshold=self._score_thresh,
            target_sizes=[pil.size[::-1]],
        )[0]

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

        if not detections or "q" not in embeds_store:
            return DetectionResult(detections=detections), []

        # Match each detection to its query by nearest predicted-box center.
        # outputs.pred_boxes: (1, num_queries, 4) cxcywh normalized [0, 1]
        H, W = frame.rgb.shape[:2]
        pred_cxcy = outputs.pred_boxes[0, :, :2].cpu()  # (num_queries, 2)
        pred_cxcy_px = pred_cxcy * torch.tensor([W, H], dtype=torch.float32)
        query_embeds = embeds_store["q"][0]  # (num_queries, hidden_dim)

        embeddings: list[np.ndarray] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_2d
            det_cx = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2])
            dists = (pred_cxcy_px - det_cx).norm(dim=-1)  # (num_queries,)
            best_q = int(dists.argmin())
            e = query_embeds[best_q].numpy().astype(np.float32)
            norm = np.linalg.norm(e)
            embeddings.append(e / norm if norm > 0 else e)

        return DetectionResult(detections=detections), embeddings
