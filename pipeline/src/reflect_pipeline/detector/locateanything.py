from __future__ import annotations
import re
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from reflect_pipeline.interfaces.IDetection import DetectedObject, DetectionResult, ObjectDetector
from reflect_pipeline.interfaces.IFrameInput import RgbdFrame


# Box coordinates appear as four normalized integers in [0, coord_scale].
# Points (<box><x><y></box>) are intentionally NOT matched here.
_BOX_RE = re.compile(r"<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>")
# Strip any residual structural tokens (e.g. <null>, semantic-block markers).
_TOKEN_RE = re.compile(r"<[^>]*>")


@dataclass
class DetectorConfig:
    model_id: str = "nvidia/LocateAnything-3B"
    device: str | None = None  # auto-pick if None
    dtype: str = "bfloat16"
    generation_mode: str = "hybrid"  # "fast" (MTP) | "slow" (AR) | "hybrid"
    max_new_tokens: int = 2048  # raise toward 8192 for very dense scenes
    do_sample: bool = False  # greedy by default for reproducible detections
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    coord_scale: float = 1000.0  # model emits coords normalized to [0, 1000]
    # LocateAnything returns no per-box confidence; every kept box gets this.
    default_confidence: float = 1.0


class LocateAnythingDetector(ObjectDetector):
    """ObjectDetector backed by NVIDIA's LocateAnything-3B vision-language model.

    Unlike Grounding DINO this is a generative VLM: one image + one text prompt
    in, a token sequence out. We parse `<box>` coordinate tokens from that text
    and scale them to pixel space. The model does not emit per-box scores, so
    confidence is a fixed placeholder (see DetectorConfig.default_confidence).
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.device = None
        self.torch_dtype = None

    def load(self) -> None:
        if self.model is not None:
            return
        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.torch_dtype = getattr(torch, self.config.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
                self.config.model_id,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

    def detect(self, frame: RgbdFrame, prompt: str) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("Detector not loaded. Call load() first.")

        image = Image.fromarray(frame.rgb).convert("RGB")
        width, height = image.size

        answer = self._generate(image, prompt)
        detections = self._parse(answer, prompt, width, height)

        if not detections:
            return DetectionResult(
                detections=[],
                success=False,
                failure_reason=f"no detections in model output (mode={self.config.generation_mode})",
                prompt_used=prompt,
            )

        return DetectionResult(detections=detections, success=True, prompt_used=prompt)

    def _generate(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.torch_dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=self.tokenizer,
                max_new_tokens=self.config.max_new_tokens,
                use_cache=True,
                generation_mode=self.config.generation_mode,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                verbose=False,
            )
        return response[0] if isinstance(response, tuple) else response

    def _parse(
        self, answer: str, prompt: str, width: int, height: int
    ) -> list[DetectedObject]:
        fallback_label = self._query_from_prompt(prompt)
        scale = self.config.coord_scale

        out: list[DetectedObject] = []
        cursor = 0
        for m in _BOX_RE.finditer(answer):
            label = self._clean_label(answer[cursor : m.start()]) or fallback_label
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            bbox = np.array(
                [
                    x1 / scale * width,
                    y1 / scale * height,
                    x2 / scale * width,
                    y2 / scale * height,
                ],
                dtype=float,
            )
            out.append(
                DetectedObject(
                    label=label,
                    score=self.config.default_confidence,
                    bbox_2d=bbox,
                    prompt=prompt,
                )
            )
            cursor = m.end()
        return out

    @staticmethod
    def _clean_label(text: str) -> str:
        """Best-effort label for a box from the text segment preceding it.

        Heuristic: strip structural tokens / separators and keep the trailing
        phrase. For single-object prompts this is rarely needed because the
        fallback label (the query itself) is already unambiguous.
        """
        text = _TOKEN_RE.sub(" ", text)
        text = text.replace("</c>", " ")
        text = re.sub(r"\s+", " ", text).strip(" .,:;-")
        if not text:
            return ""
        return text.split(",")[-1].strip()

    @staticmethod
    def _query_from_prompt(prompt: str) -> str:
        """Extract the queried description from a prompt as a label fallback."""
        tail = prompt.rsplit(":", 1)[-1].strip().rstrip(".").strip()
        tail = tail.replace("</c>", ", ")
        return tail or "object"

    def is_confident(self, detection: DetectedObject) -> bool:
        # No native confidence from the model; everything kept is "confident".
        return True
