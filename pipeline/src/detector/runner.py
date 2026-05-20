from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from data_loader.task_loader import Task
from interfaces.IDetection import DetectionResult, ObjectDetector
from interfaces.IFrameInput import RgbdFrame
from detector.prompt_strategy import PromptPlan, PromptStrategy


@dataclass
class RunRecord:
    step_idx: int
    prompt_used: str
    success: bool
    failure_reason: str | None
    detections: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False


class DetectionRunner:
    """Orchestrates prompt selection + detection with multi→single fallback.

    The detector itself is intentionally dumb: one prompt in, one result out.
    This runner is the only component that knows about the fallback policy,
    which keeps the detector reusable for unit tests and other entry points.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        strategy: PromptStrategy,
        log_path: Path | None = None,
    ):
        self.detector = detector
        self.strategy = strategy
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")

    def run(self, frame: RgbdFrame, task: Task) -> DetectionResult:
        plan = self.strategy.from_task(task)

        result = self.detector.detect(frame, plan.primary)
        fallback_used = False

        if not result.success:
            result, fallback_used = self._fallback(frame, plan)

        self._log(frame.step_idx, result, fallback_used)
        return result

    def _fallback(
        self, frame: RgbdFrame, plan: PromptPlan
    ) -> tuple[DetectionResult, bool]:
        merged: list = []
        last_reason: str | None = None
        for single in plan.singles:
            r = self.detector.detect(frame, single)
            if r.success:
                merged.extend(r.detections)
            else:
                last_reason = r.failure_reason

        if merged:
            merged.sort(key=lambda d: d.score, reverse=True)
            return (
                DetectionResult(
                    detections=merged,
                    success=True,
                    prompt_used=" | ".join(plan.singles),
                ),
                True,
            )

        return (
            DetectionResult(
                detections=[],
                success=False,
                failure_reason=last_reason
                or "multi-object and all single-object prompts failed",
                prompt_used=plan.primary,
            ),
            True,
        )

    def _log(self, step_idx: int, result: DetectionResult, fallback_used: bool) -> None:
        if self.log_path is None:
            return
        record = RunRecord(
            step_idx=step_idx,
            prompt_used=result.prompt_used or "",
            success=result.success,
            failure_reason=result.failure_reason,
            detections=[
                {
                    "label": d.label,
                    "score": d.score,
                    "bbox_2d": d.bbox_2d.tolist(),
                    "prompt": d.prompt,
                    "n_alternatives": len(d.alternatives) if d.alternatives else 0,
                }
                for d in result.detections
            ],
            fallback_used=fallback_used,
        )
        with self.log_path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
