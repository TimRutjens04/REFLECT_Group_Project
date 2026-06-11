from __future__ import annotations
import time
from pathlib import Path

from reflect_pipeline.data_loader.task_loader import Task
from reflect_pipeline.interfaces.IDetection import DetectionResult, ObjectDetector
from reflect_pipeline.interfaces.IFrameInput import RgbdFrame
from reflect_pipeline.detector.prompt_strategy import PromptPlan, PromptStrategy
from reflect_pipeline.models.base import JsonFrameWriter, JsonlWriter
from reflect_pipeline.models.detection import (
    Detection,
    DetectionFailureMode,
    DetectionFrame,
    TriggerReason,
)


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
        log_dir: Path | None = None,
        jsonl_writer: JsonlWriter | None = None,
    ):
        self.detector = detector
        self.strategy = strategy
        self.writer = JsonFrameWriter(log_dir) if log_dir else None
        self.jsonl_writer = jsonl_writer

    def run(
        self,
        frame: RgbdFrame,
        task: Task,
        trigger_reason: TriggerReason = TriggerReason.INIT,
    ) -> DetectionResult:
        plan = self.strategy.from_task(task)

        t0 = time.perf_counter()
        result, fallback_used = self._run_singles(frame, plan)
        if not result.success:
            result = self.detector.detect(frame, plan.primary)
            fallback_used = True
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        prompts_used = list(plan.singles)
        if fallback_used:
            prompts_used.append(plan.primary)

        if self.writer or self.jsonl_writer:
            df = self._to_detection_frame(
                frame=frame,
                task=task,
                result=result,
                trigger_reason=trigger_reason,
                prompts_used=prompts_used,
                runtime_ms=runtime_ms,
            )
            if self.writer:
                self.writer.write(df)
            if self.jsonl_writer:
                self.jsonl_writer.write(df)

        return result

    def _run_singles(
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
                False,
            )

        return (
            DetectionResult(
                detections=[],
                success=False,
                failure_reason=last_reason or "all single-object prompts failed",
                prompt_used=" | ".join(plan.singles),
            ),
            False,
        )

    @staticmethod
    def _to_detection_frame(
        frame: RgbdFrame,
        task: Task,
        result: DetectionResult,
        trigger_reason: TriggerReason,
        prompts_used: list[str],
        runtime_ms: float,
    ) -> DetectionFrame:
        timestamp = (
            frame.metadata.get("timestamp", float(frame.step_idx))
            if frame.metadata
            else float(frame.step_idx)
        )

        detections = [
            Detection(
                object_id=f"{obj.label}_{i}",
                label=obj.label,
                bbox_xyxy=obj.bbox_2d.tolist(),
                confidence=obj.score,
                is_selected=True,
            )
            for i, obj in enumerate(result.detections)
        ]

        failure_mode: DetectionFailureMode | None = None
        if not result.success and result.failure_reason:
            fr = result.failure_reason
            if "no detections" in fr or "prompts failed" in fr:
                failure_mode = DetectionFailureMode.NO_OBJECT
            elif "filtered out" in fr:
                failure_mode = DetectionFailureMode.LOW_CONFIDENCE
            else:
                failure_mode = DetectionFailureMode.NO_OBJECT

        return DetectionFrame(
            sequence_id=task.folder_name,
            frame_id=frame.step_idx,
            timestamp=timestamp,
            detector_ran=True,
            trigger_reason=trigger_reason,
            prompts_used=prompts_used,
            detections=detections,
            detection_success=result.success,
            failure_mode=failure_mode,
            runtime_ms=round(runtime_ms, 2),
        )
