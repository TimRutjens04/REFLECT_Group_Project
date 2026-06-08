"""
Per-object tracking log — accumulates frame-level records and exports to JSON.

Schema per frame:
{
  "sequence_id": "putAppleBowl1",
  "frame_id": 12,
  "timestamp": 12.0,
  "tracked_objects": [
    {
      "object_id": "apple_0",
      "label": "red apple",
      "bbox_xyxy": [x1, y1, x2, y2],
      "tracker_confidence": 0.91,
      "tracker_status": "ok",
      "flags": {
        "bbox_size_change_flag": false,
        "drift_flag": false,
        "recovery_trigger": false
      },
      "held_by_gripper": false,
      "last_detection_frame": 0
    }
  ]
}

tracker_status values:
  "ok"          — normal CSRT tracking, validator passed
  "frozen"      — validator flagged, holding last bbox, GDINO searching
  "searching"   — frozen, GDINO found nothing yet
  "recovered"   — re-acquired by GDINO, re-ID confirmed same object
  "redetected"  — re-acquired by GDINO, new ID assigned
  "held"        — gripper closed, GDINO suppressed
"""

from __future__ import annotations

import json
from collections import defaultdict


class TrackingLog:
    def __init__(self, sequence_id: str) -> None:
        self._sequence_id = sequence_id
        self._records: list[dict] = []

    def record(
        self,
        frame_id: int,
        timestamp: float,
        object_id: str,
        label: str,
        bbox_xyxy: list[int],
        tracker_confidence: float,
        tracker_status: str,
        bbox_size_change_flag: bool,
        drift_flag: bool,
        recovery_trigger: bool,
        held_by_gripper: bool,
        last_detection_frame: int,
    ) -> None:
        self._records.append({
            "sequence_id": self._sequence_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "object_id": object_id,
            "label": label,
            "bbox_xyxy": bbox_xyxy,
            "tracker_confidence": round(float(tracker_confidence), 4),
            "tracker_status": tracker_status,
            "flags": {
                "bbox_size_change_flag": bool(bbox_size_change_flag),
                "drift_flag": bool(drift_flag),
                "recovery_trigger": bool(recovery_trigger),
            },
            "held_by_gripper": bool(held_by_gripper),
            "last_detection_frame": int(last_detection_frame),
        })

    def frames(self) -> list[dict]:
        """Return records grouped by frame_id, matching the schema above."""
        grouped: dict[int, list[dict]] = defaultdict(list)
        for r in self._records:
            grouped[r["frame_id"]].append(r)

        result = []
        for frame_id in sorted(grouped):
            objs = grouped[frame_id]
            result.append({
                "sequence_id": self._sequence_id,
                "frame_id": frame_id,
                "timestamp": objs[0]["timestamp"],
                "tracked_objects": [
                    {
                        "object_id":            o["object_id"],
                        "label":                o["label"],
                        "bbox_xyxy":            o["bbox_xyxy"],
                        "tracker_confidence":   o["tracker_confidence"],
                        "tracker_status":       o["tracker_status"],
                        "flags":                o["flags"],
                        "held_by_gripper":      o["held_by_gripper"],
                        "last_detection_frame": o["last_detection_frame"],
                    }
                    for o in objs
                ],
            })
        return result

    def object_history(self, object_id: str) -> list[dict]:
        """Return all records for one object across all frames."""
        return [r for r in self._records if r["object_id"] == object_id]

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.frames(), f, indent=2)
