#!/usr/bin/env python3
"""
Extract RGB frames from a task's color.mp4 and save as JPEG files.

Output location:
    <repo_root>/example_data/real_data/<folder_name>/frames/<frame_id:06d>.jpg

Usage:
    # By task ID (looked up in tasks_real_world.json):
    python dashboard/extract_frames.py --task-id 1

    # By folder name (takes priority if both supplied):
    python dashboard/extract_frames.py --folder-name putAppleBowl1

    # Overwrite existing frames:
    python dashboard/extract_frames.py --folder-name putAppleBowl1 --force

    # Override the default JSONL directory for the sanity check:
    python dashboard/extract_frames.py --folder-name putAppleBowl1 \\
        --jsonl-dir pipeline/real_world/jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-relative path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent


# Walk up from the script location until we find the repo root
# (a directory containing both `pipeline/` and `example_data/`).
def _find_repo_root(start: Path) -> Path:
    p = start
    while p.parent != p:
        if (p / "pipeline").is_dir() and (p / "example_data").is_dir():
            return p
        p = p.parent
    raise RuntimeError(
        f"Could not locate repo root from {start}. "
        "Expected a parent containing 'pipeline/' and 'example_data/'."
    )


REPO_ROOT    = _find_repo_root(SCRIPT_DIR)
PIPELINE_DIR = REPO_ROOT / "pipeline"

# Let Python find the pipeline modules (TaskLoader, VideoRgbdFrameProvider, …)
sys.path.insert(0, str(PIPELINE_DIR))

import cv2  # noqa: E402 — imported after sys.path patch

from data_loader.task_loader import Task, TaskLoader                    # noqa: E402
from data_loader.rgbd_loader import VideoRgbdFrameProvider              # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_task(task_id: int | None, folder_name: str | None) -> tuple[str, Path]:
    """
    Return (folder_name, task_root).
    folder_name wins over task_id if both are supplied.
    """
    data_dir = REPO_ROOT / "example_data"
    loader   = TaskLoader(data_dir)

    if folder_name:
        # Try to find a matching task in the JSON; fall back to path-only.
        for tid in loader.all_task_ids():
            t = loader.get(tid)
            if t.folder_name == folder_name:
                return t.folder_name, Path(t.task_root)
        # Not in JSON — construct path directly (useful for ad-hoc sequences)
        task_root = data_dir / "real_data" / folder_name
        print(
            f"[warn] '{folder_name}' not found in tasks_real_world.json — "
            f"using path {task_root} directly."
        )
        return folder_name, task_root

    if task_id is not None:
        t = loader.get(task_id)
        return t.folder_name, Path(t.task_root)

    raise ValueError("Provide --task-id or --folder-name.")


def _make_provider(task_root: Path) -> VideoRgbdFrameProvider:
    """Instantiate VideoRgbdFrameProvider from just a task_root path."""
    import types
    stub = types.SimpleNamespace(task_root=task_root)
    return VideoRgbdFrameProvider(stub)   # type: ignore[arg-type]


def _max_tracked_frame(folder_name: str, jsonl_dir: Path) -> int | None:
    """Return the highest frame_id in <folder_name>__tracking.jsonl, or None."""
    p = jsonl_dir / f"{folder_name}__tracking.jsonl"
    if not p.exists():
        return None
    max_id = 0
    with open(p) as f:
        for line in f:
            try:
                row = json.loads(line)
                max_id = max(max_id, int(row.get("frame_id", 0)))
            except Exception:
                continue
    return max_id if max_id else None


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract(
    folder_name: str,
    task_root: Path,
    force: bool,
    jsonl_dir: Path,
    jpeg_quality: int = 85,
    progress_every: int = 500,
) -> None:
    out_dir = task_root / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
    if existing and not force:
        print(
            f"[error] {len(existing)} file(s) already exist in {out_dir}.\n"
            "        Pass --force to overwrite."
        )
        sys.exit(1)

    provider = _make_provider(task_root)
    n_frames = provider.n_frames
    print(f"[info] {folder_name}: {n_frames} frames in {provider.color_path}")

    written = 0
    for idx in range(n_frames):
        try:
            rgb = provider._read_rgb(idx)
        except RuntimeError as e:
            print(f"[warn] frame {idx}: {e}")
            continue

        bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dest = out_dir / f"{idx:06d}.jpg"
        cv2.imwrite(str(dest), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        written += 1

        if written % progress_every == 0:
            print(f"  … {written}/{n_frames} frames written")

    print(f"[info] Wrote {written} frames → {out_dir}")

    # Sanity check: compare with highest tracked frame_id
    max_tracked = _max_tracked_frame(folder_name, jsonl_dir)
    if max_tracked is not None:
        if max_tracked >= written:
            print(
                f"[warn] Max tracked frame_id ({max_tracked}) ≥ extracted frame count ({written}). "
                "Some tracked frames may be outside the extracted range."
            )
        else:
            print(
                f"[ok] Max tracked frame_id ({max_tracked}) < extracted frame count ({written}). "
                "All tracked frames have a corresponding JPEG."
            )
    else:
        print("[info] No tracking JSONL found — skipping sanity check.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    default_jsonl = str(REPO_ROOT / "pipeline" / "real_world" / "jsonl")

    parser = argparse.ArgumentParser(
        description="Extract RGB frames from color.mp4 to <task_root>/frames/*.jpg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task-id",     type=int,  default=None,
                        help="Task ID from tasks_real_world.json")
    parser.add_argument("--folder-name", type=str,  default=None,
                        help="Sequence folder name (overrides --task-id if both supplied)")
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing frames (default: abort if frames exist)")
    parser.add_argument("--jsonl-dir",   type=str,  default=default_jsonl,
                        help=f"Directory containing pipeline JSONL files (default: {default_jsonl})")
    parser.add_argument("--quality",     type=int,  default=85,
                        help="JPEG quality 1–100 (default: 85)")
    args = parser.parse_args()

    if args.task_id is None and args.folder_name is None:
        parser.error("Provide at least one of --task-id or --folder-name.")

    folder_name, task_root = _resolve_task(args.task_id, args.folder_name)
    jsonl_dir = Path(args.jsonl_dir)

    extract(
        folder_name=folder_name,
        task_root=task_root,
        force=args.force,
        jsonl_dir=jsonl_dir,
        jpeg_quality=args.quality,
    )


if __name__ == "__main__":
    main()



    # python dashboard/extract_frames.py --folder-name putPearDrawer1
