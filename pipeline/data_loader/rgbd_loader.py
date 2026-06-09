from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import zarr

from data_loader.task_loader import Task
from interfaces.IFrameInput import RgbdFrame, RgbdFrameProvider


class VideoRgbdFrameProvider(RgbdFrameProvider):
    def __init__(self, task: Task):
        self.task_root = Path(task.task_root)
        self.color_path = self.task_root / "videos" / "color.mp4"
        self.depth_video = self.task_root / "videos" / "depth.mp4"
        self.depth_dir = self.task_root / "videos" / "depth"
        self.zarr_path = self.task_root / "replay_buffer.zarr"

        if not self.color_path.exists():
            raise FileNotFoundError(f"Color video missing: {self.color_path}")

        self._cap = cv2.VideoCapture(str(self.color_path))
        self._cap_next_idx = 0  # next frame the cap will return on read()

        self._depth_zarr_arr = None
        if self.depth_dir.exists() and (self.depth_dir / ".zarray").exists():
            try:
                self._depth_zarr_arr = zarr.open_array(str(self.depth_dir), mode="r")
            except Exception:
                self._depth_zarr_arr = None

    def __del__(self) -> None:
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()

    def get_frame(self, step_idx: int) -> RgbdFrame:
        rgb = self._read_rgb(step_idx)
        depth = self._read_depth(step_idx)
        return RgbdFrame(
            rgb=rgb,
            depth=depth,
            step_idx=step_idx,
            metadata={"task_root": str(self.task_root)},
        )

    @property
    def n_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _read_rgb(self, idx: int) -> np.ndarray:
        if idx != self._cap_next_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._cap_next_idx = idx
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read RGB frame {idx} from {self.color_path}")
        self._cap_next_idx += 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _read_depth(self, idx: int) -> np.ndarray:
        if self.depth_video.exists():
            cap = cv2.VideoCapture(str(self.depth_video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self._depth_zarr_arr is not None and idx < self._depth_zarr_arr.shape[0]:
            try:
                return self._to_2d_float(np.asarray(self._depth_zarr_arr[idx]))
            except Exception:
                pass

        if self.depth_dir.exists() and (self.depth_dir / ".zarray").exists():
            depth = self._read_depth_chunk(idx)
            if depth is not None:
                return depth

        if self.zarr_path.exists():
            zr = zarr.open_group(str(self.zarr_path), mode="r")
            for key in ("data/depth", "depth", "data/depth_video"):
                try:
                    return self._to_2d_float(np.array(zr[key][idx]))
                except (KeyError, IndexError):
                    continue

        raise RuntimeError(
            f"No depth source available for frame {idx} in {self.task_root}"
        )

    def _read_depth_chunk(self, idx: int) -> np.ndarray | None:
        from imagecodecs import imread as ic_imread

        candidates = [self.depth_dir / f"{idx}.0.0", self.depth_dir / f"{idx}.0.0.0"]
        candidates += list(self.depth_dir.rglob(f"{idx}.0.0*"))
        for p in candidates:
            if p.exists() and p.is_file():
                return self._to_2d_float(ic_imread(str(p)))
        return None

    @staticmethod
    def _to_2d_float(depth: np.ndarray) -> np.ndarray:
        if depth.ndim == 3 and depth.shape[-1] in (3, 4):
            depth = cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        return np.asarray(depth, dtype=np.float32)
