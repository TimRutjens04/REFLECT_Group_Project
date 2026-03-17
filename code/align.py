"""
Temporal alignment pipeline — REFLECT / RoboFail dataset.

Produces synchronized (frame, audio_window) pairs at a fixed base rate,
saved as .npz files ready for CLIP / WAV2CLIP encoding.

=== CONFIRMED DATA FACTS (from inspect_data.py) ===

  boilWater / makeSalad (sim):
    Video:      original-video.mp4  — 1fps, 960x960, step-indexed
    Audio:      embedded in MP4     — 44100 Hz, stereo, float64
    Frames:     ego_img/img_step_N.png — 960x960 RGBA, pre-extracted
    Failure ts: task.json gt_failure_step — MM:SS string or list
    Base rate:  1fps  (video is natively 1fps; cannot upsample without duplication)

  putFruitsBowl (real-world):
    Video:      videos/color.mp4    — 30fps, 1280x720, 121s
    Audio:      videos/audio.wav    — 48000 Hz, mono, float32, 121s
    Zarr:       replay_buffer.zarr  — timestamps, joints, gripper, EEF (3639 steps)
    Failure ts: tasks_real_world.json gt_failure_step — MM:SS string or list
    Base rate:  2fps

  NOTE: sim data runs at 1fps (one frame per action step); CLAUDE.md target of
  2fps applies to real-world continuous video. The script uses each dataset's
  natural rate rather than duplicating sim frames.

=== OUTPUT FORMAT (per episode) ===

  aligned/<episode_id>.npz
    timestamps      (n_frames,)         float64 — seconds from episode start
    frames          (n_frames, H, W, 3) uint8   — RGB
    audio_windows   (n_frames, n_samples) float32 — mono, zero-padded at edges
    failure_labels  (n_frames,)         bool
    fps_base        scalar float
    audio_sr        scalar int
"""

import json
import os
import sys
import warnings

import cv2
import librosa
import numpy as np
from tqdm import tqdm

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "aligned"))


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def parse_mm_ss(ts: str) -> float:
    """Convert 'MM:SS' string to float seconds."""
    mm, ss = ts.strip().split(":")
    return int(mm) * 60 + int(ss)


def get_failure_timestamps(meta: dict) -> list[float]:
    """Return list of failure timestamps in seconds from a task metadata dict."""
    raw = meta.get("gt_failure_step", None)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [parse_mm_ss(raw)]
    # list of MM:SS strings
    return [parse_mm_ss(t) for t in raw]


def failure_labels_for_timestamps(timestamps: np.ndarray, failure_ts: list[float],
                                   threshold: float = 0.5) -> np.ndarray:
    """Return bool array: True if any failure timestamp is within threshold seconds."""
    labels = np.zeros(len(timestamps), dtype=bool)
    for ft in failure_ts:
        labels |= np.abs(timestamps - ft) < threshold
    return labels


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def extract_audio_window(audio: np.ndarray, sr: int, t_center: float,
                          duration: float = 0.5) -> np.ndarray:
    """
    Extract a window of `duration` seconds centered on t_center from mono audio.
    Zero-pads if the window extends before start or past end of the signal.
    """
    half = duration / 2.0
    start_sample = int((t_center - half) * sr)
    end_sample = int((t_center + half) * sr)
    n_samples = end_sample - start_sample

    window = np.zeros(n_samples, dtype=np.float32)

    src_start = max(0, start_sample)
    src_end = min(len(audio), end_sample)
    if src_start >= src_end:
        return window  # fully out of bounds → silence

    dst_start = src_start - start_sample
    dst_end = dst_start + (src_end - src_start)
    window[dst_start:dst_end] = audio[src_start:src_end]
    return window


def audio_windows_for_timestamps(audio: np.ndarray, sr: int,
                                   timestamps: np.ndarray,
                                   window_duration: float = 0.5) -> np.ndarray:
    """Vectorised helper: extract one window per timestamp."""
    n_samples = int(sr * window_duration)
    windows = np.zeros((len(timestamps), n_samples), dtype=np.float32)
    for i, t in enumerate(timestamps):
        w = extract_audio_window(audio, sr, t, window_duration)
        # w may differ by 1 sample due to int truncation — trim/pad to exact size
        if len(w) >= n_samples:
            windows[i] = w[:n_samples]
        else:
            windows[i, : len(w)] = w
    return windows


# ---------------------------------------------------------------------------
# Frame extraction helpers
# ---------------------------------------------------------------------------

def load_sim_frames(episode_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-extracted PNG frames from ego_img/ (step-indexed, native 1fps).

    Returns:
        timestamps  (n_frames,) float64 — seconds (0-indexed: frame N at t=N)
        frames      (n_frames, H, W, 3) uint8 — RGB
    """
    img_dir = os.path.join(episode_dir, "ego_img")
    files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith(".png")],
        key=lambda f: int(f.replace("img_step_", "").replace(".png", ""))
    )
    frames = []
    for fname in files:
        img = cv2.imread(os.path.join(img_dir, fname), cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    frames_arr = np.array(frames, dtype=np.uint8)
    # Step N (1-indexed) → timestamp = (N-1) / 1fps
    n = len(files)
    step_indices = np.array(
        [int(f.replace("img_step_", "").replace(".png", "")) for f in files]
    )
    timestamps = (step_indices - 1).astype(np.float64)  # t=0 at step 1
    return timestamps, frames_arr


def extract_real_frames_2fps(video_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract frames from a continuous video at 2fps.

    Returns:
        timestamps  (n_frames,) float64 — seconds
        frames      (n_frames, H, W, 3) uint8 — RGB
    """
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = 2.0
    step = native_fps / target_fps  # sample every `step` native frames

    frames = []
    timestamps = []
    idx = 0  # logical 2fps index

    while True:
        native_idx = int(idx * step)
        if native_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, native_idx)
        ret, frame = cap.read()
        if not ret:
            break
        timestamps.append(native_idx / native_fps)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1

    cap.release()
    return np.array(timestamps, dtype=np.float64), np.array(frames, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_from_mp4(video_path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """
    Extract mono audio from an MP4 file using moviepy.
    Returns (audio_mono_float32, sample_rate).
    """
    from moviepy import VideoFileClip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clip = VideoFileClip(video_path)
        audio_array = clip.audio.to_soundarray(fps=target_sr)  # (n_samples, channels)
        clip.close()

    if audio_array.ndim == 2:
        audio_mono = audio_array.mean(axis=1).astype(np.float32)
    else:
        audio_mono = audio_array.astype(np.float32)
    return audio_mono, target_sr


# ---------------------------------------------------------------------------
# Sim episode alignment (boilWater / makeSalad)
# ---------------------------------------------------------------------------

def align_sim_episode(episode_dir: str, output_dir: str) -> str:
    """
    Align one sim episode. Returns path to saved .npz file.
    """
    episode_id = os.path.basename(episode_dir)

    # --- Load task metadata ---
    with open(os.path.join(episode_dir, "task.json")) as f:
        meta = json.load(f)
    failure_ts = get_failure_timestamps(meta)

    # --- Load frames ---
    timestamps, frames = load_sim_frames(episode_dir)

    # --- Load audio from MP4 ---
    video_path = os.path.join(episode_dir, "original-video.mp4")
    audio, sr = load_audio_from_mp4(video_path)

    # --- Extract audio windows ---
    audio_windows = audio_windows_for_timestamps(audio, sr, timestamps)

    # --- Failure labels ---
    labels = failure_labels_for_timestamps(timestamps, failure_ts)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{episode_id}.npz")
    np.savez(
        out_path,
        timestamps=timestamps,
        frames=frames,
        audio_windows=audio_windows,
        failure_labels=labels,
        fps_base=1.0,
        audio_sr=sr,
    )
    return out_path


# ---------------------------------------------------------------------------
# Real-world episode alignment (putFruitsBowl)
# ---------------------------------------------------------------------------

def load_tasks_real_world(data_dir: str) -> dict:
    """Load tasks_real_world.json keyed by general_folder_name."""
    path = os.path.join(data_dir, "tasks_real_world.json")
    with open(path) as f:
        raw = json.load(f)
    by_folder = {}
    for entry in raw.values():
        folder = entry.get("general_folder_name", "")
        by_folder[folder] = entry
    return by_folder


def align_real_episode(episode_dir: str, output_dir: str,
                        tasks_meta: dict) -> str:
    """
    Align one real-world episode. Returns path to saved .npz file.
    tasks_meta: dict keyed by general_folder_name (from load_tasks_real_world).
    """
    episode_id = os.path.basename(episode_dir)

    # --- Match to tasks_real_world entry ---
    # episode_id e.g. "putFruitsBowl2" → folder name "putFruitsBowl2"
    # The general_folder_name in JSON may use a different numbering scheme;
    # try exact match first, then prefix match.
    meta = tasks_meta.get(episode_id)
    if meta is None:
        # Try matching by prefix (e.g. "putFruitsBowl" prefix of "putFruitsBowl2")
        for k, v in tasks_meta.items():
            if episode_id.startswith(k) or k.startswith(episode_id):
                meta = v
                break
    failure_ts = get_failure_timestamps(meta) if meta else []
    if not failure_ts:
        print(f"  [WARN] No failure timestamps found for {episode_id}")

    # --- Extract frames at 2fps ---
    video_path = os.path.join(episode_dir, "videos", "color.mp4")
    timestamps, frames = extract_real_frames_2fps(video_path)

    # --- Load audio ---
    audio_path = os.path.join(episode_dir, "videos", "audio.wav")
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    # --- Extract audio windows ---
    audio_windows = audio_windows_for_timestamps(audio, sr, timestamps)

    # --- Failure labels ---
    labels = failure_labels_for_timestamps(timestamps, failure_ts)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{episode_id}.npz")
    np.savez(
        out_path,
        timestamps=timestamps,
        frames=frames,
        audio_windows=audio_windows,
        failure_labels=labels,
        fps_base=2.0,
        audio_sr=sr,
    )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SIM_TASKS = ["boilWater", "makeSalad"]
REAL_TASKS = ["putFruitsBowl"]


def iter_episodes(task_name: str):
    task_dir = os.path.join(DATA_DIR, task_name)
    if not os.path.isdir(task_dir):
        return
    for ep in sorted(os.listdir(task_dir)):
        ep_dir = os.path.join(task_dir, ep)
        if os.path.isdir(ep_dir) and not ep.startswith("."):
            yield ep_dir


def main():
    tasks_meta = load_tasks_real_world(DATA_DIR)

    # --- Sim episodes ---
    sim_episodes = [ep for task in SIM_TASKS for ep in iter_episodes(task)]
    print(f"Found {len(sim_episodes)} sim episodes")
    for ep_dir in tqdm(sim_episodes, desc="Sim episodes"):
        ep_id = os.path.basename(ep_dir)
        try:
            out = align_sim_episode(ep_dir, OUTPUT_DIR)
            tqdm.write(f"  ✓ {ep_id} → {out}")
        except Exception as e:
            tqdm.write(f"  ✗ {ep_id} — {e}")

    # --- Real episodes ---
    real_episodes = [ep for task in REAL_TASKS for ep in iter_episodes(task)]
    print(f"Found {len(real_episodes)} real episodes")
    for ep_dir in tqdm(real_episodes, desc="Real episodes"):
        ep_id = os.path.basename(ep_dir)
        try:
            out = align_real_episode(ep_dir, OUTPUT_DIR, tasks_meta)
            tqdm.write(f"  ✓ {ep_id} → {out}")
        except Exception as e:
            tqdm.write(f"  ✗ {ep_id} — {e}")

    print(f"\nDone. Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Allow running on a single episode for testing:
    #   python align.py boilWater/boilWater-1
    if len(sys.argv) == 2:
        ep_path = os.path.join(DATA_DIR, sys.argv[1])
        task = os.path.basename(os.path.dirname(ep_path))
        tasks_meta = load_tasks_real_world(DATA_DIR)
        if task in SIM_TASKS:
            out = align_sim_episode(ep_path, OUTPUT_DIR)
        else:
            out = align_real_episode(ep_path, OUTPUT_DIR, tasks_meta)
        print(f"Saved: {out}")
    else:
        main()
