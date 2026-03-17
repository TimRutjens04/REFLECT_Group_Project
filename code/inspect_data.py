"""
Data inspection script — run this before align.py to confirm formats.
Results should be documented in a comment block at the top of align.py.
"""

import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def inspect_video(path, label):
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        print(f"[VIDEO] {label}")
        print(f"  FPS:         {fps}")
        print(f"  Frame count: {int(frame_count)}")
        print(f"  Resolution:  {int(width)}x{int(height)}")
        print(f"  Duration:    {duration:.2f}s")
    except Exception as e:
        print(f"[VIDEO] {label} — ERROR: {e}")


def inspect_audio_wav(path, label):
    try:
        import librosa
        audio, sr = librosa.load(path, sr=None, mono=True)
        duration = len(audio) / sr
        print(f"[AUDIO WAV] {label}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Samples:     {len(audio)}")
        print(f"  Duration:    {duration:.2f}s")
        print(f"  dtype:       {audio.dtype}")
    except Exception as e:
        print(f"[AUDIO WAV] {label} — ERROR: {e}")


def inspect_mp4_audio(path, label):
    """Try to extract audio from MP4 via moviepy."""
    try:
        from moviepy import VideoFileClip
        clip = VideoFileClip(path)
        if clip.audio is None:
            print(f"[AUDIO MP4] {label} — NO AUDIO TRACK")
        else:
            print(f"[AUDIO MP4] {label}")
            print(f"  Audio FPS:  {clip.audio.fps}")
            print(f"  Duration:   {clip.duration:.2f}s")
        clip.close()
    except Exception as e:
        print(f"[AUDIO MP4] {label} — ERROR: {e}")


def inspect_task_json(path, label):
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"[TASK JSON] {label}")
        print(f"  Keys: {list(data.keys())}")
        print(f"  gt_failure_step: {data.get('gt_failure_step')!r}")
        print(f"  gt_failure_reason: {data.get('gt_failure_reason')!r}")
        sounds = data.get("sounds", {})
        print(f"  sounds ({len(sounds)} entries): {dict(list(sounds.items())[:3])} ...")
    except Exception as e:
        print(f"[TASK JSON] {label} — ERROR: {e}")


def inspect_tasks_real_world(path):
    try:
        with open(path) as f:
            data = json.load(f)
        keys = list(data.keys())
        print(f"[TASKS REAL WORLD] {path}")
        print(f"  Total tasks: {len(keys)}")
        print(f"  First entry key: {keys[0]!r}")
        first = data[keys[0]]
        print(f"  First entry:\n  {json.dumps(first, indent=4)}")
    except Exception as e:
        print(f"[TASKS REAL WORLD] ERROR: {e}")


def inspect_ego_img(episode_dir, label):
    img_dir = os.path.join(episode_dir, "ego_img")
    if not os.path.isdir(img_dir):
        print(f"[EGO IMG] {label} — directory not found")
        return
    files = sorted(os.listdir(img_dir))
    print(f"[EGO IMG] {label}")
    print(f"  Frame count: {len(files)}")
    print(f"  First/last:  {files[0]} … {files[-1]}")
    # Check one image size
    try:
        import cv2
        sample = cv2.imread(os.path.join(img_dir, files[0]), cv2.IMREAD_UNCHANGED)
        print(f"  Shape (first): {sample.shape}, dtype: {sample.dtype}")
    except Exception as e:
        print(f"  Shape check failed: {e}")


def inspect_zarr(path, label):
    try:
        import zarr
        store = zarr.open(path, mode="r")
        print(f"[ZARR] {label}")
        def _print_tree(grp, indent=2):
            for k in grp.keys():
                item = grp[k]
                if hasattr(item, "shape"):
                    print(f"{' '*indent}{k}: shape={item.shape}, dtype={item.dtype}")
                else:
                    print(f"{' '*indent}{k}/")
                    _print_tree(item, indent + 2)
        _print_tree(store)
    except Exception as e:
        print(f"[ZARR] {label} — ERROR: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("REFLECT DATA INSPECTION")
    print("=" * 60)

    # --- boilWater-1 ---
    bw1 = os.path.join(DATA_DIR, "boilWater", "boilWater-1")
    print("\n--- boilWater-1 ---")
    inspect_video(os.path.join(bw1, "original-video.mp4"), "boilWater-1/original-video.mp4")
    inspect_mp4_audio(os.path.join(bw1, "original-video.mp4"), "boilWater-1/original-video.mp4")
    inspect_task_json(os.path.join(bw1, "task.json"), "boilWater-1/task.json")
    inspect_ego_img(bw1, "boilWater-1")

    # --- putFruitsBowl2 ---
    pfb = os.path.join(DATA_DIR, "putFruitsBowl", "putFruitsBowl2")
    print("\n--- putFruitsBowl2 ---")
    inspect_video(os.path.join(pfb, "videos", "color.mp4"), "putFruitsBowl2/videos/color.mp4")
    inspect_audio_wav(os.path.join(pfb, "videos", "audio.wav"), "putFruitsBowl2/videos/audio.wav")
    inspect_zarr(os.path.join(pfb, "replay_buffer.zarr"), "putFruitsBowl2/replay_buffer.zarr")

    # --- tasks_real_world.json ---
    print("\n--- tasks_real_world.json ---")
    inspect_tasks_real_world(os.path.join(DATA_DIR, "tasks_real_world.json"))

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
