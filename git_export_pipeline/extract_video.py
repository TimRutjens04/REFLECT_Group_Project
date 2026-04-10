import cv2
import os
from PIL import Image

def extract_frames(video_path, fps=1, save=True):
    if not os.path.exists(video_path):
        print(f"ERROR: Video file does not exist:\n{video_path}")
        return [], None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open the video:\n{video_path}")
        return [], None

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo path: {video_path}")
    print(f"Reported FPS: {video_fps}")
    print(f"Reported total frames: {total_frames}")

    if video_fps is None or video_fps <= 0:
        frame_interval = 1
    else:
        frame_interval = max(1, int(video_fps / fps))

    frames = []
    frame_count = 0
    saved_count = 0

    video_dir = os.path.dirname(video_path)
    frames_dir = os.path.join(video_dir, "frames")

    if save:
        os.makedirs(frames_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

            if save:
                frame_filename = os.path.join(frames_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)

            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Extracted {saved_count} frames at target {fps} FPS")
    print(f"Frames saved to: {frames_dir}")

    return frames, frames_dir


video_paths = [
    r"C:\Users\gtomo\OneDrive\Desktop\Sem6\testingPhase1\Data\Data\Sim Data\boilWater-1\original-video.mp4",
    r"C:\Users\gtomo\OneDrive\Desktop\Sem6\testingPhase1\Data\Data\Sim Data\toastBread-1\original-video.mp4"
]

for video_path in video_paths:
    extract_frames(video_path, fps=1)