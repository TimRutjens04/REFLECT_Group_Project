import cv2
import os
from PIL import Image

def extract_frames(video_path, fps=1, save=True):
    cap = cv2.VideoCapture(video_path)
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frames = []
    frame_count = 0
    saved_count = 0

    # Create frames folder next to video
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

    print(f"Extracted {saved_count} frames at {fps} FPS")
    if save:
        print(f"📁 Frames saved to: {frames_dir}")

    return frames, frames_dir



video_path = "/Users/guray/Desktop/Text-Object/data/putAppleBowl2/color.mp4" 
frames = extract_frames(video_path, fps=0.2)