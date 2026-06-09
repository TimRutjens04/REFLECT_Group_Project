"""Quick debug: show everything GDINO detects on frame 0 of a video, no label filter."""
import sys, os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
video_path = sys.argv[1] if len(sys.argv) > 1 else None
label      = sys.argv[2] if len(sys.argv) > 2 else "can"

if not video_path:
    print("Usage: poetry run python code/debug_gdino.py <video> [label]")
    sys.exit(1)

if not os.path.isabs(video_path):
    video_path = os.path.join(ROOT, video_path)

cap = cv2.VideoCapture(video_path)
_, frame0 = cap.read()
cap.release()

pil = Image.fromarray(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
print(f"Frame size: {pil.size}  label: '{label}'")

print("Loading GDINO...")
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model     = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cpu").eval()

text   = f"{label}."
inputs = processor(images=pil, text=text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Show all detections at very low threshold so nothing is filtered
results = processor.post_process_grounded_object_detection(
    outputs, inputs.input_ids,
    threshold=0.05, text_threshold=0.05,
    target_sizes=[pil.size[::-1]],
)[0]

print(f"\nAll detections (threshold=0.05) for prompt '{text}':")
for box, score, lbl in sorted(
    zip(results["boxes"], results["scores"], results["text_labels"]),
    key=lambda x: -x[1]
):
    print(f"  score={float(score):.3f}  label='{lbl}'  box={box.int().tolist()}")
