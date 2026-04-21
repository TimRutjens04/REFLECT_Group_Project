import time

import numpy as np
import streamlit as st
import torch
from PIL import Image

import cv2

from .config import DEVICE
from .data import load_episode
from .logger import log


def _single_attention_map(frame: np.ndarray, model, preprocess) -> np.ndarray:
    """
    Extract 7x7 CLS→patch attention from CLIP ViT-B/32 last layer.
    Returns (7, 7) float32.
    """
    img = preprocess(Image.fromarray(frame)).unsqueeze(0).to(DEVICE)
    captured = {}
    guard = [False]

    def hook(module, input, output):
        if guard[0]:
            return
        guard[0] = True
        q, k, v = input[0], input[1], input[2]
        with torch.no_grad():
            _, attn = module(q, k, v, need_weights=True, average_attn_weights=True)
        captured["attn"] = attn.detach().cpu()
        guard[0] = False

    handle = model.visual.transformer.resblocks[-1].attn.register_forward_hook(hook)
    with torch.no_grad():
        model.encode_image(img)
    handle.remove()

    return captured["attn"][0, 0, 1:].reshape(7, 7).numpy().astype(np.float32)


@st.cache_data(show_spinner="Computing attention maps (one-time)...")
def compute_attention_maps(episode_id: str, _model, _preprocess) -> np.ndarray:
    """Compute (N, 7, 7) CLS attention maps for every frame. Cached per episode."""
    log.info("Computing attention maps for %s", episode_id)
    t0 = time.perf_counter()
    frames = load_episode(episode_id)["frames"]
    n = len(frames)
    maps = np.zeros((n, 7, 7), dtype=np.float32)
    progress = st.progress(0, text="Computing attention maps...")
    for i, frame in enumerate(frames):
        maps[i] = _single_attention_map(frame, _model, _preprocess)
        progress.progress((i + 1) / n, text=f"Attention maps: {i+1}/{n}")
    progress.empty()
    elapsed = time.perf_counter() - t0
    log.info("Attention maps done: %d frames in %.2fs (%.3fs/frame)", n, elapsed, elapsed / n)
    return maps


def overlay_attention(frame: np.ndarray, attn_7x7: np.ndarray,
                      alpha: float = 0.45) -> np.ndarray:
    """
    Blend CLIP attention heatmap onto the frame and draw a red box around
    the highest-attention patch, mapped back to original image coordinates.
    Returns (H, W, 3) uint8 RGB.
    """
    h, w = frame.shape[:2]

    attn_resized = cv2.resize(attn_7x7, (w, h), interpolation=cv2.INTER_LINEAR)
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    heatmap_bgr = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    composite = (
        frame.astype(np.float32) * (1 - alpha)
        + heatmap_rgb.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)

    # Map top-attention patch back to original image coordinates.
    # CLIP center-crops to min(h,w)×min(h,w) before resizing to 224×224.
    crop_size   = min(h, w)
    top_offset  = (h - crop_size) // 2
    left_offset = (w - crop_size) // 2
    scale       = crop_size / 224.0

    r, c = np.unravel_index(attn_7x7.argmax(), (7, 7))
    y1 = max(0,     top_offset  + int(r * 32 * scale))
    x1 = max(0,     left_offset + int(c * 32 * scale))
    y2 = min(h - 1, top_offset  + int((r + 1) * 32 * scale))
    x2 = min(w - 1, left_offset + int((c + 1) * 32 * scale))

    cv2.rectangle(composite, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)
    return composite


def draw_object_box(frame: np.ndarray, box: list,
                    color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Draw a colored box on the frame at pixel coords [x1, y1, x2, y2].
    Returns a new (H, W, 3) uint8 RGB array.
    """
    result = frame.copy()
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness=2)
    return result
