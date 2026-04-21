"""
Encoding module — REFLECT / RoboFail dataset.

Reads aligned .npz files (from align.py) and produces per-frame embeddings
using pretrained CLIP (vision) and WAV2CLIP (audio).

Both models share the same 512-dim latent space, making their embeddings
directly comparable and fusable.

=== MODELS ===

  CLIP ViT-B/32 (OpenAI)
    Input:  RGB frame 224×224, normalized
    Output: 512-dim float32 embedding, L2-normalized

  WAV2CLIP (Lyrebird / descriptive.ai)
    Input:  mono audio waveform at 16000 Hz (model resamples internally)
    Output: 512-dim float32 embedding, L2-normalized

=== OUTPUT FORMAT ===

  encoded/<episode_id>.npz
    timestamps           (N,)       float64 — from aligned file
    visual_embeddings    (N, 512)   float32 — CLIP
    audio_embeddings     (N, 512)   float32 — WAV2CLIP
    failure_labels       (N,)       bool    — from aligned file
    fps_base             scalar     float
    audio_sr             scalar     int     — original audio sample rate

Usage:
    python encode.py aligned/boilWater-1.npz          # single file
    python encode.py aligned/                          # all .npz in directory
"""

import os
import sys

import clip
import numpy as np
import torch
import wav2clip
from PIL import Image
from tqdm import tqdm

ENCODED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "encoded"))
if torch.cuda.is_available():
    DEVICE = "cuda"

elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# WAV2CLIP expects 16 kHz mono audio
WAV2CLIP_SR = 16000


# ---------------------------------------------------------------------------
# Model loading (cached globals — load once per process)
# ---------------------------------------------------------------------------

_clip_model = None
_clip_preprocess = None
_wav2clip_model = None


def get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
        _clip_model.eval()
    return _clip_model, _clip_preprocess


def get_wav2clip():
    global _wav2clip_model
    if _wav2clip_model is None:
        _wav2clip_model = wav2clip.get_model()
    return _wav2clip_model


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_frames_clip(frames: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """
    Encode (N, H, W, 3) uint8 RGB frames with CLIP.
    Returns (N, 512) float32, L2-normalized.
    """
    model, preprocess = get_clip()
    embeddings = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        imgs = torch.stack([
            preprocess(Image.fromarray(f)) for f in batch_frames
        ]).to(DEVICE)

        with torch.no_grad():
            emb = model.encode_image(imgs).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize

        embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0).astype(np.float32)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio array to target sample rate."""
    if orig_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def encode_audio_wav2clip(audio_windows: np.ndarray, orig_sr: int,
                           batch_size: int = 32) -> np.ndarray:
    """
    Encode (N, n_samples) float32 audio windows with WAV2CLIP.
    Windows are resampled to 16 kHz before encoding.
    Returns (N, 512) float32, L2-normalized.
    """
    model = get_wav2clip()
    embeddings = []

    for i in range(0, len(audio_windows), batch_size):
        batch = audio_windows[i : i + batch_size]

        # Resample each window to 16 kHz (wav2clip expects 16 kHz)
        resampled = np.stack([
            resample_audio(w, orig_sr, WAV2CLIP_SR) for w in batch
        ])

        emb = wav2clip.embed_audio(resampled, model)  # (batch, 512)
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        emb = emb / norm

        embeddings.append(emb)

    return np.concatenate(embeddings, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-episode encoding
# ---------------------------------------------------------------------------

def encode_episode(npz_path: str, output_dir: str) -> str:
    """
    Load one aligned .npz, produce embeddings, save encoded .npz.
    Returns path to saved file.
    """
    episode_id = os.path.basename(npz_path).replace(".npz", "")
    d = np.load(npz_path, allow_pickle=True)

    frames = d["frames"]                      # (N, H, W, 3) uint8
    audio_windows = d["audio_windows"]        # (N, n_samples) float32
    timestamps = d["timestamps"]              # (N,) float64
    failure_labels = d["failure_labels"]      # (N,) bool
    fps_base = float(d["fps_base"])
    audio_sr = int(d["audio_sr"])

    # Encode vision
    visual_embeddings = encode_frames_clip(frames)

    # Encode audio
    audio_embeddings = encode_audio_wav2clip(audio_windows, audio_sr)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{episode_id}.npz")
    np.savez(
        out_path,
        timestamps=timestamps,
        visual_embeddings=visual_embeddings,
        audio_embeddings=audio_embeddings,
        failure_labels=failure_labels,
        fps_base=fps_base,
        audio_sr=audio_sr,
    )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(target: str):
    if os.path.isdir(target):
        npz_files = sorted([
            os.path.join(target, f)
            for f in os.listdir(target)
            if f.endswith(".npz") and "_check" not in f
        ])
    else:
        npz_files = [target]

    if not npz_files:
        print(f"No .npz files found in {target}")
        sys.exit(1)

    print(f"Device: {DEVICE}")
    print(f"Encoding {len(npz_files)} episode(s) → {ENCODED_DIR}")

    # Pre-load models once before the loop
    print("Loading CLIP ViT-B/32...")
    get_clip()
    print("Loading WAV2CLIP...")
    get_wav2clip()

    for path in tqdm(npz_files, desc="Encoding"):
        ep_id = os.path.basename(path).replace(".npz", "")
        try:
            out = encode_episode(path, ENCODED_DIR)
            tqdm.write(f"  ✓ {ep_id} → {out}")
        except Exception as e:
            tqdm.write(f"  ✗ {ep_id} — {e}")
            raise

    print(f"\nDone. Encoded files in: {ENCODED_DIR}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "aligned/"
    main(target)
