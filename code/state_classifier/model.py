"""
SigLIPStateAdapter — frozen SigLIP vision encoder + trainable MLP classification heads.

Architecture:
  SigLIP vision model  (frozen, ~400M params)
       ↓ pooler_output  (B, EMBED_DIM)
  MLP adapter          (~800k trainable params)
       ↓
  (B, N_PAIRS) logits  — one independent binary head per state pair

The adapter is saved/loaded independently from SigLIP (which is cached by HuggingFace).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from state_classifier.config import BEST_CKPT, EMBED_DIM, N_PAIRS, SIGLIP_MODEL_ID  # noqa: E402


class SigLIPStateAdapter(nn.Module):
    """
    Trainable MLP that maps frozen SigLIP pooled vision features to state logits.

    SigLIP is NOT stored inside this module — it is passed in at inference time
    so train.py and depth_state.py can share the same already-loaded model.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, n_pairs: int = N_PAIRS) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_pairs),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, embed_dim) — SigLIP vision pooler_output
        Returns:
            logits:   (B, n_pairs)
        """
        return self.adapter(features)


@torch.inference_mode()
def extract_siglip_features(
    pixel_values: torch.Tensor,
    siglip_model: nn.Module,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Run frozen SigLIP vision encoder, return pooled features.

    Args:
        pixel_values: (B, 3, H, W)
        siglip_model: the full SigLIP AutoModel (vision + text)
        device: target device
    Returns:
        features: (B, EMBED_DIM)
    """
    vision_out = siglip_model.vision_model(
        pixel_values=pixel_values.to(device)
    )
    return vision_out.pooler_output   # (B, EMBED_DIM)


def load_adapter(device: str | torch.device = "cpu") -> SigLIPStateAdapter | None:
    """
    Load the best checkpoint if it exists.  Returns None if no checkpoint found.
    """
    if not BEST_CKPT.exists():
        return None
    adapter = SigLIPStateAdapter()
    state = torch.load(BEST_CKPT, map_location=device, weights_only=True)
    adapter.load_state_dict(state)
    adapter.to(device)
    adapter.eval()
    return adapter


def save_adapter(adapter: SigLIPStateAdapter) -> None:
    BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), BEST_CKPT)
