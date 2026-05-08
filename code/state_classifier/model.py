"""
SigLIP 2 state adapter — frozen So400m encoder + depth=2 MLP classification heads.

Architecture:
  SigLIP 2 So400m  (frozen, ~400M params)
       ↓ pooler_output  (B, 1152)
  Linear(1152 → 512) → GELU → Dropout(0.1)
       ↓
  Linear(512 → 256)  → GELU → Dropout(0.1)
       ↓
  Linear(256 → 7)   — one logit per state pair
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from state_classifier.config import (  # noqa: E402
    BEST_CKPT,
    EMBED_DIM,
    N_PAIRS,
    OBJECT_PAIRS,
    STATE_PAIRS,
)


class StateAdapter(nn.Module):
    """
    Lightweight MLP adapter over frozen SigLIP 2 So400m pooled embeddings.

    Input:  (B, 1152) float32 — vision_model.pooler_output
    Output: (B, 7)    float32 — raw logit per state pair (sigmoid for probability)

    Loss during training: BCEWithLogitsLoss with per-sample applicability mask.
    At inference: use predict() to get {pair_name: probability} for applicable pairs.
    """

    PAIRS: list[str] = [f"{p}_{n}" for p, n in STATE_PAIRS]

    def __init__(
        self,
        in_dim:   int   = EMBED_DIM,
        hidden:   int   = 512,
        hidden2:  int   = 256,
        n_pairs:  int   = N_PAIRS,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_pairs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(
        self,
        embedding:   torch.Tensor,
        object_type: str | None = None,
    ) -> dict[str, float]:
        """
        Returns {pair_name: probability} for applicable pairs.

        If object_type is in OBJECT_PAIRS, only those heads are returned.
        Pass object_type=None to get all 7 probabilities.
        """
        logits = self.forward(embedding)
        probs  = torch.sigmoid(logits).squeeze(0).tolist()
        result = dict(zip(self.PAIRS, probs))
        if object_type and object_type in OBJECT_PAIRS:
            applicable = set(OBJECT_PAIRS[object_type])
            result = {k: v for k, v in result.items() if k in applicable}
        return result


@torch.inference_mode()
def extract_siglip_features(
    pixel_values: torch.Tensor,
    siglip_model: nn.Module,
    device: str | torch.device,
) -> torch.Tensor:
    """Run frozen SigLIP 2 vision encoder, return pooled features (B, 1152)."""
    vision_out = siglip_model.vision_model(
        pixel_values=pixel_values.to(device)
    )
    return vision_out.pooler_output


def load_adapter(device: str | torch.device = "cpu") -> StateAdapter | None:
    """Load best checkpoint if it exists, else return None."""
    if not BEST_CKPT.exists():
        return None
    adapter = StateAdapter()
    state   = torch.load(BEST_CKPT, map_location=device, weights_only=True)
    adapter.load_state_dict(state)
    adapter.to(device).eval()
    return adapter


def save_adapter(adapter: StateAdapter) -> None:
    BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), BEST_CKPT)
