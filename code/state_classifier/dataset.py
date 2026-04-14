"""
PyTorch Dataset for the SigLIP state adapter.

Splits by episode (not by crop) to prevent data leakage — all crops from the
same episode end up in the same split.
"""

from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from state_classifier.config import DATASET_PATH, N_PAIRS, SIGLIP_MODEL_ID  # noqa: E402


class StateDataset(Dataset):
    """
    Wraps the pickled list of crop dicts.

    Each item returns:
        pixel_values  (3, 224, 224) float32 tensor — SigLIP input
        labels        (N_PAIRS,)    float32 — 1.0 positive, 0.0 negative
        mask          (N_PAIRS,)    float32 — 1.0 if pair applies, else 0.0
    """

    def __init__(self, samples: list[dict], processor: AutoProcessor) -> None:
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]

        inputs = self.processor(
            images=s["image"],
            return_tensors="pt",
            padding="max_length",
        )
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, H, W)

        labels = torch.zeros(N_PAIRS, dtype=torch.float32)
        mask   = torch.tensor(s["mask"], dtype=torch.float32)

        for pair_idx, is_positive in s["labels"].items():
            labels[pair_idx] = 1.0 if is_positive else 0.0

        return pixel_values, labels, mask


def split_by_episode(
    samples: list[dict],
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    seed:       int   = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split samples into train/val/test by episode ID.

    All crops from one episode stay in the same split, preventing leakage.
    """
    episodes = sorted({s["episode"] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(episodes)

    n_train = max(1, int(len(episodes) * train_frac))
    n_val   = max(1, int(len(episodes) * val_frac))

    train_eps = set(episodes[:n_train])
    val_eps   = set(episodes[n_train: n_train + n_val])
    test_eps  = set(episodes[n_train + n_val:])

    train = [s for s in samples if s["episode"] in train_eps]
    val   = [s for s in samples if s["episode"] in val_eps]
    test  = [s for s in samples if s["episode"] in test_eps]

    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test crops "
          f"({len(train_eps)}/{len(val_eps)}/{len(test_eps)} episodes)")
    return train, val, test


def load_processor() -> AutoProcessor:
    return AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    if not path.exists():
        sys.exit(f"Dataset not found at {path}. Run `just classify-generate` first.")
    with open(path, "rb") as f:
        return pickle.load(f)
