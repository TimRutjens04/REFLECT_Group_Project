#!/usr/bin/env python3
"""
Evaluate the SigLIP state adapter against the zero-shot baseline on the test split.

Reports per-pair accuracy for:
  1. Adapter (fine-tuned MLP on frozen SigLIP features)
  2. Zero-shot SigLIP (current classify_state from depth_state.py)

Usage:
  poetry run python3 code/state_classifier/eval.py
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from state_classifier.config import N_PAIRS, SIGLIP_MODEL_ID, STATE_PAIRS  # noqa: E402
from state_classifier.dataset import (  # noqa: E402
    StateDataset,
    load_dataset,
    load_processor,
    split_by_episode,
)
from state_classifier.model import (  # noqa: E402
    StateAdapter,
    extract_siglip_features,
    load_adapter,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64


# ── Adapter evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def eval_adapter(
    adapter: StateAdapter,
    siglip:  nn.Module,
    loader:  DataLoader,
) -> list[float]:
    """Per-pair binary accuracy for the trained adapter."""
    adapter.eval()
    correct = np.zeros(N_PAIRS)
    total   = np.zeros(N_PAIRS)

    for pixel_values, labels, mask in tqdm(loader, desc="Adapter", leave=False):
        features = extract_siglip_features(pixel_values, siglip, DEVICE)
        logits   = adapter(features).cpu()
        preds    = (logits > 0).float()

        for i in range(N_PAIRS):
            m = mask[:, i].bool()
            if m.sum() > 0:
                correct[i] += (preds[m, i] == labels[m, i]).sum().item()
                total[i]   += m.sum().item()

    return [correct[i] / total[i] if total[i] > 0 else float("nan")
            for i in range(N_PAIRS)]


# ── Zero-shot baseline ────────────────────────────────────────────────────────

@torch.inference_mode()
def eval_zero_shot(
    samples:   list[dict],
    processor,
    siglip:    nn.Module,
) -> list[float]:
    """
    Zero-shot SigLIP: for each state pair query "a {pos} {object_type}" vs
    "a {neg} {object_type}", pick the higher sigmoid score.
    Mirrors the logic in depth_state.py classify_state.
    """
    correct = np.zeros(N_PAIRS)
    total   = np.zeros(N_PAIRS)

    for s in tqdm(samples, desc="Zero-shot", leave=False):
        obj_type = s["object_type"]
        image    = s["image"]

        for pair_idx, (pos_label, neg_label) in enumerate(STATE_PAIRS):
            if not s["mask"][pair_idx]:
                continue
            if pair_idx not in s["labels"]:
                continue

            queries = [f"a {pos_label} {obj_type}", f"a {neg_label} {obj_type}"]
            inputs = processor(
                text=queries,
                images=[image, image],
                return_tensors="pt",
                padding="max_length",
            ).to(DEVICE)
            outputs = siglip(**inputs)
            # diagonal: similarity(image_i, query_i)
            probs = torch.sigmoid(outputs.logits_per_image.diagonal()).cpu()
            pred_positive = probs[0] > probs[1]
            gt_positive   = s["labels"][pair_idx]

            correct[pair_idx] += int(pred_positive == gt_positive)
            total[pair_idx]   += 1

    return [correct[i] / total[i] if total[i] > 0 else float("nan")
            for i in range(N_PAIRS)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Evaluating state classifier | device={DEVICE}")

    samples = load_dataset()
    _, _, test_s = split_by_episode(samples)

    if not test_s:
        sys.exit("Test split is empty — not enough episodes for a 3-way split.")

    processor = load_processor()

    print(f"Loading SigLIP ({SIGLIP_MODEL_ID})…")
    siglip = AutoModel.from_pretrained(SIGLIP_MODEL_ID).to(DEVICE)
    siglip.eval()
    for p in siglip.parameters():
        p.requires_grad = False

    adapter = load_adapter(DEVICE)
    if adapter is None:
        print("[warn] No adapter checkpoint found — only zero-shot baseline will run.")

    # Adapter accuracy
    adapter_accs = [float("nan")] * N_PAIRS
    if adapter is not None:
        test_ds     = StateDataset(test_s, processor)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        adapter_accs = eval_adapter(adapter, siglip, test_loader)

    # Zero-shot accuracy
    zs_accs = eval_zero_shot(test_s, processor, siglip)

    # ── Print comparison table ────────────────────────────────────────────────
    header = f"\n{'State pair':<16} {'Zero-shot':>10} {'Adapter':>10} {'Delta':>8}"
    print(header)
    print("─" * len(header))

    deltas = []
    for i, (pos, neg) in enumerate(STATE_PAIRS):
        pair_str = f"{pos}/{neg}"
        zs  = zs_accs[i]
        ada = adapter_accs[i]
        if zs != zs or ada != ada:    # NaN = no test samples for this pair
            print(f"{pair_str:<16} {'—':>10} {'—':>10} {'—':>8}")
        else:
            delta = ada - zs
            deltas.append(delta)
            sign  = "+" if delta >= 0 else ""
            print(f"{pair_str:<16} {zs:>10.3f} {ada:>10.3f} {sign+f'{delta:.3f}':>8}")

    if deltas:
        mean_delta = sum(deltas) / len(deltas)
        sign = "+" if mean_delta >= 0 else ""
        print(f"{'MEAN':<16} {''*10} {''*10} {sign+f'{mean_delta:.3f}':>8}")

    print(f"\nTest crops: {len(test_s)}")


if __name__ == "__main__":
    main()
