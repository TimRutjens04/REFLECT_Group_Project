#!/usr/bin/env python3
"""
Train the SigLIPStateAdapter on labeled crops extracted from AI2-THOR sim data.

Loss: BCEWithLogitsLoss per-pair, masked so only applicable pairs contribute
      (e.g. a StoveBurner crop does not supervise the full/empty head).

Split: 80/10/10 by episode — all crops from the same episode stay in one split.

Usage:
  poetry run python3 code/state_classifier/train.py
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
from sklearn.metrics import average_precision_score

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
    save_adapter,
)

# ── hyper-parameters ──────────────────────────────────────────────────────────
# E3 ablation: val mAP peaked at epoch ~30; early stopping patience=8.
EPOCHS         = 30
EARLY_STOP_PAT = 8
BATCH_SIZE     = 64
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 0   # PIL Images don't pickle well across workers

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def compute_per_pair_ap(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask:   torch.Tensor,
) -> list[float]:
    """Average Precision per state pair (mask=1 only), following Newman et al."""
    probs = torch.sigmoid(logits).numpy()
    aps = []
    for i in range(N_PAIRS):
        m = mask[:, i].bool().numpy()
        if m.sum() < 2 or labels[m, i].numpy().sum() == 0:
            aps.append(float("nan"))
        else:
            aps.append(float(average_precision_score(labels[m, i].numpy(), probs[m, i])))
    return aps


@torch.no_grad()
def evaluate(
    adapter:   StateAdapter,
    siglip:    nn.Module,
    loader:    DataLoader,
    criterion: nn.BCEWithLogitsLoss,
) -> tuple[float, list[float]]:
    """Return mean masked loss and per-pair AP."""
    adapter.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_masks:  list[torch.Tensor] = []

    for pixel_values, labels, mask in loader:
        features = extract_siglip_features(pixel_values, siglip, DEVICE)
        logits   = adapter(features)
        loss_mat = criterion(logits, labels.to(DEVICE))
        masked   = (loss_mat * mask.to(DEVICE)).sum() / mask.to(DEVICE).sum().clamp(min=1)
        total_loss += masked.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels)
        all_masks.append(mask)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_masks  = torch.cat(all_masks)
    aps = compute_per_pair_ap(all_logits, all_labels, all_masks)
    return total_loss / max(len(loader), 1), aps


def main() -> None:
    print(f"Training SigLIP state adapter | device={DEVICE}")

    # ── data ──────────────────────────────────────────────────────────────────
    samples = load_dataset()
    train_s, val_s, _ = split_by_episode(samples)

    processor = load_processor()
    train_ds = StateDataset(train_s, processor)
    val_ds   = StateDataset(val_s,   processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)

    # ── models ────────────────────────────────────────────────────────────────
    print(f"Loading SigLIP ({SIGLIP_MODEL_ID})…")
    siglip = AutoModel.from_pretrained(SIGLIP_MODEL_ID).to(DEVICE)
    siglip.eval()
    for p in siglip.parameters():
        p.requires_grad = False

    adapter = StateAdapter().to(DEVICE)
    print(f"Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")

    # ── training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_map  = -1.0
    patience_left = EARLY_STOP_PAT

    for epoch in range(1, EPOCHS + 1):
        adapter.train()
        train_loss = 0.0

        for pixel_values, labels, mask in tqdm(
            train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False
        ):
            features = extract_siglip_features(pixel_values, siglip, DEVICE)
            logits   = adapter(features)
            loss_mat = criterion(logits, labels.to(DEVICE))
            mask_dev = mask.to(DEVICE)
            loss     = (loss_mat * mask_dev).sum() / mask_dev.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        val_loss, val_aps = evaluate(adapter, siglip, val_loader, criterion)
        valid_aps = [a for a in val_aps if not np.isnan(a)]
        mean_ap   = float(np.mean(valid_aps)) if valid_aps else 0.0

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_mAP={mean_ap:.3f}")

        pair_strs = []
        for (pos, neg), ap in zip(STATE_PAIRS, val_aps):
            pair_strs.append(f"{pos}/{neg}: {'—' if np.isnan(ap) else f'{ap:.3f}'}")
        print("  " + "  ".join(pair_strs))

        if mean_ap > best_val_map:
            best_val_map  = mean_ap
            patience_left = EARLY_STOP_PAT
            save_adapter(adapter)
            print(f"  ✓ new best mAP={best_val_map:.3f} — checkpoint saved")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stopping at epoch {epoch} (patience={EARLY_STOP_PAT})")
                break

    print(f"\nTraining complete. Best val mAP: {best_val_map:.3f}")


if __name__ == "__main__":
    main()
