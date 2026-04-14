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

from state_classifier.config import N_PAIRS, SIGLIP_MODEL_ID, STATE_PAIRS  # noqa: E402
from state_classifier.dataset import (  # noqa: E402
    StateDataset,
    load_dataset,
    load_processor,
    split_by_episode,
)
from state_classifier.model import (  # noqa: E402
    SigLIPStateAdapter,
    extract_siglip_features,
    save_adapter,
)

# ── hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS      = 20
BATCH_SIZE  = 64
LR          = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0   # 0 = main process (PIL Images don't pickle well across workers)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def compute_per_pair_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask:   torch.Tensor,
) -> list[float]:
    """Binary accuracy per state pair (only where mask=1)."""
    preds = (logits > 0).float()
    accs = []
    for i in range(N_PAIRS):
        m = mask[:, i].bool()
        if m.sum() == 0:
            accs.append(float("nan"))
        else:
            accs.append(float((preds[m, i] == labels[m, i]).float().mean()))
    return accs


@torch.no_grad()
def evaluate(
    adapter:       SigLIPStateAdapter,
    siglip:        nn.Module,
    loader:        DataLoader,
    criterion:     nn.BCEWithLogitsLoss,
) -> tuple[float, list[float]]:
    """Return mean masked loss and per-pair accuracy."""
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
    accs = compute_per_pair_accuracy(all_logits, all_labels, all_masks)
    return total_loss / max(len(loader), 1), accs


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

    adapter = SigLIPStateAdapter().to(DEVICE)
    print(f"Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")

    # ── training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_acc = -1.0

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

        val_loss, val_accs = evaluate(adapter, siglip, val_loader, criterion)
        valid_accs = [a for a in val_accs if a == a]   # exclude NaN
        mean_acc   = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={mean_acc:.3f}")

        # Per-pair accuracy
        pair_strs = []
        for i, ((pos, neg), acc) in enumerate(zip(STATE_PAIRS, val_accs)):
            if acc != acc:   # NaN
                pair_strs.append(f"{pos}/{neg}: —")
            else:
                pair_strs.append(f"{pos}/{neg}: {acc:.2f}")
        print("  " + "  ".join(pair_strs))

        if mean_acc > best_val_acc:
            best_val_acc = mean_acc
            save_adapter(adapter)
            print(f"  ✓ new best val_acc={best_val_acc:.3f} — checkpoint saved")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
