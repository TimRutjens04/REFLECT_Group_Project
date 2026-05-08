"""
Shared configuration for the SigLIP 2 state adapter.

STATE_PAIRS: 7 binary state groups, alphabetical order (matches training index).
CAPABILITY_TO_PAIR: AI2-THOR capability flags → pair index — no per-label hardcoding;
  the simulator's own schema decides which heads are supervised per object.
OBJECT_PAIRS: inference-time filter — which pairs apply to each AI2-THOR object type.
  Justified by E1 (pairs outside this mapping had no supervision signal).

Reference: FINDINGS.md — Experiments 1–3.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# ── State pairs — alphabetical order, positive class first ────────────────────
# 7 pairs: held/free is intentionally excluded — handled by the tracking stage.
# Ablation winner (E3): depth=1 adapter, mean AP=0.899 on AI2-THOR sim.
STATE_PAIRS: list[tuple[str, str]] = [
    ("broken", "intact"),   # 0
    ("cooked", "raw"),      # 1
    ("dirty",  "clean"),    # 2
    ("full",   "empty"),    # 3
    ("on",     "off"),      # 4
    ("open",   "closed"),   # 5
    ("sliced", "whole"),    # 6
]
N_PAIRS = len(STATE_PAIRS)   # 7

# ── AI2-THOR capability flag → pair index ─────────────────────────────────────
CAPABILITY_TO_PAIR: dict[str, int] = {
    "breakable":         0,  # → (broken, intact)
    "cookable":          1,  # → (cooked, raw)
    "dirtyable":         2,  # → (dirty, clean)
    "canFillWithLiquid": 3,  # → (full, empty)
    "toggleable":        4,  # → (on, off)
    "openable":          5,  # → (open, closed)
    "sliceable":         6,  # → (sliced, whole)
}

# ── Ground-truth state field → pair index ─────────────────────────────────────
STATE_FIELD_TO_PAIR: dict[str, int] = {
    "isBroken":           0,
    "isCooked":           1,
    "isDirty":            2,
    "isFilledWithLiquid": 3,
    "isToggled":          4,
    "isOpen":             5,
    "isSliced":           6,
}

# ── Object-type → applicable pair names ───────────────────────────────────────
# Grounded in AI2-THOR capability flags; empirically justified by E1 (pairs
# outside this mapping showed ceiling AP as test-set artefact, not real signal).
OBJECT_PAIRS: dict[str, list[str]] = {
    "Bowl":          ["full_empty"],
    "Cup":           ["full_empty"],
    "Pot":           ["full_empty"],
    "Cabinet":       ["open_closed"],
    "Fridge":        ["open_closed"],
    "Microwave":     ["open_closed"],
    "Faucet":        ["on_off"],
    "CoffeeMachine": ["on_off"],
    "Potato":        ["cooked_raw", "dirty_clean", "sliced_whole"],
    "Bread":         ["cooked_raw", "dirty_clean", "sliced_whole"],
    "Egg":           ["cooked_raw", "broken_intact"],
    "Apple":         ["dirty_clean", "sliced_whole"],
    "Plate":         ["dirty_clean", "broken_intact"],
    "Bottle":        ["broken_intact"],
}

# ── Model / paths ──────────────────────────────────────────────────────────────
# SigLIP 2 So400m: empirically best encoder for physical state (E1: probe AP 0.866
# vs CLIP 0.831; mechanistically more disentangled per E2 superposition analysis).
SIGLIP_MODEL_ID = "google/siglip2-so400m-patch16-384"
EMBED_DIM       = 1152   # pooler_output dim for siglip2-so400m

DATASET_PATH    = ROOT / "state_classifier" / "dataset.pkl"
CHECKPOINT_DIR  = ROOT / "state_classifier" / "checkpoints"
BEST_CKPT       = CHECKPOINT_DIR / "best.pt"

MIN_CROP_PX = 8
