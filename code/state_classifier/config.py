"""
Shared configuration for the SigLIP state adapter.

STATE_PAIRS defines 8 mutually exclusive binary state groups.
CAPABILITY_TO_PAIR maps AI2-THOR object capability flags to the pair index they supervise —
this avoids per-label hardcoding; the simulator's own schema decides which heads are trained
on which objects (e.g. any object where canFillWithLiquid=True gets the full/empty head).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# ── State pairs (binary: positive class = index 0 of each tuple) ──────────────
STATE_PAIRS: list[tuple[str, str]] = [
    ("full",   "empty"),    # 0
    ("open",   "closed"),   # 1
    ("on",     "off"),      # 2
    ("held",   "free"),     # 3
    ("sliced", "whole"),    # 4
    ("cooked", "raw"),      # 5
    ("dirty",  "clean"),    # 6
    ("broken", "intact"),   # 7
]
N_PAIRS = len(STATE_PAIRS)

# ── AI2-THOR capability flag → pair index ─────────────────────────────────────
# These flags live in event.metadata["objects"][i] as booleans.
# A True flag means the object *can* exhibit that state pair, so we supervise it.
CAPABILITY_TO_PAIR: dict[str, int] = {
    "canFillWithLiquid": 0,
    "openable":          1,
    "toggleable":        2,
    "pickupable":        3,
    "sliceable":         4,
    "cookable":          5,
    "dirtyable":         6,
    "breakable":         7,
}

# ── Ground-truth state field → pair index ─────────────────────────────────────
# These flags hold the actual current state of the object (True = positive class).
STATE_FIELD_TO_PAIR: dict[str, int] = {
    "isFilledWithLiquid": 0,
    "isOpen":             1,
    "isToggled":          2,
    "isPickedUp":         3,
    "isSliced":           4,
    "isCooked":           5,
    "isDirty":            6,
    "isBroken":           7,
}

# ── Model / paths ──────────────────────────────────────────────────────────────
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
EMBED_DIM       = 768    # pooler_output dim for siglip-base-patch16-224

DATASET_PATH    = ROOT / "state_classifier" / "dataset.pkl"
CHECKPOINT_DIR  = ROOT / "state_classifier" / "checkpoints"
BEST_CKPT       = CHECKPOINT_DIR / "best.pt"

MIN_CROP_PX = 8
