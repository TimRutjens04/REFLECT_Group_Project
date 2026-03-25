import os

import torch

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ALIGNED_DIR = os.path.join(ROOT, "aligned")
ENCODED_DIR = os.path.join(ROOT, "encoded")
OWL_DIR     = os.path.join(ROOT, "owl")
DATA_DIR    = os.path.join(ROOT, "data")
TASKS_JSON  = os.path.join(DATA_DIR, "tasks_real_world.json")
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
LOG_PATH    = os.path.join(ROOT, "logs", "gui.log")

DARK_BG   = "#0e1117"
GRID_COL  = "#1e2130"
LINE_VIS  = "#4c9be8"
LINE_AUD  = "#7ec87e"
LINE_DELT = "#e8754c"
CURSOR    = "#f0c040"
