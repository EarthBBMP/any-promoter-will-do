"""Paths and global settings.

This package is deliberately kept outside the original ASPINE repository so the
CC BY 4.0 artifact stays unmodified and the new contribution is separable. Data
and released weights are read from the parent repository, never written to.

Paths can be overridden by environment variable so the same code runs
unchanged on Colab, where the repository layout does not exist:

    ASPINE_DATA_DIR     directory holding train/valid/test_aspine.csv
    ASPINE_WEIGHTS      path to the released model_weight.keras
    ASPINE_RESULTS_DIR  where to write predictions and tables
"""

import os
from pathlib import Path

# aspine-newinml/aspine_bench/config.py -> aspine-newinml -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("ASPINE_DATA_DIR", REPO_ROOT / "data" / "Aspine"))
WEIGHTS_PATH = Path(
    os.environ.get("ASPINE_WEIGHTS", REPO_ROOT / "data" / "Saved_Weights" / "model_weight.keras")
)
RESULTS_DIR = Path(
    os.environ.get("ASPINE_RESULTS_DIR", Path(__file__).resolve().parents[1] / "results")
)

TRAIN_CSV = DATA_DIR / "train_aspine.csv"
VALID_CSV = DATA_DIR / "valid_aspine.csv"
TEST_CSV = DATA_DIR / "test_aspine.csv"

# Single seed used everywhere, so every number in the paper is reproducible.
SEED = 0

# Bootstrap resamples for test-set confidence intervals.
N_BOOTSTRAP = 2000
