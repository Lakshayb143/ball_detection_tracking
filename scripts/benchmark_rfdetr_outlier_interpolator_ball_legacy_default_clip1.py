#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import benchmark_rfdetr_outlier_interpolator_1120 as base


base.DEFAULT_DATA_ROOT = base.REPO_ROOT / "train"
base.DEFAULT_OUTPUT_ROOT = base.REPO_ROOT / "clip1_fresh_runs"
base.DEFAULT_RUN_NAME = "rfdetr_ball_legacy_default_outlier_interpolator__clip1"
base.DEFAULT_BALL_MODEL_PATH = base.REPO_ROOT / "checkpoints" / "ball.pth"
base.DEFAULT_BALL_MODEL_RESOLUTION = 0
base.BALL_CATEGORY_ID = 1


if __name__ == "__main__":
    base.main()
