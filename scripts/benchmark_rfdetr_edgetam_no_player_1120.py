#!/usr/bin/env python3
from __future__ import annotations

import benchmark_rfdetr_edgetam_fallback_tracking as base


base.DEFAULT_DATA_ROOT = base.REPO_ROOT / "benchmark_sets" / "central_test_v1"
base.DEFAULT_OUTPUT_ROOT = base.REPO_ROOT / "outputs" / "central_test_v1_best_models"
base.DEFAULT_RUN_NAME = "rfdetr_1120_edgetam_no_player__central_test_v1"
base.DEFAULT_BALL_MODEL_PATH = base.REPO_ROOT / "checkpoints" / "ball_1120.pth"
base.DEFAULT_BALL_RESOLUTION = 1120
base.DEFAULT_ENABLE_PLAYER_EXCLUSION = False
base.DEFAULT_POSITION_THRESHOLD = 50.0
base.DEFAULT_VELOCITY_THRESHOLD = 100.0


if __name__ == "__main__":
    base.main()
