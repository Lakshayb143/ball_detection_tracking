#!/usr/bin/env python3
from __future__ import annotations

import benchmark_rfdetr_outlier_gdino_fallback_tracking as base


base.DEFAULT_OUTPUT_ROOT = base.REPO_ROOT / "outputs" / "rfdetr_1120_shortlist"
base.DEFAULT_RUN_NAME = "rfdetr_1120_outlier_position_only_v1"
base.DEFAULT_BALL_MODEL_PATH = base.REPO_ROOT / "checkpoints" / "ball_1120.pth"
base.DEFAULT_BALL_MODEL_RESOLUTION = 1120
base.DEFAULT_ENABLE_GDINO_FALLBACK = False
base.DEFAULT_POSITION_THRESHOLD = 60.0
base.DEFAULT_VELOCITY_THRESHOLD = 1000000.0


if __name__ == "__main__":
    base.main()
