#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _eval_rfdetr_ball_only_common as base


base.DEFAULT_DATA_ROOT = base.REPO_ROOT / "benchmark_sets" / "central_test_v1"


if __name__ == "__main__":
    base.main(
        default_run_name="rfdetr_ball_legacy_default__central_test_v1",
        default_output_root=base.REPO_ROOT / "outputs" / "central_test_v1_best_models",
        default_model_path=base.REPO_ROOT / "checkpoints" / "ball.pth",
        default_resolution=None,
    )
