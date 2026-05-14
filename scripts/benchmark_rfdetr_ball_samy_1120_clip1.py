#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _eval_rfdetr_ball_only_common as base


base.DEFAULT_DATA_ROOT = base.REPO_ROOT / "train"
CLIP1_FRESH_OUTPUT_ROOT = base.REPO_ROOT / "clip1_fresh_runs"


if __name__ == "__main__":
    base.main(
        default_run_name="rfdetr_samy_1120_only__clip1",
        default_output_root=CLIP1_FRESH_OUTPUT_ROOT,
        default_model_path=base.REPO_ROOT / "checkpoints" / "ball_samy_1120.pth",
        default_resolution=1120,
    )
