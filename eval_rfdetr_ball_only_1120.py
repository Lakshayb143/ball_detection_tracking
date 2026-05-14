#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _eval_rfdetr_ball_only_common import REPO_ROOT, main


if __name__ == "__main__":
    main(
        default_run_name="rfdetr_ball_only_1120_v1",
        default_output_root=Path(REPO_ROOT) / "outputs" / "rfdetr_ball_only_1120",
        default_model_path=Path(REPO_ROOT) / "checkpoints" / "ball_1120.pth",
        default_resolution=1120,
    )
