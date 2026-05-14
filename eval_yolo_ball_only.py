#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _eval_yolo_ball_only_common import REPO_ROOT, main


if __name__ == "__main__":
    main(
        default_run_name="yolo_ball_only_v1",
        default_output_root=Path(REPO_ROOT) / "outputs" / "yolo_ball_only",
        default_model_path=Path(REPO_ROOT) / "samy_models" / "best_combined_v2.pt",
        default_image_size=1280,
    )
