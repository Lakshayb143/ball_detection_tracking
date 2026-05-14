#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _eval_yolo_ball_only_common import main  # noqa: E402


if __name__ == "__main__":
    main(
        default_run_name="yolo_ball_only_v1",
        default_output_root=REPO_ROOT / "outputs" / "yolo_ball_only",
        default_model_path=REPO_ROOT / "samy_models" / "best_combined_v2.pt",
        default_image_size=1280,
    )
