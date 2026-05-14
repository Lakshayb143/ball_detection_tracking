#!/usr/bin/env python3
"""
Explicit full-frame fallback entrypoint for the RF-DETR + outlier + SST benchmark.

This reuses the same logic as benchmark_rfdetr_outlier_sst_fallback_tracking.py,
but writes to a separate output directory so it can be compared cleanly against
crop-based fallback experiments.
"""

from __future__ import annotations

import benchmark_rfdetr_outlier_sst_fallback_tracking as base


base.DEFAULT_OUTPUT_ROOT = (
    base.REPO_ROOT / "outputs" / "rfdetr_outlier_sst_fullframe_fallback_tracking"
)
base.DEFAULT_RUN_NAME = "rfdetr_outlier_sst_fullframe_fallback_v1"


if __name__ == "__main__":
    base.main()
