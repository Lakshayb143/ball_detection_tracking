#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

from distance_rule_adjustment import (
    build_distance_rule_adjusted_metrics,
    resolve_center_distance_bucket_threshold_px,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "clip1_fresh_runs"
DEFAULT_PYTORCH_RUN_DIR = DEFAULT_OUTPUT_ROOT / "rfdetr_samy_1120_only__clip1"
DEFAULT_TRT_RUN_DIR = DEFAULT_OUTPUT_ROOT / "tensorrt_samy_1120_only__clip1"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_ROOT / "samy_1120_pytorch_vs_tensorrt__clip1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare standalone PyTorch samy_1120 vs TensorRT samy_1120 runs on clip1."
    )
    parser.add_argument("--pytorch_run_dir", type=Path, default=DEFAULT_PYTORCH_RUN_DIR)
    parser.add_argument("--trt_run_dir", type=Path, default=DEFAULT_TRT_RUN_DIR)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_run(run_dir: Path, backend_label: str) -> Dict[str, object]:
    summary = load_json(run_dir / "experiment_summary.json")
    evaluation = summary.get("evaluation", {})
    if not isinstance(evaluation, dict) or evaluation.get("status") != "ok":
        raise ValueError(f"Evaluation summary is not usable for {run_dir}")

    annotations_path_raw = evaluation.get("annotations_path")
    annotations_path = Path(annotations_path_raw).expanduser().resolve() if annotations_path_raw else None
    ball_category_id = int(evaluation.get("ball_category_id", 1))
    threshold_px = resolve_center_distance_bucket_threshold_px(
        annotations_path=annotations_path,
        ball_category_id=ball_category_id,
    )
    adjusted = build_distance_rule_adjusted_metrics(
        evaluation["final"]["aggregate"],
        center_distance_bucket_threshold_px=threshold_px,
    )
    aggregate = summary.get("aggregate", {})
    config = summary.get("config", {})

    return {
        "backend": backend_label,
        "run_name": str(summary.get("run_name", run_dir.name)),
        "model_path": str(summary.get("model_path", "")),
        "resolution": summary.get("resolution", config.get("resolution", config.get("input_size", ""))),
        "confidence_threshold": config.get(
            "confidence_threshold",
            config.get("confidence", ""),
        ),
        "distance_threshold_px": float(adjusted["center_distance_bucket_threshold_px"]),
        "tp": int(adjusted["tp"]),
        "missed_detection_count": int(adjusted["missed_detection_count"]),
        "false_positive_count": int(adjusted["false_positive_count"]),
        "distance_le_threshold_px_count": int(adjusted["distance_le_threshold_px_count"]),
        "tn": int(adjusted["tn"]),
        "no_gt_count": int(adjusted["no_gt_count"]),
        "latency_ms": float(aggregate.get("inference_ms_avg", 0.0)),
        "runtime_fps": float(aggregate.get("runtime_fps", 0.0)),
    }


def main() -> None:
    args = parse_args()
    pytorch_run_dir = args.pytorch_run_dir.expanduser().resolve()
    trt_run_dir = args.trt_run_dir.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    rows = [
        row_from_run(pytorch_run_dir, "pytorch"),
        row_from_run(trt_run_dir, "tensorrt"),
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote comparison CSV: {output_csv}")
    for row in rows:
        print(
            f"[INFO] {row['backend']}: tp={row['tp']}, missed={row['missed_detection_count']}, "
            f"fp={row['false_positive_count']}, near_fp={row['distance_le_threshold_px_count']}, "
            f"latency_ms={row['latency_ms']:.2f}, fps={row['runtime_fps']:.2f}"
        )


if __name__ == "__main__":
    main()
