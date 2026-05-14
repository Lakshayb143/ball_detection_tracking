#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "standalone_ball_model_matrix"
DEFAULT_DEVICE = "0"
DEFAULT_EVAL_IOU = 0.01
DEFAULT_THRESHOLDS = [0.05, 0.5]


def slug_for_threshold(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_experiment_summary(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_runs(device: str, eval_iou: float) -> List[Dict[str, object]]:
    datasets = [
        {
            "name": "test",
            "data_root": str(REPO_ROOT / "test"),
            "annotations": None,
        },
        {
            "name": "samy_test",
            "data_root": str(REPO_ROOT / "samy_combined_ball_dataset" / "test"),
            "annotations": None,
        },
        {
            "name": "soccer_tracking_6_merged",
            "data_root": str(REPO_ROOT / "Soccer-Tracking-6"),
            "annotations": str(REPO_ROOT / "Soccer-Tracking-6" / "_annotations_merged.coco.json"),
        },
    ]
    models = [
        {
            "name": "yolo_best_combined_v2",
            "entrypoint": str(REPO_ROOT / "eval_yolo_ball_only.py"),
            "extra_args": ["--device", device],
            "thresholds": [0.05, 0.5],
        },
        {
            "name": "rfdetr_ball",
            "entrypoint": str(REPO_ROOT / "eval_rfdetr_ball_only.py"),
            "extra_args": [],
            "thresholds": [0.5],
        },
        {
            "name": "rfdetr_ball_1120",
            "entrypoint": str(REPO_ROOT / "eval_rfdetr_ball_only_1120.py"),
            "extra_args": [],
            "thresholds": [0.5],
        },
    ]

    runs: List[Dict[str, object]] = []
    for dataset in datasets:
        for model in models:
            for threshold in model.get("thresholds", DEFAULT_THRESHOLDS):
                run_name = f"{model['name']}__{dataset['name']}__conf_{slug_for_threshold(threshold)}"
                cmd = [
                    sys.executable,
                    str(model["entrypoint"]),
                    "--data_root",
                    str(dataset["data_root"]),
                    "--output_root",
                    str(OUTPUT_ROOT),
                    "--run_name",
                    run_name,
                    "--confidence_threshold",
                    str(threshold),
                    "--eval_score_threshold",
                    str(threshold),
                    "--eval_iou",
                    str(eval_iou),
                ]
                if dataset.get("annotations"):
                    cmd.extend(["--annotations", str(dataset["annotations"])])
                cmd.extend(model.get("extra_args", []))
                runs.append(
                    {
                        "dataset_name": dataset["name"],
                        "dataset_root": dataset["data_root"],
                        "annotations_path": dataset.get("annotations"),
                        "model_name": model["name"],
                        "run_name": run_name,
                        "threshold": threshold,
                        "command": cmd,
                    }
                )
    return runs


def write_summary(rows: List[Dict[str, object]], summary_path: Path) -> None:
    if not rows:
        return
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone ball-detector benchmark matrix.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--eval_iou", type=float, default=DEFAULT_EVAL_IOU)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    ensure_dir(OUTPUT_ROOT)
    runs = build_runs(device=args.device, eval_iou=args.eval_iou)
    summary_rows: List[Dict[str, object]] = []

    print(f"[INFO] Output root: {OUTPUT_ROOT}")
    print(f"[INFO] Planned runs: {len(runs)}")

    for index, run in enumerate(runs, start=1):
        run_dir = OUTPUT_ROOT / str(run["run_name"])
        summary_path = run_dir / "experiment_summary.json"
        print(
            f"[INFO] ({index}/{len(runs)}) {run['run_name']} "
            f"[dataset={run['dataset_name']}, threshold={run['threshold']}]"
        )

        if args.skip_existing and summary_path.exists():
            print(f"[INFO] Skipping existing run: {run['run_name']}")
        else:
            started_at = time.perf_counter()
            subprocess.run(run["command"], check=True, cwd=REPO_ROOT)
            elapsed_s = time.perf_counter() - started_at
            print(f"[INFO] Completed in {elapsed_s:.1f}s")

        experiment = read_experiment_summary(summary_path)
        aggregate = experiment.get("aggregate", {})
        evaluation = experiment.get("evaluation", {})
        summary_rows.append(
            {
                "run_name": run["run_name"],
                "model_name": run["model_name"],
                "dataset_name": run["dataset_name"],
                "dataset_root": run["dataset_root"],
                "annotations_path": run.get("annotations_path"),
                "confidence_threshold": run["threshold"],
                "eval_iou": args.eval_iou,
                "precision": aggregate.get("precision", 0.0),
                "recall": aggregate.get("recall", 0.0),
                "mean_matched_iou": aggregate.get("mean_matched_iou", 0.0),
                "detections": aggregate.get("detections", 0),
                "frames_total": aggregate.get("frames_total", 0),
                "runtime_fps": aggregate.get("runtime_fps", 0.0),
                "inference_ms_avg": aggregate.get("inference_ms_avg", 0.0),
                "ball_category_id": evaluation.get("ball_category_id"),
                "resolved_data_root": experiment.get("resolved_data_root"),
                "experiment_summary_path": str(summary_path),
            }
        )

        write_summary(summary_rows, OUTPUT_ROOT / "matrix_summary.csv")

    print(f"[INFO] Matrix summary written to {OUTPUT_ROOT / 'matrix_summary.csv'}")


if __name__ == "__main__":
    main()
