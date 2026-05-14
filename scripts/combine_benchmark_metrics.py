#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_OUTPUTS_ROOT = Path("outputs")
DEFAULT_COMBINED_CSV = DEFAULT_OUTPUTS_ROOT / "combined_benchmark_metrics.csv"


FIELDNAMES = [
    "benchmark_name",
    "run_name",
    "output_dir",
    "data_root",
    "requested_data_root",
    "resolved_data_root",
    "annotations_path",
    "sequence_count",
    "frames_total",
    "runtime_fps",
    "tp",
    "missed_detection_count",
    "false_positive_count",
    "distance_le_threshold_px_count",
    "distance_le_50_px_count",
    "tn",
    "no_gt_count",
    "avg_false_positive_center_distance_px",
    "raw_tp",
    "raw_missed_detection_count",
    "raw_false_positive_count",
    "raw_distance_le_threshold_px_count",
    "raw_distance_le_50_px_count",
    "raw_tn",
    "raw_no_gt_count",
    "raw_avg_false_positive_center_distance_px",
    "final_tp",
    "final_missed_detection_count",
    "final_false_positive_count",
    "final_distance_le_threshold_px_count",
    "final_distance_le_50_px_count",
    "final_tn",
    "final_no_gt_count",
    "final_avg_false_positive_center_distance_px",
    "precision",
    "recall",
    "latency_ms",
    "resolution",
    "match_iou",
    "eval_score_threshold",
    "confidence_threshold",
    "position_threshold",
    "velocity_threshold",
    "outlier_history_frames",
    "outlier_wait_frames",
    "outlier_reset_frames",
    "fps_assumption",
    "max_lost_seconds",
    "stable_track_threshold",
    "validation_gate_threshold",
    "interpolation_velocity_threshold",
    "max_interpolation_frames",
    "interpolated_box_size",
    "gdino_crop_expansion",
    "gdino_min_crop_size",
    "gdino_max_crop_size",
    "sst_crop_expansion",
    "sst_min_crop_size",
    "sst_max_crop_size",
    "enable_gdino_fallback",
    "enable_sst_fallback",
]


def first_non_empty(*values: object) -> object:
    for value in values:
        if value not in ("", None):
            return value
    return ""


def read_single_row_csv(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return {key: value for key, value in row.items()}
    return {}


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_data_root(summary: Dict[str, object]) -> object:
    return first_non_empty(
        summary.get("data_root", ""),
        summary.get("requested_data_root", ""),
        summary.get("resolved_data_root", ""),
    )


def summary_requested_data_root(summary: Dict[str, object]) -> object:
    return first_non_empty(
        summary.get("requested_data_root", ""),
        summary.get("data_root", ""),
        summary.get("resolved_data_root", ""),
    )


def summary_resolved_data_root(summary: Dict[str, object]) -> object:
    return first_non_empty(
        summary.get("resolved_data_root", ""),
        summary.get("data_root", ""),
        summary.get("requested_data_root", ""),
    )


def summary_annotations_path(summary: Dict[str, object]) -> object:
    evaluation = summary.get("evaluation", {})
    if isinstance(evaluation, dict):
        return first_non_empty(evaluation.get("annotations_path", ""))
    return ""


def aggregate_metric(aggregate: Dict[str, object], *keys: str) -> object:
    for key in keys:
        value = aggregate.get(key, "")
        if value not in ("", None):
            return value
    return ""


def build_row(metrics_path: Path) -> Dict[str, object]:
    run_dir = metrics_path.parent
    benchmark_name = run_dir.parent.name

    metrics = read_single_row_csv(metrics_path)
    summary_path = run_dir / "experiment_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}

    aggregate = summary.get("aggregate", {}) if isinstance(summary, dict) else {}
    config = summary.get("config", {}) if isinstance(summary, dict) else {}
    sequences = summary.get("sequences", []) if isinstance(summary, dict) else []

    row: Dict[str, object] = {
        "benchmark_name": benchmark_name,
        "run_name": summary.get("run_name", run_dir.name) if isinstance(summary, dict) else run_dir.name,
        "output_dir": str(run_dir),
        "data_root": summary_data_root(summary) if isinstance(summary, dict) else "",
        "requested_data_root": summary_requested_data_root(summary) if isinstance(summary, dict) else "",
        "resolved_data_root": summary_resolved_data_root(summary) if isinstance(summary, dict) else "",
        "annotations_path": summary_annotations_path(summary) if isinstance(summary, dict) else "",
        "sequence_count": len(sequences) if isinstance(sequences, list) else "",
        "frames_total": aggregate_metric(aggregate, "frames_total", "images_total") if isinstance(aggregate, dict) else "",
        "runtime_fps": aggregate_metric(aggregate, "runtime_fps") if isinstance(aggregate, dict) else "",
        "tp": first_non_empty(
            metrics.get("tp", ""),
            aggregate_metric(aggregate, "event_tp"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("tp", ""),
        "missed_detection_count": first_non_empty(
            metrics.get("missed_detection_count", ""),
            aggregate_metric(aggregate, "missed_detection_count"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("missed_detection_count", ""),
        "false_positive_count": first_non_empty(
            metrics.get("false_positive_count", ""),
            aggregate_metric(aggregate, "false_positive_count"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("false_positive_count", ""),
        "distance_le_50_px_count": first_non_empty(
            metrics.get("distance_le_threshold_px_count", ""),
            metrics.get("distance_le_50_px_count", ""),
            metrics.get("reclassified_fp_to_tp_count", ""),
        ),
        "distance_le_threshold_px_count": first_non_empty(
            metrics.get("distance_le_threshold_px_count", ""),
            metrics.get("distance_le_50_px_count", ""),
            metrics.get("reclassified_fp_to_tp_count", ""),
        ),
        "tn": first_non_empty(
            metrics.get("tn", ""),
            aggregate_metric(aggregate, "tn"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("tn", ""),
        "no_gt_count": first_non_empty(
            metrics.get("no_gt_count", ""),
            metrics.get("no_gt_predicted", ""),
            aggregate_metric(aggregate, "no_gt_predicted"),
        )
        if isinstance(aggregate, dict)
        else first_non_empty(metrics.get("no_gt_count", ""), metrics.get("no_gt_predicted", "")),
        "avg_false_positive_center_distance_px": first_non_empty(
            metrics.get("avg_false_positive_center_distance_px", ""),
            aggregate_metric(aggregate, "avg_false_positive_center_distance_px"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("avg_false_positive_center_distance_px", ""),
        "raw_tp": first_non_empty(
            metrics.get("raw_tp", ""),
            aggregate_metric(aggregate, "raw_event_tp"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("raw_tp", ""),
        "raw_missed_detection_count": first_non_empty(
            metrics.get("raw_missed_detection_count", ""),
            aggregate_metric(aggregate, "raw_missed_detection_count"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("raw_missed_detection_count", ""),
        "raw_false_positive_count": first_non_empty(
            metrics.get("raw_false_positive_count", ""),
            aggregate_metric(aggregate, "raw_false_positive_count"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("raw_false_positive_count", ""),
        "raw_distance_le_50_px_count": first_non_empty(
            metrics.get("raw_distance_le_threshold_px_count", ""),
            metrics.get("raw_distance_le_50_px_count", ""),
            metrics.get("raw_reclassified_fp_to_tp_count", ""),
        ),
        "raw_distance_le_threshold_px_count": first_non_empty(
            metrics.get("raw_distance_le_threshold_px_count", ""),
            metrics.get("raw_distance_le_50_px_count", ""),
            metrics.get("raw_reclassified_fp_to_tp_count", ""),
        ),
        "raw_tn": first_non_empty(
            metrics.get("raw_tn", ""),
            aggregate_metric(aggregate, "raw_tn"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("raw_tn", ""),
        "raw_no_gt_count": first_non_empty(
            metrics.get("raw_no_gt_count", ""),
            metrics.get("raw_no_gt_predicted", ""),
            aggregate_metric(aggregate, "raw_no_gt_predicted"),
        )
        if isinstance(aggregate, dict)
        else first_non_empty(metrics.get("raw_no_gt_count", ""), metrics.get("raw_no_gt_predicted", "")),
        "raw_avg_false_positive_center_distance_px": first_non_empty(
            metrics.get("raw_avg_false_positive_center_distance_px", ""),
            aggregate_metric(aggregate, "raw_avg_false_positive_center_distance_px"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("raw_avg_false_positive_center_distance_px", ""),
        "final_tp": first_non_empty(
            metrics.get("final_tp", ""),
            aggregate_metric(aggregate, "final_event_tp"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("final_tp", ""),
        "final_missed_detection_count": first_non_empty(
            metrics.get("final_missed_detection_count", ""),
            aggregate_metric(aggregate, "final_missed_detection_count"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("final_missed_detection_count", ""),
        "final_false_positive_count": first_non_empty(
            metrics.get("final_false_positive_count", ""),
            aggregate_metric(aggregate, "final_false_positive_count"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("final_false_positive_count", ""),
        "final_distance_le_50_px_count": first_non_empty(
            metrics.get("final_distance_le_threshold_px_count", ""),
            metrics.get("final_distance_le_50_px_count", ""),
            metrics.get("final_reclassified_fp_to_tp_count", ""),
        ),
        "final_distance_le_threshold_px_count": first_non_empty(
            metrics.get("final_distance_le_threshold_px_count", ""),
            metrics.get("final_distance_le_50_px_count", ""),
            metrics.get("final_reclassified_fp_to_tp_count", ""),
        ),
        "final_tn": first_non_empty(
            metrics.get("final_tn", ""),
            aggregate_metric(aggregate, "final_tn"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("final_tn", ""),
        "final_no_gt_count": first_non_empty(
            metrics.get("final_no_gt_count", ""),
            metrics.get("final_no_gt_predicted", ""),
            aggregate_metric(aggregate, "final_no_gt_predicted"),
        )
        if isinstance(aggregate, dict)
        else first_non_empty(metrics.get("final_no_gt_count", ""), metrics.get("final_no_gt_predicted", "")),
        "final_avg_false_positive_center_distance_px": first_non_empty(
            metrics.get("final_avg_false_positive_center_distance_px", ""),
            aggregate_metric(aggregate, "final_avg_false_positive_center_distance_px"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("final_avg_false_positive_center_distance_px", ""),
        "precision": first_non_empty(metrics.get("precision", ""), aggregate_metric(aggregate, "final_precision", "precision"))
        if isinstance(aggregate, dict)
        else metrics.get("precision", ""),
        "recall": first_non_empty(metrics.get("recall", ""), aggregate_metric(aggregate, "final_recall", "recall"))
        if isinstance(aggregate, dict)
        else metrics.get("recall", ""),
        "latency_ms": first_non_empty(
            metrics.get("latency_ms", ""),
            aggregate_metric(aggregate, "total_ms_avg", "inference_ms_avg"),
        )
        if isinstance(aggregate, dict)
        else metrics.get("latency_ms", ""),
        "resolution": (
            first_non_empty(
                config.get("resolution", ""),
                config.get("ball_model_resolution", ""),
                summary.get("resolution", ""),
            )
            if isinstance(config, dict)
            else first_non_empty(summary.get("resolution", ""))
        ),
        "match_iou": (
            config.get("match_iou", config.get("eval_iou", ""))
            if isinstance(config, dict)
            else ""
        ),
        "eval_score_threshold": config.get("eval_score_threshold", "") if isinstance(config, dict) else "",
        "confidence_threshold": (
            config.get("confidence_threshold", config.get("ball_confidence", ""))
            if isinstance(config, dict)
            else ""
        ),
        "position_threshold": config.get("position_threshold", "") if isinstance(config, dict) else "",
        "velocity_threshold": config.get("velocity_threshold", "") if isinstance(config, dict) else "",
        "outlier_history_frames": config.get("outlier_history_frames", "") if isinstance(config, dict) else "",
        "outlier_wait_frames": config.get("outlier_wait_frames", "") if isinstance(config, dict) else "",
        "outlier_reset_frames": config.get("outlier_reset_frames", "") if isinstance(config, dict) else "",
        "fps_assumption": config.get("fps_assumption", "") if isinstance(config, dict) else "",
        "max_lost_seconds": config.get("max_lost_seconds", "") if isinstance(config, dict) else "",
        "stable_track_threshold": config.get("stable_track_threshold", "") if isinstance(config, dict) else "",
        "validation_gate_threshold": config.get("validation_gate_threshold", "") if isinstance(config, dict) else "",
        "interpolation_velocity_threshold": (
            config.get("interpolation_velocity_threshold", "") if isinstance(config, dict) else ""
        ),
        "max_interpolation_frames": config.get("max_interpolation_frames", "") if isinstance(config, dict) else "",
        "interpolated_box_size": config.get("interpolated_box_size", "") if isinstance(config, dict) else "",
        "gdino_crop_expansion": config.get("gdino_crop_expansion", "") if isinstance(config, dict) else "",
        "gdino_min_crop_size": config.get("gdino_min_crop_size", "") if isinstance(config, dict) else "",
        "gdino_max_crop_size": config.get("gdino_max_crop_size", "") if isinstance(config, dict) else "",
        "sst_crop_expansion": config.get("sst_crop_expansion", "") if isinstance(config, dict) else "",
        "sst_min_crop_size": config.get("sst_min_crop_size", "") if isinstance(config, dict) else "",
        "sst_max_crop_size": config.get("sst_max_crop_size", "") if isinstance(config, dict) else "",
        "enable_gdino_fallback": config.get("enable_gdino_fallback", "") if isinstance(config, dict) else "",
        "enable_sst_fallback": config.get("enable_sst_fallback", "") if isinstance(config, dict) else "",
    }
    return row


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def matches_any(value: object, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    text = str(value)
    return any(pattern in text for pattern in patterns)


def filter_rows(
    rows: Iterable[Dict[str, object]],
    *,
    run_name_contains: Sequence[str],
    data_root_contains: Sequence[str],
) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for row in rows:
        if not matches_any(row.get("run_name", ""), run_name_contains):
            continue
        data_root_value = first_non_empty(
            row.get("resolved_data_root", ""),
            row.get("requested_data_root", ""),
            row.get("data_root", ""),
        )
        if not matches_any(data_root_value, data_root_contains):
            continue
        filtered.append(row)
    return filtered


def gather_metric_files(search_roots: Sequence[Path]) -> List[Path]:
    metric_files: List[Path] = []
    for root in search_roots:
        metric_files.extend(sorted(root.rglob("benchmark_metrics.csv")))
    unique_paths = {path.resolve(): path for path in metric_files}
    return [unique_paths[key] for key in sorted(unique_paths)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine all benchmark run metrics under outputs/ into a single CSV."
    )
    parser.add_argument("--outputs_root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_COMBINED_CSV)
    parser.add_argument(
        "--search_root",
        type=Path,
        action="append",
        default=[],
        help="Specific benchmark output root(s) to scan recursively. If omitted, --outputs_root is used.",
    )
    parser.add_argument(
        "--run_name_contains",
        type=str,
        action="append",
        default=[],
        help="Keep only rows whose run_name contains one of these substrings.",
    )
    parser.add_argument(
        "--data_root_contains",
        type=str,
        action="append",
        default=[],
        help="Keep only rows whose data_root/requested/resolved data root contains one of these substrings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    search_roots = [path.expanduser().resolve() for path in args.search_root] or [outputs_root]

    metric_files = gather_metric_files(search_roots)
    rows = [build_row(path) for path in metric_files]
    rows = filter_rows(
        rows,
        run_name_contains=args.run_name_contains,
        data_root_contains=args.data_root_contains,
    )
    rows.sort(key=lambda item: (str(item["benchmark_name"]), str(item["run_name"])))

    write_rows(output_csv, rows)
    print(
        f"[INFO] Combined {len(rows)} benchmark runs into {output_csv} "
        f"from {', '.join(str(path) for path in search_roots)}"
    )


if __name__ == "__main__":
    main()
