#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from distance_rule_adjustment import (
    DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
    resolve_center_distance_bucket_threshold_px,
)


def maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def final_aggregate_from_summary(summary_payload: dict) -> dict:
    evaluation = summary_payload.get("evaluation", {})
    final = evaluation.get("final")
    if isinstance(final, dict) and "aggregate" in final:
        return final["aggregate"]
    if isinstance(final, dict):
        return final
    if isinstance(evaluation, dict) and "aggregate" in evaluation:
        return evaluation["aggregate"]
    return evaluation


def original_final_counts(row: dict) -> dict:
    top_tp = maybe_float(row.get("tp"))
    if top_tp is not None:
        return {
            "metric_stage_used": "final",
            "original_tp": int(top_tp),
            "original_missed_detection_count": int(maybe_float(row.get("missed_detection_count")) or 0.0),
            "original_false_positive_count": int(maybe_float(row.get("false_positive_count")) or 0.0),
            "original_tn": int(maybe_float(row.get("tn")) or 0.0),
            "original_no_gt_count": int(
                maybe_float(row.get("no_gt_count"))
                if maybe_float(row.get("no_gt_count")) is not None
                else (maybe_float(row.get("no_gt_predicted")) or 0.0)
            ),
            "original_avg_false_positive_center_distance_px": float(
                maybe_float(row.get("avg_false_positive_center_distance_px")) or 0.0
            ),
        }

    return {
        "metric_stage_used": "final",
        "original_tp": int(maybe_float(row.get("final_tp")) or 0.0),
        "original_missed_detection_count": int(maybe_float(row.get("final_missed_detection_count")) or 0.0),
        "original_false_positive_count": int(maybe_float(row.get("final_false_positive_count")) or 0.0),
        "original_tn": int(maybe_float(row.get("final_tn")) or 0.0),
        "original_no_gt_count": int(
            maybe_float(row.get("final_no_gt_count"))
            if maybe_float(row.get("final_no_gt_count")) is not None
            else (maybe_float(row.get("final_no_gt_predicted")) or 0.0)
        ),
        "original_avg_false_positive_center_distance_px": float(
            maybe_float(row.get("final_avg_false_positive_center_distance_px")) or 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore a combined benchmark comparison CSV using the rule: "
            "GT-frame false positives within a center-distance threshold get their own bucket "
            "while false_positive_count remains unchanged."
        )
    )
    parser.add_argument("--comparison_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--distance_threshold_px", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_csv = args.comparison_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    cli_distance_threshold_px = (
        float(args.distance_threshold_px) if args.distance_threshold_px is not None else None
    )

    with comparison_csv.open("r", encoding="utf-8", newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    if not input_rows:
        raise ValueError(f"No rows found in {comparison_csv}")

    output_rows: List[dict] = []
    for row in input_rows:
        output_dir = Path(row["output_dir"]).expanduser().resolve()
        summary_path = output_dir / "experiment_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing experiment summary for {row['run_name']}: {summary_path}")

        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        final_aggregate = final_aggregate_from_summary(summary_payload)
        distance_map = final_aggregate.get("false_positive_center_distance_by_frame_px", {})
        if not isinstance(distance_map, dict):
            distance_map = {}

        annotations_path_value = (
            summary_payload.get("annotations_path")
            or summary_payload.get("evaluation", {}).get("annotations_path")
        )
        annotations_path = Path(annotations_path_value).expanduser().resolve() if annotations_path_value else None
        ball_category_id = int(
            summary_payload.get("ball_category_id")
            or summary_payload.get("evaluation", {}).get("ball_category_id")
            or 1
        )
        distance_threshold_px = (
            cli_distance_threshold_px
            if cli_distance_threshold_px is not None
            else resolve_center_distance_bucket_threshold_px(
                annotations_path=annotations_path,
                ball_category_id=ball_category_id,
                fallback_threshold_px=DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
            )
        )

        base = original_final_counts(row)
        recovered_values = [
            float(distance_px)
            for distance_px in distance_map.values()
            if float(distance_px) <= distance_threshold_px
        ]
        remaining_values = [
            float(distance_px)
            for distance_px in distance_map.values()
            if float(distance_px) > distance_threshold_px
        ]
        recovered_count = len(recovered_values)

        adjusted_tp = base["original_tp"]
        adjusted_fp = base["original_false_positive_count"]
        adjusted_missed = base["original_missed_detection_count"]
        adjusted_tn = base["original_tn"]
        adjusted_no_gt_count = base["original_no_gt_count"]

        gt_frames = adjusted_tp + adjusted_missed
        adjusted_tp_over_gt_frames = float(adjusted_tp / gt_frames) if gt_frames else 0.0
        adjusted_tp_over_tp_plus_fp = (
            float(adjusted_tp / (adjusted_tp + adjusted_fp)) if (adjusted_tp + adjusted_fp) else 0.0
        )
        adjusted_tp_over_tp_plus_fp_plus_no_gt = (
            float(adjusted_tp / (adjusted_tp + adjusted_fp + adjusted_no_gt_count))
            if (adjusted_tp + adjusted_fp + adjusted_no_gt_count)
            else 0.0
        )
        remaining_avg_distance = base["original_avg_false_positive_center_distance_px"]

        updated_row = dict(row)
        updated_row.update(
            {
                "distance_threshold_px": distance_threshold_px,
                "metric_stage_used": base["metric_stage_used"],
                "distance_le_threshold_px_count": recovered_count,
                "distance_le_50_px_count": recovered_count,
                "original_tp": base["original_tp"],
                "original_missed_detection_count": base["original_missed_detection_count"],
                "original_false_positive_count": base["original_false_positive_count"],
                "original_tn": base["original_tn"],
                "original_no_gt_count": base["original_no_gt_count"],
                "original_avg_false_positive_center_distance_px": base[
                    "original_avg_false_positive_center_distance_px"
                ],
                "tp": adjusted_tp,
                "missed_detection_count": adjusted_missed,
                "false_positive_count": adjusted_fp,
                "tn": adjusted_tn,
                "no_gt_count": adjusted_no_gt_count,
                "remaining_false_positive_avg_distance_px": remaining_avg_distance,
                "adjusted_tp_over_gt_frames": adjusted_tp_over_gt_frames,
                "adjusted_tp_over_tp_plus_fp": adjusted_tp_over_tp_plus_fp,
                "adjusted_tp_over_tp_plus_fp_plus_no_gt_count": adjusted_tp_over_tp_plus_fp_plus_no_gt,
            }
        )
        output_rows.append(updated_row)

    output_rows.sort(
        key=lambda item: (
            -int(item["tp"]),
            int(item["false_positive_count"]),
            int(item["missed_detection_count"]),
            float(maybe_float(item.get("latency_ms")) or 1e18),
        )
    )
    for index, row in enumerate(output_rows, start=1):
        row["rank_by_adjusted_rule"] = index

    preferred_columns = [
        "rank_by_adjusted_rule",
        "run_name",
        "benchmark_name",
        "output_dir",
        "distance_threshold_px",
        "metric_stage_used",
        "distance_le_threshold_px_count",
        "distance_le_50_px_count",
        "original_tp",
        "original_missed_detection_count",
        "original_false_positive_count",
        "original_tn",
        "original_no_gt_count",
        "tp",
        "missed_detection_count",
        "false_positive_count",
        "tn",
        "no_gt_count",
        "remaining_false_positive_avg_distance_px",
        "adjusted_tp_over_gt_frames",
        "adjusted_tp_over_tp_plus_fp",
        "adjusted_tp_over_tp_plus_fp_plus_no_gt_count",
        "latency_ms",
        "resolution",
        "confidence_threshold",
        "match_iou",
    ]
    existing_fieldnames = list(output_rows[0].keys())
    fieldnames = preferred_columns + [name for name in existing_fieldnames if name not in preferred_columns]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"[INFO] Wrote {len(output_rows)} rescored rows to {output_csv} "
        f"using distance_threshold_px={distance_threshold_px}"
    )


if __name__ == "__main__":
    main()
