#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from ball_detection_metrics import evaluate_detections
from distance_rule_adjustment import build_distance_rule_adjusted_metrics

ANN = Path("train/_annotations.coco.json")
BALL_CATEGORY_ID = 1


def four_metrics(det_path: Path) -> dict:
    detections = json.loads(det_path.read_text())["detections"]
    agg = evaluate_detections(
        detections=detections,
        annotations_path=ANN,
        stage="final",
        iou_threshold=0.01,
        score_threshold=0.01,
        ball_category_id=BALL_CATEGORY_ID,
    )["aggregate"]
    adj = build_distance_rule_adjusted_metrics(agg)
    return {
        "tp": int(agg["event_tp"]),
        "missed": int(agg["missed_detection_count"]),
        "fp_all": int(agg["false_positive_count"]) + int(agg["no_gt_predicted"]),
        "dist_le_threshold": int(adj["distance_le_threshold_px_count"]),
    }


def save_comparison_csv(path: Path, samy: dict, ball_and_ball_out: dict, ball_and_ball_out_col: str) -> None:
    fields = ["metric", "ball_samy_1120", ball_and_ball_out_col]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for metric in samy:
            w.writerow({
                "metric": metric,
                "ball_samy_1120": samy[metric],
                ball_and_ball_out_col: ball_and_ball_out[metric],
            })
    print(f"Saved {path}")


if __name__ == "__main__":
    samy = four_metrics(Path("clip1_samy1120_standalone.json"))
    ball_only = four_metrics(Path("clip1_fresh_runs/rfdetr_ball_and_ball_out_1120__clip1/detections.json"))
    both = four_metrics(Path("clip1_fresh_runs/rfdetr_ball_and_ball_out_1120_both_classes__clip1/detections.json"))

    save_comparison_csv(
        Path("clip1_ball_and_ball_out_ball_class_only.csv"),
        samy,
        ball_only,
        "ball_and_ball_out_1120_ball_only",
    )
    save_comparison_csv(
        Path("clip1_ball_and_ball_out_both_classes.csv"),
        samy,
        both,
        "ball_and_ball_out_1120_both_classes",
    )
