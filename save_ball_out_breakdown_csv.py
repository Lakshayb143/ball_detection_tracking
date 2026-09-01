#!/usr/bin/env python3
"""Task 3: CSV with ball_samy_1120 vs ball_and_ball_out split into ball / ball_out sub-columns.

ball sub-col  : top-1 confidence class-0 detection per frame (standard event metric).
ball_out sub-col: every class-1 detection evaluated independently (Option B):
  - Sort dets by confidence desc.
  - First det with IoU >= IOU_THRESHOLD against GT → 1 TP; rest of dets in frame → FP.
  - If no det matches GT → all dets are FP (false_positive_count).
  - Frame with no dets when GT exists → missed.
  - Dets in frames with no GT → no_gt_predicted (each counts as +1).
  - dist_le_threshold: each FP det (when GT exists) where center_distance <= perimeter of that det.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from ball_detection_metrics import (
    center_distance_px,
    evaluate_detections,
    iou_xyxy,
    load_coco_ball_ground_truth,
    perimeter_of_xyxy,
    xywh_to_xyxy,
)
from distance_rule_adjustment import build_distance_rule_adjusted_metrics

ANN = Path("train/_annotations.coco.json")
BALL_CATEGORY_ID = 1
IOU = 0.01
SCORE_THRESHOLD = 0.01


def standard_four_metrics(det_path: Path) -> dict:
    detections = json.loads(det_path.read_text())["detections"]
    agg = evaluate_detections(
        detections=detections,
        annotations_path=ANN,
        stage="final",
        iou_threshold=IOU,
        score_threshold=SCORE_THRESHOLD,
        ball_category_id=BALL_CATEGORY_ID,
    )["aggregate"]
    adj = build_distance_rule_adjusted_metrics(agg)
    return {
        "tp": int(agg["event_tp"]),
        "missed": int(agg["missed_detection_count"]),
        "fp_all": int(agg["false_positive_count"]) + int(agg["no_gt_predicted"]),
        "dist_le_threshold": int(adj["distance_le_threshold_px_count"]),
    }


def ball_out_option_b_metrics(det_path: Path) -> dict:
    """Evaluate every ball_out detection independently per frame (Option B)."""
    detections = json.loads(det_path.read_text())["detections"]
    gt_data = load_coco_ball_ground_truth(ANN, ball_category_id=BALL_CATEGORY_ID)

    # Group detections by image_id, filtered by stage and score
    dets_by_image = defaultdict(list)
    for det in detections:
        if det.get("stage") != "final":
            continue
        if float(det.get("score", 0)) < SCORE_THRESHOLD:
            continue
        iid = det.get("image_id")
        if iid is not None and int(iid) in gt_data:
            dets_by_image[int(iid)].append(det)

    # Sort each frame's dets by confidence desc
    for iid in dets_by_image:
        dets_by_image[iid].sort(key=lambda d: float(d.get("score", 0)), reverse=True)

    tp = 0
    missed = 0
    fp_count = 0       # GT exists, det exists, no IoU match (or extra dets after first match)
    no_gt_predicted = 0  # no GT, det exists
    dist_le_threshold = 0

    for image_id, gt_image in gt_data.items():
        gt_boxes = gt_image.boxes_xyxy
        gt_box = gt_boxes[0] if gt_boxes else None
        frame_dets = dets_by_image.get(image_id, [])

        if gt_box is None:
            # No GT in this frame — every det is a false alarm
            no_gt_predicted += len(frame_dets)
        else:
            if not frame_dets:
                missed += 1
            else:
                gt_matched = False
                for det in frame_dets:
                    det_box = xywh_to_xyxy(det["bbox_xywh"])
                    iou_val = iou_xyxy(det_box, gt_box)
                    if not gt_matched and iou_val >= IOU:
                        tp += 1
                        gt_matched = True
                    else:
                        # This detection is a FP
                        fp_count += 1
                        dist = center_distance_px(det_box, gt_box)
                        perim = perimeter_of_xyxy(det_box)
                        if dist <= perim:
                            dist_le_threshold += 1

    return {
        "tp": tp,
        "missed": missed,
        "fp_all": fp_count + no_gt_predicted,
        "dist_le_threshold": dist_le_threshold,
    }


def save_csv(path: Path, rows: list, fields: list) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


if __name__ == "__main__":
    samy = standard_four_metrics(Path("clip1_samy1120_standalone.json"))
    ball_col = standard_four_metrics(
        Path("clip1_fresh_runs/rfdetr_ball_and_ball_out_1120__clip1/detections.json")
    )
    ball_out_col = ball_out_option_b_metrics(
        Path("clip1_fresh_runs/rfdetr_ball_out_only__clip1/detections.json")
    )

    fields = ["metric", "ball_samy_1120", "ball_and_ball_out__ball", "ball_and_ball_out__ball_out"]
    rows = []
    for metric in samy:
        rows.append({
            "metric": metric,
            "ball_samy_1120": samy[metric],
            "ball_and_ball_out__ball": ball_col[metric],
            "ball_and_ball_out__ball_out": ball_out_col[metric],
        })

    save_csv(Path("clip1_ball_out_breakdown.csv"), rows, fields)

    print(f"\n{'metric':<20} {'ball_samy_1120':>18} {'ball_and_ball_out__ball':>24} {'ball_and_ball_out__ball_out':>28}")
    for r in rows:
        print(f"{r['metric']:<20} {r['ball_samy_1120']:>18} {r['ball_and_ball_out__ball']:>24} {r['ball_and_ball_out__ball_out']:>28}")
