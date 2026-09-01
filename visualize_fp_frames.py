#!/usr/bin/env python3
"""Task 2: Visualize the 72 FP frames from the both-classes eval.

For each frame where GT exists and the top prediction has IoU < 0.01:
- Draw GT box in green
- Draw prediction box in red
- Draw a line between box centers with distance label
- Save to fp_visualizations/frame_XXXX.png
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from ball_detection_metrics import (
    center_of_xyxy,
    evaluate_detections,
    iou_xyxy,
    load_coco_ball_ground_truth,
    xywh_to_xyxy,
)

REPO_ROOT = Path(__file__).parent
TRAIN_DIR = REPO_ROOT / "train"
ANN = TRAIN_DIR / "_annotations.coco.json"
DETECTIONS_PATH = REPO_ROOT / "clip1_fresh_runs" / "rfdetr_ball_and_ball_out_1120_both_classes__clip1" / "detections.json"
OUT_DIR = REPO_ROOT / "fp_visualizations"
BALL_CATEGORY_ID = 1
IOU = 0.01
SCORE_THRESHOLD = 0.01

COLOUR_GT = (0, 200, 0)       # green
COLOUR_PRED = (0, 0, 220)     # red
COLOUR_LINE = (0, 220, 220)   # yellow


def center_distance(box_a, box_b):
    cx_a, cy_a = center_of_xyxy(box_a)
    cx_b, cy_b = center_of_xyxy(box_b)
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def draw_box(img, box, colour, label):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    pos = (x1, max(y1 - 8, 16))
    cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = json.loads(DETECTIONS_PATH.read_text())
    detections = payload["detections"]

    # Run evaluate_detections to find FP frames
    result = evaluate_detections(
        detections=detections,
        annotations_path=ANN,
        stage="final",
        iou_threshold=IOU,
        score_threshold=SCORE_THRESHOLD,
        ball_category_id=BALL_CATEGORY_ID,
    )
    fp_frame_keys = set(result["aggregate"]["false_positive_center_distance_by_frame_px"].keys())
    print(f"FP frames to visualize: {len(fp_frame_keys)}")

    # Build frame_number -> image metadata mapping from COCO annotations
    ann_data = json.loads(ANN.read_text())
    frame_re = re.compile(r"frame_(\d+)", re.IGNORECASE)
    image_by_frame_num = {}  # frame_number (int) -> image dict
    for img in ann_data["images"]:
        m = frame_re.search(img["file_name"])
        if m:
            image_by_frame_num[int(m.group(1))] = img

    # Build GT boxes by image_id
    gt_data = load_coco_ball_ground_truth(ANN, ball_category_id=BALL_CATEGORY_ID)

    # Build detections by image_id, sorted by score desc
    dets_by_image = defaultdict(list)
    for det in detections:
        if det.get("stage") != "final":
            continue
        if float(det.get("score", 0)) < SCORE_THRESHOLD:
            continue
        iid = det.get("image_id")
        if iid is not None:
            dets_by_image[int(iid)].append(det)
    for iid in dets_by_image:
        dets_by_image[iid].sort(key=lambda d: float(d.get("score", 0)), reverse=True)

    saved = 0
    for frame_key in sorted(fp_frame_keys, key=lambda k: int(k) if k.isdigit() else 0):
        frame_num = int(frame_key) if frame_key.isdigit() else None
        if frame_num is None:
            print(f"  Skipping non-numeric frame key: {frame_key}")
            continue

        img_meta = image_by_frame_num.get(frame_num)
        if img_meta is None:
            print(f"  No image metadata for frame {frame_num}")
            continue

        img_path = TRAIN_DIR / img_meta["file_name"]
        if not img_path.exists():
            print(f"  Image not found: {img_path}")
            continue

        image_id = img_meta["id"]
        gt_image = gt_data.get(image_id)
        gt_boxes = gt_image.boxes_xyxy if gt_image else []
        gt_box = gt_boxes[0] if gt_boxes else None

        image_dets = dets_by_image.get(image_id, [])
        chosen_det = image_dets[0] if image_dets else None

        if gt_box is None or chosen_det is None:
            print(f"  Frame {frame_num}: missing GT or prediction, skipping")
            continue

        pred_box = xywh_to_xyxy(chosen_det["bbox_xywh"])
        dist = center_distance(pred_box, gt_box)
        score = float(chosen_det.get("score", 0))
        iou_val = iou_xyxy(pred_box, gt_box)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not read image: {img_path}")
            continue

        draw_box(img, gt_box, COLOUR_GT, "GT")
        draw_box(img, pred_box, COLOUR_PRED, f"pred {score:.2f}")

        # Line between centers with distance label
        cx_gt, cy_gt = center_of_xyxy(gt_box)
        cx_p, cy_p = center_of_xyxy(pred_box)
        cv2.line(img, (int(cx_gt), int(cy_gt)), (int(cx_p), int(cy_p)), COLOUR_LINE, 2)
        mid_x = int((cx_gt + cx_p) / 2)
        mid_y = int((cy_gt + cy_p) / 2)
        dist_label = f"dist={dist:.1f}px  iou={iou_val:.4f}"
        cv2.putText(img, dist_label, (mid_x + 4, mid_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, dist_label, (mid_x + 4, mid_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOUR_LINE, 2)

        out_file = OUT_DIR / f"frame_{frame_num:04d}.png"
        cv2.imwrite(str(out_file), img)
        saved += 1

    print(f"Saved {saved} frames to {OUT_DIR}/")


if __name__ == "__main__":
    main()
