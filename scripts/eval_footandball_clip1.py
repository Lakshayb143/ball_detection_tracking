#!/usr/bin/env python3
"""
Evaluate FootAndBall ball detections on clip1 using IoU-based matching (threshold 0.01).
Follows metrics contract from CONTEXT.md.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = REPO_ROOT / 'train' / '_annotations.coco.json'
BOX_SIZE = 20  # Standard box size for FootAndBall point detections


def iou_xyxy(box_a: List[float], box_b: List[float]) -> float:
    """Calculate IoU between two boxes in xyxy format."""
    x_a = max(float(box_a[0]), float(box_b[0]))
    y_a = max(float(box_a[1]), float(box_b[1]))
    x_b = min(float(box_a[2]), float(box_b[2]))
    y_b = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1]))
    area_b = (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def center_distance_px(box_a: List[float], box_b: List[float]) -> float:
    """Calculate center-to-center distance between two xyxy boxes."""
    cx_a = (float(box_a[0]) + float(box_a[2])) / 2.0
    cy_a = (float(box_a[1]) + float(box_a[3])) / 2.0
    cx_b = (float(box_b[0]) + float(box_b[2])) / 2.0
    cy_b = (float(box_b[1]) + float(box_b[3])) / 2.0
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def load_coco_gt(coco_path: Path) -> Dict[int, List[Dict]]:
    """
    Load COCO GT and extract ball annotations by frame.
    Returns dict: frame_idx (0-indexed) -> list of GT boxes (xyxy format)
    """
    with open(coco_path) as f:
        coco = json.load(f)

    # Find ball category
    ball_cat_id = None
    for cat in coco['categories']:
        if cat['name'] == 'ball':
            ball_cat_id = cat['id']
            break

    if ball_cat_id is None:
        raise ValueError("Ball category not found in COCO annotations")

    # Build image ID to frame number mapping
    img_id_to_frame = {}
    for img in coco['images']:
        match = re.search(r'frame_(\d+)', img['file_name'])
        if match:
            coco_frame = int(match.group(1))
            tracker_frame = coco_frame - 1  # COCO is 1-indexed
            img_id_to_frame[img['id']] = tracker_frame

    # Collect GT boxes by frame
    gt_by_frame: Dict[int, List[Dict]] = {}
    for ann in coco['annotations']:
        if ann['category_id'] == ball_cat_id:
            image_id = ann['image_id']
            if image_id in img_id_to_frame:
                frame_idx = img_id_to_frame[image_id]
                bbox_xywh = ann['bbox']
                # Convert COCO xywh to xyxy
                x, y, w, h = float(bbox_xywh[0]), float(bbox_xywh[1]), float(bbox_xywh[2]), float(bbox_xywh[3])
                bbox_xyxy = [x, y, x + w, y + h]

                if frame_idx not in gt_by_frame:
                    gt_by_frame[frame_idx] = []
                gt_by_frame[frame_idx].append({'bbox': bbox_xyxy})

    return gt_by_frame


def load_footandball_detections(json_path: Path) -> Dict[int, List[Dict]]:
    """Load FootAndBall detections and convert to boxes."""
    with open(json_path) as f:
        detections = json.load(f)

    # Convert point detections to boxes
    dets_by_frame: Dict[int, List[Dict]] = {}
    for det in detections:
        frame_idx = det['frame_idx']
        x = det['x']
        y = det['y']
        confidence = det['confidence']

        # Convert center point to xyxy box
        x1 = x - BOX_SIZE / 2
        y1 = y - BOX_SIZE / 2
        x2 = x + BOX_SIZE / 2
        y2 = y + BOX_SIZE / 2
        bbox = [x1, y1, x2, y2]

        if frame_idx not in dets_by_frame:
            dets_by_frame[frame_idx] = []
        dets_by_frame[frame_idx].append({
            'bbox': bbox,
            'confidence': confidence,
            'x': x,
            'y': y
        })

    # Sort detections by confidence (descending) for each frame
    for frame_idx in dets_by_frame:
        dets_by_frame[frame_idx].sort(key=lambda d: d['confidence'], reverse=True)

    return dets_by_frame


def evaluate_footandball(match_iou_threshold: float = 0.01):
    """
    Evaluate FootAndBall detections with IoU-based matching.
    Implements metrics contract from CONTEXT.md.
    """
    detections_path = REPO_ROOT / 'detections_footandball' / 'clip1.json'
    annotations_path = DEFAULT_ANNOTATIONS

    print(f"Loading FootAndBall detections from {detections_path}")
    dets_by_frame = load_footandball_detections(detections_path)
    total_dets = sum(len(d) for d in dets_by_frame.values())
    print(f"Found {total_dets} total detections across {len(dets_by_frame)} frames")

    print(f"\nLoading COCO GT from {annotations_path}")
    gt_by_frame = load_coco_gt(annotations_path)
    print(f"Found {len(gt_by_frame)} GT frames with ball")

    # Evaluate
    print("\n" + "="*70)
    print("FOOTANDBALL STANDALONE METRICS (clip1)")
    print("="*70)

    tp = 0
    missed = 0
    false_positive_on_gt_frames = 0
    no_gt_predicted = 0

    # Evaluate all GT frames
    for frame_idx in sorted(gt_by_frame.keys()):
        gt_boxes = gt_by_frame[frame_idx]

        if frame_idx in dets_by_frame:
            # Frame has prediction(s)
            dets = dets_by_frame[frame_idx]
            top_det = dets[0]  # Top-scored detection

            # Check if top detection matches any GT box
            max_iou = 0.0
            for gt_box_dict in gt_boxes:
                iou = iou_xyxy(top_det['bbox'], gt_box_dict['bbox'])
                max_iou = max(max_iou, iou)

            if max_iou >= match_iou_threshold:
                tp += 1
            else:
                false_positive_on_gt_frames += 1
        else:
            # Frame has GT but no prediction
            missed += 1

    # Count no-GT predictions
    for frame_idx in dets_by_frame.keys():
        if frame_idx not in gt_by_frame:
            no_gt_predicted += len(dets_by_frame[frame_idx])

    # Metrics
    fp_all = false_positive_on_gt_frames + no_gt_predicted
    total_gt_frames = len(gt_by_frame)

    print(f"\nMatch IoU threshold:           {match_iou_threshold}")
    print(f"Total GT frames:               {total_gt_frames}")
    print(f"\n✓ TP (True Positives):         {tp}")
    print(f"✗ Missed:                      {missed}")
    print(f"✗ FP (GT frames):              {false_positive_on_gt_frames}")
    print(f"✗ No-GT predicted:             {no_gt_predicted}")
    print(f"✗ FP (all):                    {fp_all}")

    if total_gt_frames > 0:
        recall = tp / total_gt_frames
        print(f"\nRecall (TP / total GT):        {recall:.4f}")

    if (tp + fp_all) > 0:
        precision = tp / (tp + fp_all)
        print(f"Precision (TP / (TP + FP)):    {precision:.4f}")

    if total_gt_frames > 0 and (tp + fp_all) > 0:
        recall = tp / total_gt_frames
        precision = tp / (tp + fp_all)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"F1:                            {f1:.4f}")

    print("\n" + "="*70)


if __name__ == '__main__':
    evaluate_footandball(match_iou_threshold=0.01)
