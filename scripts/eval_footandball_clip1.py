#!/usr/bin/env python3
"""
Evaluate FootAndBall ball detections on clip1.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = REPO_ROOT / 'train' / '_annotations.coco.json'


def load_footandball_detections(json_path):
    """Load FootAndBall detections from JSON."""
    with open(json_path) as f:
        detections = json.load(f)

    # Convert to tracker format (same as v4)
    tracker_output = []
    for det in detections:
        tracker_output.append({
            'frame_idx': det['frame_idx'],
            'x': det['x'],
            'y': det['y'],
            'confidence': det['confidence'],
        })

    return tracker_output


def convert_to_coco_format(tracker_points, box_size=20):
    """Convert tracker points to COCO format."""
    coco_detections = []

    for point in tracker_points:
        frame_idx = point['frame_idx']
        x = point['x']
        y = point['y']
        confidence = point['confidence']

        # Convert center point to COCO bbox format [x, y, width, height]
        x1 = x - box_size / 2
        y1 = y - box_size / 2
        bbox = [x1, y1, box_size, box_size]

        coco_detections.append({
            'image_id': frame_idx,
            'category_id': 3,  # ball category
            'bbox': bbox,
            'score': confidence,
            'area': box_size * box_size,
            'iscrowd': 0,
        })

    return coco_detections


def load_coco_gt_frames(coco_path):
    """Load COCO GT and extract frames with ball.

    COCO uses 1-indexed frames (1-480), but tracker uses 0-indexed (0-520).
    Adjust by subtracting 1 from COCO frame numbers.
    """
    import re

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

    # Collect GT frames with balls
    gt_frames = set()
    for ann in coco['annotations']:
        if ann['category_id'] == ball_cat_id:
            image_id = ann['image_id']
            for img in coco['images']:
                if img['id'] == image_id:
                    # Extract frame number from file_name
                    file_name = img['file_name']
                    match = re.search(r'frame_(\d+)', file_name)
                    if match:
                        # COCO frames are 1-indexed, convert to 0-indexed
                        coco_frame_num = int(match.group(1))
                        tracker_frame_num = coco_frame_num - 1
                        gt_frames.add(tracker_frame_num)
                    break

    return sorted(gt_frames)


def evaluate_footandball():
    """Evaluate FootAndBall detections."""
    detections_path = REPO_ROOT / 'detections_footandball' / 'clip1.json'
    annotations_path = DEFAULT_ANNOTATIONS

    print(f"Loading FootAndBall detections from {detections_path}")
    footandball_dets = load_footandball_detections(detections_path)
    print(f"Found {len(footandball_dets)} ball detections")

    print(f"\nLoading COCO annotations from {annotations_path}")
    gt_frames = load_coco_gt_frames(annotations_path)
    print(f"Found {len(gt_frames)} GT frames with ball annotations (0-indexed)")

    # Simple evaluation: compare frame by frame
    print("\n" + "="*60)
    print("FOOTANDBALL EVALUATION METRICS (clip1)")
    print("="*60)

    tp = 0
    missed = 0
    fp = 0
    no_gt = 0

    # Create detection dict indexed by frame
    det_by_frame = {d['frame_idx']: d for d in footandball_dets}

    # Evaluate GT frames
    print(f"\nGT frames with ball: {len(gt_frames)}")

    for gt_frame_idx in gt_frames:
        if gt_frame_idx in det_by_frame:
            tp += 1
        else:
            missed += 1

    # Count no-GT predictions
    for det_frame_idx in det_by_frame.keys():
        if det_frame_idx not in gt_frames:
            no_gt += 1

    # FP = false positives on GT frames + predictions where there's no GT
    fp = no_gt

    print(f"\n✓ TP (True Positives):     {tp}")
    print(f"✗ Missed:                 {missed}")
    print(f"✗ FP (all):               {fp}")
    print(f"   - No-GT predicted:     {no_gt}")

    if tp + missed > 0:
        recall = tp / (tp + missed)
        print(f"\nRecall:                   {recall:.4f}")

    if tp + fp > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"Precision:                {precision:.4f}")

    if tp + missed > 0 and tp + fp > 0:
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"F1:                       {f1:.4f}")

    print("\n" + "="*60)
    print("COMPARISON WITH V4 BASELINE")
    print("="*60)
    print("\nv4 baseline on clip1:")
    print("  TP:        350")
    print("  Missed:    50")
    print("  FP (all):  80")
    print("  Recall:    0.8750")
    print("  Precision: 0.8140")
    print("  F1:        0.8439")

    print("\nFootAndBall on clip1:")
    print(f"  TP:        {tp}")
    print(f"  Missed:    {missed}")
    print(f"  FP (all):  {fp}")
    if tp + missed > 0:
        recall = tp / (tp + missed)
        print(f"  Recall:    {recall:.4f}")
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"  Precision: {precision:.4f}")
        if tp + missed > 0:
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  F1:        {f1:.4f}")


if __name__ == '__main__':
    evaluate_footandball()
