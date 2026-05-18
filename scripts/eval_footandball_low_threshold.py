#!/usr/bin/env python3
"""
Evaluate FootAndBall detections at different confidence thresholds.
"""
import json
import re
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = REPO_ROOT / 'train' / '_annotations.coco.json'
BOX_SIZE = 20


def iou_xyxy(box_a: List[float], box_b: List[float]) -> float:
    """Calculate IoU between two xyxy boxes."""
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


def load_coco_gt(coco_path: Path) -> Dict[int, List[Dict]]:
    """Load COCO GT and extract ball annotations by frame."""
    with open(coco_path) as f:
        coco = json.load(f)

    ball_cat_id = None
    for cat in coco['categories']:
        if cat['name'] == 'ball':
            ball_cat_id = cat['id']
            break

    img_id_to_frame = {}
    for img in coco['images']:
        match = re.search(r'frame_(\d+)', img['file_name'])
        if match:
            coco_frame = int(match.group(1))
            tracker_frame = coco_frame - 1
            img_id_to_frame[img['id']] = tracker_frame

    gt_by_frame: Dict[int, List[Dict]] = {}
    for ann in coco['annotations']:
        if ann['category_id'] == ball_cat_id:
            image_id = ann['image_id']
            if image_id in img_id_to_frame:
                frame_idx = img_id_to_frame[image_id]
                bbox_xywh = ann['bbox']
                x, y, w, h = float(bbox_xywh[0]), float(bbox_xywh[1]), float(bbox_xywh[2]), float(bbox_xywh[3])
                bbox_xyxy = [x, y, x + w, y + h]

                if frame_idx not in gt_by_frame:
                    gt_by_frame[frame_idx] = []
                gt_by_frame[frame_idx].append({'bbox': bbox_xyxy})

    return gt_by_frame


def load_footandball_detections(json_path: Path) -> Dict[int, List[Dict]]:
    """Load FootAndBall detections."""
    with open(json_path) as f:
        detections = json.load(f)

    dets_by_frame: Dict[int, List[Dict]] = {}
    for det in detections:
        frame_idx = det['frame_idx']
        x, y = det['x'], det['y']
        confidence = det['confidence']

        x1 = x - BOX_SIZE / 2
        y1 = y - BOX_SIZE / 2
        x2 = x + BOX_SIZE / 2
        y2 = y + BOX_SIZE / 2

        if frame_idx not in dets_by_frame:
            dets_by_frame[frame_idx] = []
        dets_by_frame[frame_idx].append({
            'bbox': [x1, y1, x2, y2],
            'confidence': confidence,
        })

    for frame_idx in dets_by_frame:
        dets_by_frame[frame_idx].sort(key=lambda d: d['confidence'], reverse=True)

    return dets_by_frame


def evaluate_at_confidence_threshold(dets_by_frame, gt_by_frame, conf_threshold=0.01, match_iou=0.01):
    """Evaluate with a confidence threshold."""
    tp = 0
    missed = 0
    fp_on_gt = 0
    no_gt_pred = 0

    for frame_idx in sorted(gt_by_frame.keys()):
        gt_boxes = gt_by_frame[frame_idx]

        # Filter detections by confidence threshold
        frame_dets = dets_by_frame.get(frame_idx, [])
        frame_dets = [d for d in frame_dets if d['confidence'] >= conf_threshold]

        if frame_dets:
            top_det = frame_dets[0]
            max_iou = max(iou_xyxy(top_det['bbox'], gb['bbox']) for gb in gt_boxes)

            if max_iou >= match_iou:
                tp += 1
            else:
                fp_on_gt += 1
        else:
            missed += 1

    # No-GT predictions
    for frame_idx in dets_by_frame.keys():
        if frame_idx not in gt_by_frame:
            frame_dets = [d for d in dets_by_frame[frame_idx] if d['confidence'] >= conf_threshold]
            no_gt_pred += len(frame_dets)

    fp_all = fp_on_gt + no_gt_pred
    total_gt = len(gt_by_frame)

    recall = tp / total_gt if total_gt > 0 else 0
    precision = tp / (tp + fp_all) if (tp + fp_all) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp,
        'missed': missed,
        'fp_gt': fp_on_gt,
        'no_gt': no_gt_pred,
        'fp_all': fp_all,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'total_dets': sum(len([d for d in dets_by_frame.get(f, []) if d['confidence'] >= conf_threshold]) for f in dets_by_frame)
    }


def main():
    """Evaluate at multiple confidence thresholds."""
    detections_path = REPO_ROOT / 'detections_footandball' / 'clip1_threshold_001.json'
    annotations_path = DEFAULT_ANNOTATIONS

    print(f"Loading data...")
    dets_by_frame = load_footandball_detections(detections_path)
    gt_by_frame = load_coco_gt(annotations_path)

    print(f"GT frames: {len(gt_by_frame)}")
    print(f"Frames with detections: {len(dets_by_frame)}")

    print("\n" + "="*80)
    print("FOOTANDBALL EVALUATION AT DIFFERENT CONFIDENCE THRESHOLDS")
    print("="*80)

    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    print(f"\n{'Threshold':<12} {'TP':<8} {'Missed':<8} {'FP(all)':<10} {'Recall':<10} {'Precision':<12} {'F1':<10} {'Dets':<8}")
    print("-" * 98)

    results = {}
    for threshold in thresholds:
        result = evaluate_at_confidence_threshold(dets_by_frame, gt_by_frame, conf_threshold=threshold, match_iou=0.01)
        results[threshold] = result
        print(f"{threshold:<12.2f} {result['tp']:<8} {result['missed']:<8} {result['fp_all']:<10} "
              f"{result['recall']:<10.4f} {result['precision']:<12.4f} {result['f1']:<10.4f} {result['total_dets']:<8}")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    r_001 = results[0.01]['recall']
    r_05 = results[0.5]['recall']
    print(f"\nRecall improvement from 0.5 to 0.01: {r_001:.4f} vs {r_05:.4f} ({r_001/r_05:.1f}x)")

    print(f"\nDetections:")
    print(f"  Threshold 0.01: {results[0.01]['total_dets']} predictions")
    print(f"  Threshold 0.50: {results[0.5]['total_dets']} predictions")

    if r_001 > r_05 * 1.5:
        print(f"\n✅ SIGNIFICANT: Lowering threshold improves recall {r_001/r_05:.1f}x")
        print("   Model IS detecting more but being too conservative.")
        print("   Fine-tuning with confidence calibration could help significantly.")
    elif r_001 > r_05 * 1.1:
        print(f"\n⚠️  MODERATE: Recall improves {r_001/r_05:.1f}x at lower threshold")
        print("   Some signal, but improvement is modest.")
    else:
        print(f"\n❌ MINIMAL: Threshold has little effect")
        print("   Model is not detecting well at any confidence level.")
        print("   Fine-tuning unlikely to help much.")

    print(f"\nBest F1: {max(r['f1'] for r in results.values()):.4f} at threshold {max(results.items(), key=lambda x: x[1]['f1'])[0]:.2f}")


if __name__ == '__main__':
    main()
