#!/usr/bin/env python3
"""
Analyze patterns in the 24 TP frames to identify why most frames are missed.
"""
import json
import re
from pathlib import Path
from typing import Dict, List
import numpy as np

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


def box_area(bbox_xyxy: List[float]) -> float:
    """Calculate box area."""
    x1, y1, x2, y2 = bbox_xyxy
    return (x2 - x1) * (y2 - y1)


def load_coco_gt(coco_path: Path) -> Dict[int, List[Dict]]:
    """Load COCO GT."""
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


def analyze_patterns(match_iou_threshold: float = 0.01):
    """Analyze patterns in TP vs missed frames."""
    detections_path = REPO_ROOT / 'detections_footandball' / 'clip1.json'
    annotations_path = DEFAULT_ANNOTATIONS

    print("Loading data...")
    dets_by_frame = load_footandball_detections(detections_path)
    gt_by_frame = load_coco_gt(annotations_path)

    # Separate TP and missed frames
    tp_frames = []
    missed_frames = []
    fp_frames = []

    for frame_idx in sorted(gt_by_frame.keys()):
        gt_boxes = gt_by_frame[frame_idx]
        gt_area = box_area(gt_boxes[0]['bbox']) if gt_boxes else 0

        if frame_idx in dets_by_frame:
            top_det = dets_by_frame[frame_idx][0]
            max_iou = max(iou_xyxy(top_det['bbox'], gt_box['bbox']) for gt_box in gt_boxes)

            if max_iou >= match_iou_threshold:
                tp_frames.append({'frame': frame_idx, 'gt_area': gt_area, 'confidence': top_det['confidence']})
            else:
                fp_frames.append({'frame': frame_idx, 'gt_area': gt_area, 'confidence': top_det['confidence']})
        else:
            missed_frames.append({'frame': frame_idx, 'gt_area': gt_area})

    # Analyze
    print("\n" + "="*70)
    print("FOOTANDBALL DETECTION PATTERN ANALYSIS")
    print("="*70)

    print(f"\nTP frames: {len(tp_frames)}")
    print(f"Missed frames: {len(missed_frames)}")
    print(f"FP frames (on GT): {len(fp_frames)}")

    # Ball size analysis
    if tp_frames:
        tp_areas = [f['gt_area'] for f in tp_frames]
        print(f"\nDetected frames (TP):")
        print(f"  Ball area - Min: {min(tp_areas):.0f}, Max: {max(tp_areas):.0f}, Mean: {np.mean(tp_areas):.0f}")
        print(f"  Ball size - Min: {np.sqrt(min(tp_areas)):.1f}x{np.sqrt(min(tp_areas)):.1f}, "
              f"Max: {np.sqrt(max(tp_areas)):.1f}x{np.sqrt(max(tp_areas)):.1f}")

    if missed_frames:
        missed_areas = [f['gt_area'] for f in missed_frames]
        print(f"\nMissed frames:")
        print(f"  Ball area - Min: {min(missed_areas):.0f}, Max: {max(missed_areas):.0f}, Mean: {np.mean(missed_areas):.0f}")
        print(f"  Ball size - Min: {np.sqrt(min(missed_areas)):.1f}x{np.sqrt(min(missed_areas)):.1f}, "
              f"Max: {np.sqrt(max(missed_areas)):.1f}x{np.sqrt(max(missed_areas)):.1f}")

    if tp_frames and missed_frames:
        tp_mean = np.mean([f['gt_area'] for f in tp_frames])
        missed_mean = np.mean([f['gt_area'] for f in missed_frames])
        print(f"\n** Detected balls are {tp_mean/missed_mean:.2f}x larger on average **")

    # Frame position analysis
    if tp_frames:
        tp_positions = [f['frame'] for f in tp_frames]
        print(f"\nDetected frames span:")
        print(f"  Frames {min(tp_positions)}-{max(tp_positions)}")
        print(f"  Clusters: {tp_positions}")

    # Confidence analysis
    if tp_frames:
        tp_conf = [f['confidence'] for f in tp_frames]
        print(f"\nDetection confidence:")
        print(f"  Min: {min(tp_conf):.3f}, Max: {max(tp_conf):.3f}, Mean: {np.mean(tp_conf):.3f}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    if tp_frames and missed_frames:
        tp_mean = np.mean([f['gt_area'] for f in tp_frames])
        missed_mean = np.mean([f['gt_area'] for f in missed_frames])
        ratio = tp_mean / missed_mean
        if ratio > 2.0:
            print(f"\n⚠ SCALE BIAS: Model detects large balls {ratio:.1f}x better than small balls.")
            print("  This suggests architecture or input resolution issues, not just appearance.")
            print("  Fine-tuning may help but check input resolution first.")
        elif ratio > 1.3:
            print(f"\n⚠ MILD SCALE BIAS: Model slightly favors larger balls ({ratio:.1f}x).")
            print("  Could be fine-tuned, but domain mismatch is primary issue.")
        else:
            print(f"\n✓ NO SCALE BIAS: Ball sizes are similar across TP and missed.")
            print("  Failure likely due to appearance/domain mismatch → fine-tuning could help.")


if __name__ == '__main__':
    analyze_patterns()
