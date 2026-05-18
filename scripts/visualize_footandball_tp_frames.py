#!/usr/bin/env python3
"""
Visualize FootAndBall TP frames with predictions and ground truth.
Saves all 24 detected frames to a folder for analysis.
"""
import json
import re
import cv2
from pathlib import Path
from typing import Dict, List, Tuple

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

    if ball_cat_id is None:
        raise ValueError("Ball category not found")

    # Build image ID to frame number mapping
    img_id_to_frame = {}
    for img in coco['images']:
        match = re.search(r'frame_(\d+)', img['file_name'])
        if match:
            coco_frame = int(match.group(1))
            tracker_frame = coco_frame - 1
            img_id_to_frame[img['id']] = tracker_frame

    # Collect GT boxes by frame
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
        x = det['x']
        y = det['y']
        confidence = det['confidence']

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

    # Sort by confidence
    for frame_idx in dets_by_frame:
        dets_by_frame[frame_idx].sort(key=lambda d: d['confidence'], reverse=True)

    return dets_by_frame


def draw_box(image, bbox_xyxy, color, label='', thickness=2):
    """Draw a box on image with label."""
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.6, 1)[0]
        cv2.rectangle(image, (x1, y1 - 25), (x1 + text_size[0] + 5, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 7), font, 0.6, (255, 255, 255), 1)


def visualize_tp_frames(output_dir: str = 'footandball_tp_frames', match_iou_threshold: float = 0.01):
    """Extract and visualize all TP frames."""
    detections_path = REPO_ROOT / 'detections_footandball' / 'clip1.json'
    annotations_path = DEFAULT_ANNOTATIONS
    video_path = REPO_ROOT / 'clips' / 'clip1.mp4'
    output_path = REPO_ROOT / output_dir

    print(f"Loading video from {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    print(f"Loading detections and GT...")
    dets_by_frame = load_footandball_detections(detections_path)
    gt_by_frame = load_coco_gt(annotations_path)

    # Find TP frames
    tp_frames = []
    for frame_idx in sorted(dets_by_frame.keys()):
        if frame_idx not in gt_by_frame:
            continue

        gt_boxes = gt_by_frame[frame_idx]
        dets = dets_by_frame[frame_idx]
        top_det = dets[0]

        # Check if matches
        max_iou = 0.0
        matched_gt = None
        for gt_box_dict in gt_boxes:
            iou = iou_xyxy(top_det['bbox'], gt_box_dict['bbox'])
            if iou > max_iou:
                max_iou = iou
                matched_gt = gt_box_dict

        if max_iou >= match_iou_threshold:
            tp_frames.append({
                'frame_idx': frame_idx,
                'iou': max_iou,
                'confidence': top_det['confidence'],
                'pred_bbox': top_det['bbox'],
                'gt_bbox': matched_gt['bbox']
            })

    print(f"Found {len(tp_frames)} TP frames with IoU >= {match_iou_threshold}")

    # Create output directory
    output_path.mkdir(exist_ok=True)
    print(f"\nSaving visualizations to {output_path}/")

    # Extract and save frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for tp in tp_frames:
            if tp['frame_idx'] == frame_count:
                # Draw boxes
                draw_box(frame, tp['gt_bbox'], (0, 255, 0), 'GT', thickness=2)  # Green for GT
                draw_box(frame, tp['pred_bbox'], (0, 0, 255), f"Pred (IoU={tp['iou']:.3f})", thickness=2)  # Red for pred

                # Save
                filename = f"frame_{tp['frame_idx']:04d}_iou_{tp['iou']:.3f}_conf_{tp['confidence']:.3f}.jpg"
                filepath = output_path / filename
                cv2.imwrite(str(filepath), frame)
                print(f"  Saved: {filename}")

        frame_count += 1

    cap.release()
    print(f"\nDone! Check {output_path} for visualizations.")
    print(f"\nColor coding:")
    print(f"  - GREEN box: Ground Truth")
    print(f"  - RED box: FootAndBall Prediction")


if __name__ == '__main__':
    visualize_tp_frames()
