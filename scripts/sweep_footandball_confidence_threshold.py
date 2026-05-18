#!/usr/bin/env python3
"""
Sweep FootAndBall confidence thresholds to understand raw detection capability.
Tests thresholds: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 to find recall vs precision tradeoff.
"""
import torch
import cv2
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'FootAndBall'))

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import BALL_LABEL

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_footandball_with_threshold(video_path, model_weights, ball_threshold, device='cuda'):
    """Run FootAndBall with a specific ball confidence threshold."""
    model = footandball.model_factory('fb1', 'detect', ball_threshold=ball_threshold, player_threshold=0.7)
    model = model.to(device)

    if device == 'cpu':
        state_dict = torch.load(model_weights, map_location=lambda storage, loc: storage)
    else:
        state_dict = torch.load(model_weights)

    model.load_state_dict(state_dict)
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detections = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor = augmentations.numpy2tensor(frame)

        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(dim=0).to(device)
            dets = model(img_tensor)[0]

        # Extract ball detections
        for box, label, score in zip(dets['boxes'], dets['labels'], dets['scores']):
            if label == BALL_LABEL:
                x1, y1, x2, y2 = box
                x = float((x1 + x2) / 2)
                y = float((y1 + y2) / 2)
                confidence = float(score)

                detections.append({
                    'frame_idx': frame_idx,
                    'x': x,
                    'y': y,
                    'confidence': confidence
                })

        frame_idx += 1

    cap.release()
    return detections


def sweep_thresholds():
    """Sweep different confidence thresholds."""
    video_path = REPO_ROOT / 'clips' / 'clip1.mp4'
    model_weights = REPO_ROOT / 'FootAndBall' / 'models' / 'model_20201019_1416_final.pth'

    print("Running FootAndBall with different confidence thresholds...")
    print("This will take ~3-4 minutes (multiple passes through video)\n")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    results = {}

    for threshold in thresholds:
        print(f"Testing threshold {threshold}...")
        dets = run_footandball_with_threshold(str(video_path), str(model_weights), threshold, device='cuda')
        results[threshold] = dets
        print(f"  Found {len(dets)} detections\n")

    # Summary
    print("\n" + "="*70)
    print("FOOTANDBALL CONFIDENCE THRESHOLD SWEEP RESULTS")
    print("="*70)
    print(f"\n{'Threshold':<12} {'Detections':<15} {'Avg Confidence':<20}")
    print("-" * 47)

    for threshold in thresholds:
        dets = results[threshold]
        if dets:
            avg_conf = sum(d['confidence'] for d in dets) / len(dets)
            print(f"{threshold:<12.1f} {len(dets):<15} {avg_conf:<20.3f}")
        else:
            print(f"{threshold:<12.1f} {0:<15} {'N/A':<20}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    min_dets = min(len(dets) for dets in results.values())
    max_dets = max(len(dets) for dets in results.values())
    threshold_at_min = [t for t, d in results.items() if len(d) == min_dets][0]
    threshold_at_max = [t for t, d in results.items() if len(d) == max_dets][0]

    print(f"\nThreshold 0.1 (most permissive):  {len(results[0.1])} detections")
    print(f"Threshold 0.5 (used in main run):  {len(results[0.5])} detections")
    print(f"Threshold 0.7 (most strict):       {len(results[0.7])} detections")

    improvement_10_to_05 = len(results[0.1]) / len(results[0.5]) if results[0.5] else 0
    print(f"\nDetections at 0.1 vs 0.5: {improvement_10_to_05:.1f}x more")

    if improvement_10_to_05 > 1.5:
        print("\n⚠️  SIGNIFICANT improvement at lower threshold!")
        print("The model IS detecting more balls, but filtering them out.")
        print("This suggests fine-tuning could improve detection by:")
        print("  1. Learning better confidence calibration")
        print("  2. Adapting to your domain (appearance of 'ball-like' regions)")
    else:
        print("\n✓ Minimal improvement at lower threshold")
        print("The model genuinely doesn't detect many balls.")
        print("Fine-tuning may help but architectural mismatch likely remains.")


if __name__ == '__main__':
    sweep_thresholds()
