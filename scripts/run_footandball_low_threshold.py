#!/usr/bin/env python3
"""
Run FootAndBall with very low confidence threshold (0.01) to see raw detection capability.
"""
import torch
import cv2
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'FootAndBall'))

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import BALL_LABEL

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_footandball_detector(video_path, model_weights, output_json, ball_threshold=0.01, device='cuda'):
    """Run FootAndBall with specified ball confidence threshold."""
    print(f"Loading model from {model_weights}")
    model = footandball.model_factory('fb1', 'detect', ball_threshold=ball_threshold, player_threshold=0.7)
    model = model.to(device)

    print(f"Loading weights...")
    if device == 'cpu':
        state_dict = torch.load(model_weights, map_location=lambda storage, loc: storage)
    else:
        state_dict = torch.load(model_weights)

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detections = []
    frame_idx = 0

    print(f"Processing {n_frames} frames with ball_threshold={ball_threshold}...")
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
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{n_frames} frames processed, {len(detections)} detections so far")

    cap.release()

    print(f"\nTotal detections found: {len(detections)}")

    # Save to JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)

    print(f"Saved to {output_json}")

    # Statistics
    if detections:
        confidences = [d['confidence'] for d in detections]
        print(f"\nConfidence statistics:")
        print(f"  Min: {min(confidences):.4f}")
        print(f"  Max: {max(confidences):.4f}")
        print(f"  Mean: {sum(confidences)/len(confidences):.4f}")
        print(f"  Detections > 0.5: {sum(1 for c in confidences if c > 0.5)}")
        print(f"  Detections > 0.3: {sum(1 for c in confidences if c > 0.3)}")
        print(f"  Detections > 0.1: {sum(1 for c in confidences if c > 0.1)}")

    return detections


if __name__ == '__main__':
    project_root = REPO_ROOT
    video_path = project_root / 'clips' / 'clip1.mp4'
    model_weights = project_root / 'FootAndBall' / 'models' / 'model_20201019_1416_final.pth'
    output_json = project_root / 'detections_footandball' / 'clip1_threshold_001.json'

    assert video_path.exists(), f"Video not found: {video_path}"
    assert model_weights.exists(), f"Model weights not found: {model_weights}"

    run_footandball_detector(
        str(video_path),
        str(model_weights),
        str(output_json),
        ball_threshold=0.01,
        device='cuda'
    )
