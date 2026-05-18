#!/usr/bin/env python3
"""
Run FootAndBall model on clip1 and extract ball detections to JSON format.
"""
import torch
import cv2
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

# Add FootAndBall to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FootAndBall'))

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import PLAYER_LABEL, BALL_LABEL


def run_footandball_detector(video_path, model_weights, output_json, device='cuda', ball_threshold=0.5):
    """
    Run FootAndBall detector and extract ball detections to JSON.

    Args:
        video_path: Path to input video
        model_weights: Path to model weights
        output_json: Path to save detection JSON
        device: Device to use ('cuda' or 'cpu')
        ball_threshold: Confidence threshold for ball detections
    """
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
    sequence = cv2.VideoCapture(video_path)
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))

    detections = []
    frame_idx = 0

    print(f"Processing {n_frames} frames...")
    pbar = tqdm(total=n_frames)

    while sequence.isOpened():
        ret, frame = sequence.read()
        if not ret:
            break

        # Convert color space from BGR to RGB, convert to tensor and normalize
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
        pbar.update(1)

    pbar.close()
    sequence.release()

    print(f"Found {len(detections)} ball detections")

    # Save to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(detections, f, indent=2)

    print(f"Saved detections to {output_json}")
    return detections


if __name__ == '__main__':
    # Configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(project_root, 'clips', 'clip1.mp4')
    model_weights = os.path.join(project_root, 'FootAndBall', 'models', 'model_20201019_1416_final.pth')
    output_json = os.path.join(project_root, 'detections_footandball', 'clip1.json')

    assert os.path.exists(video_path), f"Video not found: {video_path}"
    assert os.path.exists(model_weights), f"Model weights not found: {model_weights}"

    run_footandball_detector(
        video_path,
        model_weights,
        output_json,
        device='cuda',
        ball_threshold=0.5
    )
