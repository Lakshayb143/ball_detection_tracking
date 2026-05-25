#!/usr/bin/env python3
"""
Precompute player detections for all clips using checkpoints/player.pth.

Outputs: player_detections_v5/<clip_name>.json
Format: {frame_idx: [{"x1": float, "y1": float, "x2": float, "y2": float, "class_id": int, "confidence": float}, ...], ...}
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRMedium

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "checkpoints" / "player.pth"
CLIPS_DIR = ROOT / "clips"
OUTPUT_DIR = ROOT / "player_detections_v5"
CONFIDENCE_THRESHOLD = 0.3
ENABLE_OPTIMIZE = True


def precompute_player_detections(clip_path: Path, output_path: Path) -> None:
    """Detect all player/GK/referee in video and save as JSON."""
    print(f"\n[{clip_path.stem}] loading video...")
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise IOError(f"Could not open {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {frame_count} frames @ {fps:.1f}fps")

    print(f"[{clip_path.stem}] loading model from {MODEL_PATH}...")
    model = RFDETRMedium(pretrain_weights=str(MODEL_PATH))
    if ENABLE_OPTIMIZE:
        try:
            model.optimize_for_inference()
            print(f"  optimized for inference")
        except Exception as e:
            print(f"  [WARN] optimization failed: {e}")

    detections_by_frame = {}
    frame_idx = 0

    print(f"[{clip_path.stem}] processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{frame_count}")

        detections = model.predict(frame, confidence=CONFIDENCE_THRESHOLD)
        frame_dets = []

        if len(detections) > 0:
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                class_id = int(detections.class_id[i])
                confidence = float(detections.confidence[i])
                frame_dets.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "class_id": class_id,
                    "confidence": confidence,
                })

        detections_by_frame[frame_idx] = frame_dets
        frame_idx += 1

    cap.release()

    print(f"[{clip_path.stem}] saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(detections_by_frame, f)
    print(f"  done: {len(detections_by_frame)} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="+", type=int, metavar="N", help="clip numbers to process (default: all)")
    parser.add_argument("--force", action="store_true", help="overwrite existing detections")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.clips:
        clip_names = [f"clip{n}" for n in args.clips]
    else:
        clip_names = sorted(
            p.stem for p in CLIPS_DIR.glob("*.mp4")
            if p.stem not in ["testing_clip_1080"]
        )
        clip_names.append("testing_clip_1080")

    for clip_name in clip_names:
        clip_path = CLIPS_DIR / f"{clip_name}.mp4"
        output_path = OUTPUT_DIR / f"{clip_name}.json"

        if not clip_path.exists():
            print(f"[skip] {clip_name} - video not found")
            continue

        if output_path.exists() and not args.force:
            print(f"[skip] {clip_name} - output exists (use --force to overwrite)")
            continue

        precompute_player_detections(clip_path, output_path)


if __name__ == "__main__":
    main()
