#!/usr/bin/env python3
"""Run ball_samy_1120 (single-class ball detector) on a video and write an annotated output video."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


COLOUR_BALL = (0, 255, 0)  # green
CONFIDENCE = 0.01
RESOLUTION = 1120


def draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    out = frame.copy()
    if detections is None or len(detections) == 0:
        return out
    for xyxy, conf in zip(detections.xyxy, detections.confidence):
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOUR_BALL, 2)
        label = f"ball {conf:.2f}"
        pos = (x1, max(y1 - 8, 16))
        cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOUR_BALL, 2)
    return out


def main(args: argparse.Namespace) -> None:
    from rfdetr import RFDETRMedium

    print(f"Loading model: {args.model_path}")
    model = RFDETRMedium(pretrain_weights=str(args.model_path), resolution=args.resolution)
    if args.optimize:
        try:
            model.optimize_for_inference()
            print("Model optimized for inference.")
        except Exception as exc:
            print(f"Warning: optimize_for_inference failed: {exc}")

    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(args.output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"Input : {args.input_video}  ({w}x{h} @ {fps:.1f} fps, {total} frames)")
    print(f"Output: {out_path}")

    frame_num = 0
    t0 = time.time()
    ball_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, confidence=args.confidence)

        annotated = draw_detections(frame, detections)

        if detections is not None and len(detections) > 0:
            ball_frames += 1

        writer.write(annotated)

        if frame_num % 500 == 0:
            elapsed = time.time() - t0
            print(f"  frame {frame_num}/{total}  ({frame_num/elapsed:.1f} fps)")

    cap.release()
    writer.release()

    elapsed = time.time() - t0
    print(f"\nDone. {frame_num} frames in {elapsed:.1f}s ({frame_num/elapsed:.1f} fps)")
    print(f"  Frames with ball: {ball_frames}")
    print(f"  Output saved to : {out_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Visualize ball_samy_1120 model on a video.")
    parser.add_argument("--model_path", type=Path, default=repo_root / "checkpoints" / "ball_samy_1120.pth")
    parser.add_argument("--input_video", type=Path, required=True)
    parser.add_argument("--output_video", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--optimize", action="store_true", default=True)
    main(parser.parse_args())
