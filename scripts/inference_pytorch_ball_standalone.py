#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import supervision as sv
from rfdetr import RFDETRMedium


BALL_CLASS_ID = 0


def process_frame(
    frame_bgr,
    model,
    confidence,
    frame_count,
    pred_file,
    box_annotator,
    label_annotator,
    class_map,
):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = model.predict(frame_rgb, confidence=confidence)

    ball_mask = detections.class_id == BALL_CLASS_ID
    ball_detections = detections[ball_mask]

    for box in ball_detections.xyxy:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        pred_file.write(f"{frame_count},-1,{x1},{y1},{width},{height},1,-1,-1,-1\n")

    labels = [
        f"{class_map[class_id]} {score:0.2f}"
        for score, class_id in zip(ball_detections.confidence, ball_detections.class_id)
    ]

    annotated = box_annotator.annotate(scene=frame_bgr.copy(), detections=ball_detections)
    annotated = label_annotator.annotate(scene=annotated, detections=ball_detections, labels=labels)
    return annotated


def main(args):
    print(f"Loading PyTorch RF-DETR model from checkpoint: {args.model_path}")
    model_kwargs = {"pretrain_weights": args.model_path}
    if int(args.input_size) > 0:
        model_kwargs["resolution"] = int(args.input_size)

    model = RFDETRMedium(**model_kwargs)
    if args.optimize_for_inference:
        try:
            model.optimize_for_inference()
            print("Model optimized for inference.")
        except Exception as exc:
            print(f"Warning: optimize_for_inference failed: {exc}")

    class_map = {BALL_CLASS_ID: "ball"}

    cap = None
    image_files = []
    is_video_input = False

    if os.path.isdir(args.input_path):
        image_files = sorted(
            os.path.join(args.input_path, name)
            for name in os.listdir(args.input_path)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not image_files:
            raise IOError(f"No images found in directory: {args.input_path}")

        first_frame = cv2.imread(image_files[0])
        if first_frame is None:
            raise IOError(f"Could not read first image: {image_files[0]}")
        frame_height, frame_width = first_frame.shape[:2]
        total_frames = len(image_files)
    elif os.path.isfile(args.input_path):
        is_video_input = True
        cap = cv2.VideoCapture(args.input_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {args.input_path}")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        raise FileNotFoundError(f"Input path does not exist: {args.input_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = Path(args.prediction_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (frame_width, frame_height))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    print(f"\nStarting inference on {total_frames} frames...")
    start_time = time.time()
    processed_count = 0

    with open(prediction_path, "w", encoding="utf-8") as pred_file:
        if is_video_input:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_count += 1
                annotated = process_frame(
                    frame,
                    model,
                    args.confidence,
                    processed_count,
                    pred_file,
                    box_annotator,
                    label_annotator,
                    class_map,
                )
                out_writer.write(annotated)
        else:
            for processed_count, image_path in enumerate(image_files, start=1):
                frame = cv2.imread(image_path)
                if frame is None:
                    continue
                annotated = process_frame(
                    frame,
                    model,
                    args.confidence,
                    processed_count,
                    pred_file,
                    box_annotator,
                    label_annotator,
                    class_map,
                )
                out_writer.write(annotated)

    total_time = time.time() - start_time
    actual_fps = processed_count / total_time if total_time > 0 else 0.0

    if cap is not None:
        cap.release()
    out_writer.release()

    print("\n" + "=" * 40)
    print(" PYTORCH INFERENCE COMPLETE ")
    print("=" * 40)
    print(f"Total Frames Processed: {processed_count}")
    print(f"Total Time Taken:       {total_time:.2f} seconds")
    print(f"Average Speed:          {actual_fps:.2f} FPS")
    print(f"Predictions saved to:   {prediction_path}")
    print(f"Annotated video saved:  {output_path}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run standalone PyTorch RF-DETR ball inference on a video file or frame directory."
    )
    parser.add_argument("--model_path", default="ball_samy_1120.pth", type=str)
    parser.add_argument("--input_path", default="testing_clip.mp4", type=str)
    parser.add_argument("--prediction_path", default="prediction_pytorch.txt", type=str)
    parser.add_argument("--output_path", default="output_testing_clip_pytorch.mp4", type=str)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--input_size", type=int, default=1120)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--optimize_for_inference", action="store_true")
    main(parser.parse_args())
