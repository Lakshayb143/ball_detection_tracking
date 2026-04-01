import cv2
import torch
import argparse
import os
import supervision as sv
import numpy as np
from rfdetr import RFDETRMedium

# Define the ball class ID at a global scope
BALL_CLASS_ID = 0


def process_frame(
    frame,
    model,
    confidence,
    frame_count,
    pred_file,
    box_annotator,
    label_annotator,
    class_map,
):
    """
    Runs inference on a single frame, writes detections to file,
    and returns the annotated frame.
    """

    detections = model.predict(frame, confidence=confidence)

    # Filter for only ball detections
    ball_mask = detections.class_id == BALL_CLASS_ID
    ball_detections = detections[ball_mask]

    # --- Write detections to prediction file ---
    # Format: <frame>, -1, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
    # The original script used a hardcoded confidence of 1. We'll stick to that.
    for box in ball_detections.xyxy:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        line = f"{frame_count},-1,{x1},{y1},{width},{height},1,-1,-1,-1\n"
        pred_file.write(line)

    labels = [
        f"{class_map[class_id]} {confidence:0.2f}"
        for confidence, class_id in zip(
            ball_detections.confidence, ball_detections.class_id
        )
    ]

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), detections=ball_detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=ball_detections, labels=labels
    )

    return annotated_frame


def main(args):
    """
    Main function to run raw model inference on a video file or a directory of
    images and save detections.
    """

    print(f"Loading model from checkpoint: {args.model_path}")
    try:
        model = RFDETRMedium(pretrain_weights=args.model_path)
        # model.optimize_for_inference()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Simplified class map
    class_map = {BALL_CLASS_ID: "ball"}
    print(f"Using hardcoded class map: {class_map}")

    # --- Input Handling ---
    frame_width, frame_height = 0, 0
    cap = None  # cv2.VideoCapture object
    image_files = []  # List of image file paths
    is_video_input = False

    if os.path.isdir(args.input_path):
        print(f"Input is a directory: {args.input_path}")
        is_video_input = False
        try:
            image_files = sorted(
                [
                    os.path.join(args.input_path, f)
                    for f in os.listdir(args.input_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )
            if not image_files:
                raise IOError(f"No images found in directory: {args.input_path}")

            # Get frame dimensions from the first image
            first_frame = cv2.imread(image_files[0])
            if first_frame is None:
                raise IOError(f"Could not read the first image at {image_files[0]}")
            frame_height, frame_width, _ = first_frame.shape

        except Exception as e:
            print(e)
            return

    elif os.path.isfile(args.input_path):
        print(f"Input is a video file: {args.input_path}")
        is_video_input = True
        try:
            cap = cv2.VideoCapture(args.input_path)
            if not cap.isOpened():
                raise IOError(f"Error: Could not open video file: {args.input_path}")

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        except Exception as e:
            print(e)
            return
    else:
        print(f"Error: Input path is not a valid directory or file: {args.input_path}")
        return

    # --- Setup video writer (now that we have dimensions) ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        args.output_path, fourcc, args.fps, (frame_width, frame_height)
    )
    print(f"Output video will be saved to: {args.output_path} at {args.fps} FPS")

    # --- Setup Annotators ---
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    print("\nStarting inference...")

    # Open the prediction file for writing
    with open(args.prediction_path, "w") as pred_file:
        if is_video_input:
            # --- Process Video File ---
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                annotated_frame = process_frame(
                    frame,
                    model,
                    args.confidence,
                    frame_count,
                    pred_file,
                    box_annotator,
                    label_annotator,
                    class_map,
                )
                out_writer.write(annotated_frame)

        else:
            # --- Process Image Directory ---
            for frame_count, image_path in enumerate(image_files, 1):
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"Warning: Could not read image {image_path}, skipping.")
                    continue

                annotated_frame = process_frame(
                    frame,
                    model,
                    args.confidence,
                    frame_count,
                    pred_file,
                    box_annotator,
                    label_annotator,
                    class_map,
                )
                out_writer.write(annotated_frame)

    # --- Cleanup ---
    if cap:
        cap.release()
    out_writer.release()
    print(f"Processing complete. Predictions saved to: {args.prediction_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run raw RF-DETR inference on an image directory or video file."
    )

    parser.add_argument(
        "--model_path",
        default="ball.pth",
        type=str,
        help="Path to the model checkpoint file.",
    )

    parser.add_argument(
        "--input_path",
        default="testing_clip.mp4",
        type=str,
        help="Path to the input video file or directory containing image frames.",
    )

    parser.add_argument(
        "--prediction_path",
        default="prediction.txt",
        type=str,
        help="Path to save the prediction output file.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="output_detections.mp4",
        help="Path to save the output annotated video.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections.",
    )

    parser.add_argument(
        "--fps", type=int, default=25, help="Frame rate for the output video."
    )

    args = parser.parse_args()
    main(args)
