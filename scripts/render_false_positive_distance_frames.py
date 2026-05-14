#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ball_detection_metrics import (  # noqa: E402
    FLAT_SEQUENCE_NAME,
    center_of_xyxy,
    iou_xyxy,
    load_coco_ball_ground_truth,
    xywh_to_xyxy,
)
from benchmark_dataset import resolve_sequences  # noqa: E402


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the false-positive final-stage frames for a benchmark run, "
            "drawing GT and predicted boxes, a center-to-center line, and the pixel distance."
        )
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--sequence", type=str, default=FLAT_SEQUENCE_NAME)
    parser.add_argument("--stage", type=str, default="final", choices=["raw", "final"])
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--output_video", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--sort_by", type=str, default="frame", choices=["frame", "distance_desc"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--make_video", type=str2bool, default=True)
    return parser.parse_args()


def infer_output_dir(run_dir: Path, stage: str, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return run_dir / f"{stage}_false_positive_distance_frames"


def infer_output_video(run_dir: Path, stage: str, output_video: Optional[Path]) -> Path:
    if output_video is not None:
        return output_video.expanduser().resolve()
    return run_dir / f"{stage}_false_positive_distance_frames.mp4"


def draw_box(image: np.ndarray, xyxy: np.ndarray, color: Tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if not label:
        return
    cv2.putText(
        image,
        label,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_text_block(image: np.ndarray, lines: List[str]) -> None:
    for idx, text in enumerate(lines):
        y = 28 + idx * 24
        cv2.putText(
            image,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )


def build_sequence_maps(sequence) -> Tuple[Dict[int, Path], Dict[int, str], Dict[int, Optional[int]]]:
    image_paths_by_frame = {idx: path for idx, path in enumerate(sequence.image_paths, start=1)}
    file_names_by_frame = dict(sequence.file_names_by_frame)
    image_ids_by_frame = dict(sequence.image_ids_by_frame)
    return image_paths_by_frame, file_names_by_frame, image_ids_by_frame


def load_distance_map(summary_payload: dict, stage: str) -> Dict[str, float]:
    evaluation = summary_payload.get("evaluation", {})
    stage_payload = evaluation.get(stage, {})
    aggregate = stage_payload.get("aggregate", {})
    distance_map = aggregate.get("false_positive_center_distance_by_frame_px")
    if isinstance(distance_map, dict):
        return {str(key): float(value) for key, value in distance_map.items()}

    aggregate_payload = summary_payload.get("aggregate", {})
    prefixed = aggregate_payload.get(f"{stage}_false_positive_center_distance_by_frame_px", {})
    if isinstance(prefixed, dict):
        return {str(key): float(value) for key, value in prefixed.items()}
    return {}


def open_video_writer(output_video: Path, first_frame_path: Path, fps: float, codec: str) -> cv2.VideoWriter:
    image = cv2.imread(str(first_frame_path))
    if image is None:
        raise RuntimeError(f"Could not read sample frame: {first_frame_path}")
    height, width = image.shape[:2]
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_video}")
    return writer


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    detections_path = run_dir / "detections.json"
    summary_path = run_dir / "experiment_summary.json"
    if not detections_path.exists():
        raise FileNotFoundError(f"Missing detections export: {detections_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing experiment summary: {summary_path}")

    payload = json.loads(detections_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    data_root = Path(payload["data_root"]).expanduser().resolve()
    annotations_path_raw = payload.get("annotations_path")
    annotations_path = Path(annotations_path_raw).expanduser().resolve() if annotations_path_raw else None
    if annotations_path is None or not annotations_path.exists():
        raise FileNotFoundError("This renderer requires a valid annotations_path in detections.json")

    config = payload.get("config", {})
    sequences = resolve_sequences(
        data_root=data_root,
        seq_start=int(config.get("seq_start", 0)),
        seq_end=int(config.get("seq_end", 999)),
        seq_list=str(config.get("seq_list", "")),
        max_frames_per_seq=int(config.get("max_frames_per_seq", 0)),
        annotations_path=annotations_path,
    )
    sequence_map = {sequence.name: sequence for sequence in sequences}
    if args.sequence not in sequence_map:
        raise FileNotFoundError(
            f"Sequence {args.sequence!r} not found in run. Available: {', '.join(sorted(sequence_map))}"
        )
    sequence = sequence_map[args.sequence]
    image_paths_by_frame, file_names_by_frame, image_ids_by_frame = build_sequence_maps(sequence)

    ball_category_id = int(
        summary_payload.get("evaluation", {}).get("ball_category_id", config.get("ball_category_id", 1))
    )
    ground_truth = load_coco_ball_ground_truth(annotations_path, ball_category_id=ball_category_id)
    distance_map = load_distance_map(summary_payload, args.stage)
    if not distance_map:
        raise ValueError(f"No false-positive distance map found for stage={args.stage!r} in {summary_path}")

    detections_by_frame: Dict[int, dict] = {}
    for detection in payload.get("detections", []):
        if detection.get("sequence") != args.sequence:
            continue
        if str(detection.get("stage", "")) != args.stage:
            continue
        frame_index = int(detection["frame_index"])
        detections_by_frame[frame_index] = detection

    selected: List[Tuple[int, float]] = []
    for key, distance_px in distance_map.items():
        if ":" in key:
            sequence_name, frame_text = key.split(":", 1)
            if sequence_name != args.sequence:
                continue
            frame_index = int(frame_text)
        else:
            frame_index = int(key)
        selected.append((frame_index, float(distance_px)))

    if args.sort_by == "distance_desc":
        selected.sort(key=lambda item: (-item[1], item[0]))
    else:
        selected.sort(key=lambda item: item[0])

    if args.limit > 0:
        selected = selected[: args.limit]

    if not selected:
        raise ValueError("No selected false-positive frames to render")

    output_dir = infer_output_dir(run_dir, args.stage, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.make_video:
        writer = open_video_writer(
            infer_output_video(run_dir, args.stage, args.output_video),
            image_paths_by_frame[selected[0][0]],
            fps=args.fps,
            codec=args.codec,
        )

    try:
        for render_index, (frame_index, saved_distance_px) in enumerate(selected, start=1):
            image_path = image_paths_by_frame.get(frame_index)
            if image_path is None:
                raise KeyError(f"Frame {frame_index} missing from resolved sequence")
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read frame: {image_path}")

            image_id = image_ids_by_frame.get(frame_index)
            if image_id is None or int(image_id) not in ground_truth:
                raise KeyError(f"GT image_id missing for frame {frame_index}")
            gt_image = ground_truth[int(image_id)]
            if not gt_image.boxes_xyxy:
                raise ValueError(f"No GT ball box found for frame {frame_index}")
            gt_box = np.asarray(gt_image.boxes_xyxy[0], dtype=np.float32)

            detection = detections_by_frame.get(frame_index)
            if detection is None:
                raise KeyError(f"No {args.stage!r} detection found for frame {frame_index}")
            pred_box = np.asarray(xywh_to_xyxy(detection["bbox_xywh"]), dtype=np.float32)

            gt_center = center_of_xyxy(gt_box)
            pred_center = center_of_xyxy(pred_box)
            iou = iou_xyxy(pred_box, gt_box)

            draw_box(image, gt_box, (0, 200, 0), "GT")
            draw_box(
                image,
                pred_box,
                (0, 80, 255),
                f"PRED {detection.get('source', '')} {float(detection.get('score', 0.0)):.2f}",
            )

            gt_center_pt = (int(round(gt_center[0])), int(round(gt_center[1])))
            pred_center_pt = (int(round(pred_center[0])), int(round(pred_center[1])))
            cv2.circle(image, gt_center_pt, 5, (0, 200, 0), -1)
            cv2.circle(image, pred_center_pt, 5, (0, 80, 255), -1)
            cv2.line(image, gt_center_pt, pred_center_pt, (255, 255, 0), 2, cv2.LINE_AA)

            mid_x = int(round((gt_center_pt[0] + pred_center_pt[0]) / 2.0))
            mid_y = int(round((gt_center_pt[1] + pred_center_pt[1]) / 2.0))
            distance_label = f"{saved_distance_px:.1f}px"
            cv2.putText(
                image,
                distance_label,
                (mid_x + 8, max(24, mid_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            lines = [
                f"Frame: {frame_index:04d}",
                f"File: {file_names_by_frame[frame_index]}",
                f"Distance: {saved_distance_px:.2f} px",
                f"IoU: {iou:.4f}",
                f"Stage: {args.stage}",
                f"Output source: {detection.get('source', '')}",
                f"Rendered: {render_index}/{len(selected)}",
            ]
            draw_text_block(image, lines)

            output_path = output_dir / f"frame_{frame_index:04d}__dist_{saved_distance_px:.1f}px.png"
            cv2.imwrite(str(output_path), image)
            if writer is not None:
                writer.write(image)
    finally:
        if writer is not None:
            writer.release()

    print(f"[INFO] Rendered {len(selected)} frames to {output_dir}")
    if args.make_video:
        output_video = infer_output_video(run_dir, args.stage, args.output_video)
        print(f"[INFO] Output video: {output_video}")


if __name__ == "__main__":
    main()
