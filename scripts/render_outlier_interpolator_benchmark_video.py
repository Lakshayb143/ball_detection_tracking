#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

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
        description="Render a video overlay from an outlier/interpolator benchmark run."
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--sequence", type=str, default="FLAT-COCO")
    parser.add_argument("--output_video", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--show_raw", type=str2bool, default=True)
    parser.add_argument("--show_status", type=str2bool, default=True)
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def xywh_to_xyxy(box_xywh: list[float]) -> np.ndarray:
    x, y, w, h = [float(item) for item in box_xywh]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def draw_box(
    image: np.ndarray,
    xyxy: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    thickness: int,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def infer_output_video(run_dir: Path, sequence: str, output_video: Optional[Path]) -> Path:
    if output_video is not None:
        return output_video.expanduser().resolve()
    safe_sequence = sequence.replace("/", "__")
    return run_dir / f"{safe_sequence}_overlay.mp4"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    detections_path = run_dir / "detections.json"
    if not detections_path.exists():
        raise FileNotFoundError(f"Missing detections export: {detections_path}")

    payload = json.loads(detections_path.read_text(encoding="utf-8"))
    data_root = Path(payload["data_root"]).expanduser().resolve()
    annotations_path_raw = payload.get("annotations_path")
    annotations_path = Path(annotations_path_raw).expanduser().resolve() if annotations_path_raw else None
    config = payload.get("config", {})

    sequences = resolve_sequences(
        data_root=data_root,
        seq_start=int(config.get("seq_start", 0)),
        seq_end=int(config.get("seq_end", 999)),
        seq_list=str(config.get("seq_list", "")),
        max_frames_per_seq=int(config.get("max_frames_per_seq", 0)),
        annotations_path=annotations_path if annotations_path is not None and annotations_path.exists() else None,
    )
    sequence_map = {sequence.name: sequence for sequence in sequences}
    if args.sequence not in sequence_map:
        raise FileNotFoundError(
            f"Sequence {args.sequence!r} not found in run. Available: {', '.join(sorted(sequence_map))}"
        )
    sequence = sequence_map[args.sequence]

    frame_trace_path = run_dir / args.sequence / "frame_trace.csv"
    if not frame_trace_path.exists():
        raise FileNotFoundError(f"Missing frame trace CSV: {frame_trace_path}")
    with frame_trace_path.open("r", encoding="utf-8", newline="") as handle:
        frame_trace_rows = {int(row["frame"]): row for row in csv.DictReader(handle)}

    raw_by_frame: Dict[int, dict] = {}
    final_by_frame: Dict[int, dict] = {}
    for detection in payload.get("detections", []):
        if detection.get("sequence") != args.sequence:
            continue
        frame_index = int(detection["frame_index"])
        stage = str(detection.get("stage", ""))
        if stage == "raw" and frame_index not in raw_by_frame:
            raw_by_frame[frame_index] = detection
        if stage == "final" and frame_index not in final_by_frame:
            final_by_frame[frame_index] = detection

    sample_image = cv2.imread(str(sequence.image_paths[0]))
    if sample_image is None:
        raise RuntimeError(f"Could not read sample frame: {sequence.image_paths[0]}")
    height, width = sample_image.shape[:2]
    output_video = infer_output_video(run_dir, args.sequence, args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*args.codec),
        float(args.fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_video}")

    try:
        for frame_index, image_path in enumerate(sequence.image_paths, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read frame: {image_path}")

            trace = frame_trace_rows.get(frame_index, {})
            raw_detection = raw_by_frame.get(frame_index)
            final_detection = final_by_frame.get(frame_index)

            if args.show_raw and raw_detection is not None:
                raw_xyxy = xywh_to_xyxy(raw_detection["bbox_xywh"])
                raw_label = f"RAW {float(raw_detection['score']):.2f}"
                if final_detection is None:
                    raw_label += " REJECTED"
                draw_box(image, raw_xyxy, (0, 215, 255), raw_label, thickness=1)

            if final_detection is not None:
                final_xyxy = xywh_to_xyxy(final_detection["bbox_xywh"])
                source = str(final_detection.get("source", ""))
                score = float(final_detection.get("score", 0.0))
                if source == "interpolation":
                    color = (0, 0, 255)
                    label = f"INTERP {score:.2f}"
                    center_x = int(round((final_xyxy[0] + final_xyxy[2]) / 2.0))
                    center_y = int(round((final_xyxy[1] + final_xyxy[3]) / 2.0))
                    cv2.circle(image, (center_x, center_y), 10, color, 2)
                else:
                    color = (0, 200, 0)
                    label = f"RF-DETR {score:.2f}"
                draw_box(image, final_xyxy, color, label, thickness=2)

            if args.show_status:
                lines = [
                    f"Frame: {frame_index:04d}",
                    f"Output: {trace.get('output_source', 'none')}",
                    f"Tracker active: {trace.get('tracker_active', 'False')}",
                    f"Tracker gap: {trace.get('tracker_gap', '0')}",
                    f"Interpolation active: {trace.get('interpolation_active', 'False')}",
                    f"Interpolated output: {trace.get('interpolated_output', 'False')}",
                ]
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

            writer.write(image)
    finally:
        writer.release()

    print(f"[INFO] Output video: {output_video}")


if __name__ == "__main__":
    main()
