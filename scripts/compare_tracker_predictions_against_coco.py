#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ball_detection_metrics import build_detection_record, evaluate_detections  # noqa: E402


FRAME_PATTERN = re.compile(r"frame_(\d+)", re.IGNORECASE)
VISIBLE_TOKENS = {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tracker prediction CSVs against a COCO ground-truth file by frame number."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        help="Prediction spec in the form label=/absolute/path/to/predictions.csv. Repeat for multiple runs.",
    )
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--score_threshold", type=float, default=0.01)
    return parser.parse_args()


def frame_num_from_text(text: str) -> int | None:
    match = FRAME_PATTERN.search(str(text))
    if not match:
        return None
    return int(match.group(1))


def parse_prediction_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(
            f"Invalid --prediction {spec!r}. Expected label=/absolute/path/to/predictions.csv"
        )
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    csv_path = Path(raw_path.strip()).expanduser().resolve()
    if not label:
        raise ValueError(f"Invalid --prediction {spec!r}: label cannot be empty")
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")
    return label, csv_path


def load_ground_truth(annotations_path: Path) -> tuple[dict, Dict[int, dict], int]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    images_by_frame: Dict[int, dict] = {}
    for image in payload.get("images", []):
        frame_num = frame_num_from_text(image.get("file_name", ""))
        if frame_num is not None:
            images_by_frame[frame_num] = image

    ball_category_id = None
    for category in payload.get("categories", []):
        if str(category.get("name", "")).strip().lower() == "ball":
            ball_category_id = int(category["id"])
            break
    if ball_category_id is None:
        raise RuntimeError(f"Could not find a 'ball' category in {annotations_path}")
    return payload, images_by_frame, ball_category_id


def score_prediction_csv(
    *,
    label: str,
    csv_path: Path,
    annotations_path: Path,
    images_by_frame: Dict[int, dict],
    ball_category_id: int,
    iou_threshold: float,
    score_threshold: float,
) -> dict:
    detections: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_num = frame_num_from_text(row.get("frame_name", ""))
            if frame_num is None:
                continue
            image = images_by_frame.get(frame_num)
            if image is None:
                continue

            visible_text = str(row.get("visible", "")).strip().lower()
            if visible_text not in VISIBLE_TOKENS:
                continue

            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                build_detection_record(
                    sequence="clip1",
                    frame_index=frame_num,
                    original_frame=frame_num,
                    file_name=str(image["file_name"]),
                    image_id=int(image["id"]),
                    bbox_xyxy=[x1, y1, x2, y2],
                    score=float(row.get("score") or 1.0),
                    stage="final",
                    source=label,
                )
            )

    aggregate = evaluate_detections(
        detections=detections,
        annotations_path=annotations_path,
        stage="final",
        iou_threshold=float(iou_threshold),
        score_threshold=float(score_threshold),
        ball_category_id=ball_category_id,
    )["aggregate"]
    precision = float(aggregate["precision"])
    recall = float(aggregate["recall"])
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "label": label,
        "detections": len(detections),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(aggregate["mean_matched_iou"]),
        "tp": int(aggregate["tp"]),
        "fp": int(aggregate["fp"]),
        "fn": int(aggregate["fn"]),
        "source_file": str(csv_path),
    }


def write_rows_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    annotations_path = args.annotations.expanduser().resolve()
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    _, images_by_frame, ball_category_id = load_ground_truth(annotations_path)
    prediction_specs = [parse_prediction_spec(spec) for spec in args.prediction]
    if not prediction_specs:
        raise ValueError("At least one --prediction label=/path/to/predictions.csv is required")

    rows = [
        score_prediction_csv(
            label=label,
            csv_path=csv_path,
            annotations_path=annotations_path,
            images_by_frame=images_by_frame,
            ball_category_id=ball_category_id,
            iou_threshold=float(args.iou_threshold),
            score_threshold=float(args.score_threshold),
        )
        for label, csv_path in prediction_specs
    ]
    rows.sort(key=lambda item: item["f1"], reverse=True)

    if args.output_csv is not None:
        write_rows_csv(args.output_csv.expanduser().resolve(), rows)
    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                {
                    "annotations_path": str(annotations_path),
                    "iou_threshold": float(args.iou_threshold),
                    "score_threshold": float(args.score_threshold),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
