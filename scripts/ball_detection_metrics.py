from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BALL_CATEGORY_ID = 3
FLAT_SEQUENCE_NAME = "FLAT-COCO"
FRAME_NUMBER_PATTERN = re.compile(r"frame_(\d+)", re.IGNORECASE)


@dataclass
class GroundTruthImage:
    image_id: int
    file_name: str
    sequence: str
    boxes_xyxy: List[List[float]]


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def infer_sequence_name(file_name: str) -> str:
    relative_path = Path(file_name)
    stem = relative_path.stem
    if "_" not in stem:
        parent_parts = [part for part in relative_path.parent.parts if part != "images"]
        return "__".join(parent_parts) if parent_parts else FLAT_SEQUENCE_NAME

    sequence_token = stem.rsplit("_", 1)[0].strip()
    if sequence_token.upper().startswith("SNMOT-"):
        return sequence_token.upper()
    if sequence_token.isdigit():
        return f"SNMOT-{int(sequence_token):03d}"
    parent_parts = [part for part in relative_path.parent.parts if part != "images"]
    return "__".join(parent_parts) if parent_parts else FLAT_SEQUENCE_NAME


def xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + max(0.0, w), y + max(0.0, h)]


def area_of_xyxy(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def perimeter_of_xyxy(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in box]
    return 2.0 * (max(0.0, x2 - x1) + max(0.0, y2 - y1))


def center_of_xyxy(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance_px(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    center_a = center_of_xyxy(box_a)
    center_b = center_of_xyxy(box_b)
    return float(((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5)


def frame_key_from_image(image: GroundTruthImage) -> str:
    match = FRAME_NUMBER_PATTERN.search(image.file_name)
    if match:
        frame_number = int(match.group(1))
        if image.sequence == FLAT_SEQUENCE_NAME:
            return str(frame_number)
        return f"{image.sequence}:{frame_number}"
    return image.file_name


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    x_a = max(float(box_a[0]), float(box_b[0]))
    y_a = max(float(box_a[1]), float(box_b[1]))
    x_b = min(float(box_a[2]), float(box_b[2]))
    y_b = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = area_of_xyxy(box_a) + area_of_xyxy(box_b) - inter
    return float(inter / union) if union > 0.0 else 0.0


def build_detection_record(
    sequence: str,
    frame_index: int,
    original_frame: int,
    file_name: str,
    image_id: Optional[int],
    bbox_xyxy: Sequence[float],
    score: float,
    stage: str,
    source: str,
) -> dict:
    return {
        "sequence": sequence,
        "frame_index": int(frame_index),
        "original_frame": int(original_frame),
        "file_name": file_name,
        "image_id": image_id,
        "bbox_xywh": xyxy_to_xywh(bbox_xyxy),
        "score": float(score),
        "stage": stage,
        "source": source,
    }


def write_detection_export(
    path: Path,
    run_name: str,
    data_root: Path,
    detections: Sequence[dict],
    config: Optional[dict] = None,
    annotations_path: Optional[Path] = None,
) -> None:
    payload = {
        "format": "ball-benchmark-detections/v1",
        "run_name": run_name,
        "data_root": str(data_root),
        "annotations_path": str(annotations_path) if annotations_path is not None else None,
        "config": config or {},
        "detections": list(detections),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_benchmark_metrics_csv(
    path: Path,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    latency_ms: float = 0.0,
    **extra_fields: object,
) -> None:
    row: Dict[str, object] = {}
    if precision is not None:
        row["precision"] = float(precision)
    if recall is not None:
        row["recall"] = float(recall)
    row["latency_ms"] = float(latency_ms)
    for key, value in extra_fields.items():
        if value is None:
            row[key] = ""
        elif isinstance(value, (dict, list)):
            row[key] = json.dumps(value, sort_keys=True)
        else:
            row[key] = value

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def load_detection_export(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_category_id_by_name(
    annotations_path: Path,
    category_name: str,
    fallback_category_id: Optional[int] = None,
) -> Optional[int]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    normalized_target = category_name.strip().lower()
    for category in payload.get("categories", []):
        name = str(category.get("name", "")).strip().lower()
        if name == normalized_target:
            return int(category["id"])
    return fallback_category_id


def load_coco_ball_ground_truth(
    annotations_path: Path,
    ball_category_id: int = BALL_CATEGORY_ID,
) -> Dict[int, GroundTruthImage]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = {
        int(image["id"]): GroundTruthImage(
            image_id=int(image["id"]),
            file_name=str(image["file_name"]),
            sequence=infer_sequence_name(str(image["file_name"])),
            boxes_xyxy=[],
        )
        for image in payload.get("images", [])
    }

    for annotation in payload.get("annotations", []):
        if int(annotation.get("category_id", -1)) != int(ball_category_id):
            continue
        image_id = int(annotation["image_id"])
        if image_id not in images:
            continue
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = [float(value) for value in bbox]
        if w <= 0.0 or h <= 0.0:
            continue
        images[image_id].boxes_xyxy.append([x, y, x + w, y + h])

    return images


def _metric_bucket() -> Dict[str, float]:
    return {
        "images_total": 0.0,
        "gt_count": 0.0,
        "detection_count": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "matched_iou_sum": 0.0,
        "gt_frames": 0.0,
        "no_gt_frames": 0.0,
        "predicted_frames": 0.0,
        "event_tp": 0.0,
        "missed_detection_count": 0.0,
        "false_positive_count": 0.0,
        "tn": 0.0,
        "no_gt_predicted": 0.0,
        "false_positive_center_distance_sum_px": 0.0,
        "false_positive_center_distance_count": 0.0,
        "false_positive_center_distance_by_frame_px": {},
        "false_positive_center_distance_threshold_sum_px": 0.0,
        "false_positive_center_distance_threshold_count": 0.0,
        "false_positive_center_distance_threshold_by_frame_px": {},
    }


def _finalize_bucket(sequence: str, bucket: Dict[str, float]) -> Dict[str, float]:
    row = {
        "sequence": sequence,
        "images_total": bucket["images_total"],
        "gt_count": bucket["gt_count"],
        "detection_count": bucket["detection_count"],
        "tp": bucket["tp"],
        "fp": bucket["fp"],
        "fn": bucket["fn"],
        "matched_iou_sum": bucket["matched_iou_sum"],
        "gt_frames": bucket["gt_frames"],
        "no_gt_frames": bucket["no_gt_frames"],
        "predicted_frames": bucket["predicted_frames"],
        "event_tp": bucket["event_tp"],
        "missed_detection_count": bucket["missed_detection_count"],
        "false_positive_count": bucket["false_positive_count"],
        "tn": bucket["tn"],
        "no_gt_predicted": bucket["no_gt_predicted"],
        "false_positive_center_distance_count": bucket["false_positive_center_distance_count"],
        "false_positive_center_distance_by_frame_px": bucket["false_positive_center_distance_by_frame_px"],
        "false_positive_center_distance_threshold_count": bucket[
            "false_positive_center_distance_threshold_count"
        ],
        "false_positive_center_distance_threshold_by_frame_px": bucket[
            "false_positive_center_distance_threshold_by_frame_px"
        ],
    }
    row["precision"] = safe_div(bucket["tp"], bucket["tp"] + bucket["fp"])
    row["recall"] = safe_div(bucket["tp"], bucket["gt_count"])
    row["mean_matched_iou"] = safe_div(bucket["matched_iou_sum"], bucket["tp"])
    row["avg_false_positive_center_distance_px"] = safe_div(
        bucket["false_positive_center_distance_sum_px"],
        bucket["false_positive_center_distance_count"],
    )
    row["avg_false_positive_center_distance_threshold_px"] = safe_div(
        bucket["false_positive_center_distance_threshold_sum_px"],
        bucket["false_positive_center_distance_threshold_count"],
    )
    return row


def evaluate_detections(
    detections: Sequence[dict],
    annotations_path: Path,
    stage: str,
    iou_threshold: float,
    score_threshold: float,
    ball_category_id: int = BALL_CATEGORY_ID,
    allowed_image_ids: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    ground_truth = load_coco_ball_ground_truth(annotations_path, ball_category_id=ball_category_id)
    allowed_image_id_set = None
    if allowed_image_ids is not None:
        allowed_image_id_set = {int(image_id) for image_id in allowed_image_ids}
        ground_truth = {
            image_id: image
            for image_id, image in ground_truth.items()
            if image_id in allowed_image_id_set
        }
    image_id_by_file_name = {
        image.file_name: image_id for image_id, image in ground_truth.items()
    }

    detections_by_image: Dict[int, List[dict]] = defaultdict(list)
    for detection in detections:
        if detection.get("stage") != stage:
            continue
        if float(detection.get("score", 0.0)) < score_threshold:
            continue

        image_id = detection.get("image_id")
        if image_id is None:
            image_id = image_id_by_file_name.get(str(detection.get("file_name", "")))
        if image_id is None or int(image_id) not in ground_truth:
            continue
        detections_by_image[int(image_id)].append(detection)

    aggregate = _metric_bucket()
    per_sequence_buckets: Dict[str, Dict[str, float]] = defaultdict(_metric_bucket)

    for image_id, image in ground_truth.items():
        gt_boxes = list(image.boxes_xyxy)
        if len(gt_boxes) > 1:
            raise ValueError(
                "The event-style ball metric expects at most one GT ball per frame. "
                f"Found {len(gt_boxes)} GT boxes for image_id={image_id} file={image.file_name!r}."
            )
        image_detections = sorted(
            detections_by_image.get(image_id, []),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )

        matched_gt_indexes = set()
        matched_iou_sum = 0.0
        tp = 0
        fp = 0
        for detection in image_detections:
            det_box = xywh_to_xyxy(detection["bbox_xywh"])
            best_iou = 0.0
            best_gt_index = None
            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in matched_gt_indexes:
                    continue
                iou = iou_xyxy(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if best_gt_index is not None and best_iou >= iou_threshold:
                matched_gt_indexes.add(best_gt_index)
                tp += 1
                matched_iou_sum += best_iou
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt_indexes)
        bucket = per_sequence_buckets[image.sequence]
        for target in (aggregate, bucket):
            target["images_total"] += 1.0
            target["gt_count"] += float(len(gt_boxes))
            target["detection_count"] += float(len(image_detections))
            target["tp"] += float(tp)
            target["fp"] += float(fp)
            target["fn"] += float(fn)
            target["matched_iou_sum"] += matched_iou_sum
            gt_exists = len(gt_boxes) > 0
            if gt_exists:
                target["gt_frames"] += 1.0
            else:
                target["no_gt_frames"] += 1.0
            if image_detections:
                target["predicted_frames"] += 1.0

        gt_box = gt_boxes[0] if gt_boxes else None
        chosen_detection = image_detections[0] if image_detections else None
        frame_key = frame_key_from_image(image)
        if gt_box is not None and chosen_detection is not None:
            chosen_box = xywh_to_xyxy(chosen_detection["bbox_xywh"])
            event_iou = iou_xyxy(chosen_box, gt_box)
            if event_iou >= iou_threshold:
                aggregate["event_tp"] += 1.0
                bucket["event_tp"] += 1.0
            else:
                distance_px = center_distance_px(chosen_box, gt_box)
                threshold_px = perimeter_of_xyxy(chosen_box)
                for target in (aggregate, bucket):
                    target["false_positive_count"] += 1.0
                    target["false_positive_center_distance_sum_px"] += distance_px
                    target["false_positive_center_distance_count"] += 1.0
                    target["false_positive_center_distance_by_frame_px"][frame_key] = distance_px
                    target["false_positive_center_distance_threshold_sum_px"] += threshold_px
                    target["false_positive_center_distance_threshold_count"] += 1.0
                    target["false_positive_center_distance_threshold_by_frame_px"][frame_key] = threshold_px
        elif gt_box is not None and chosen_detection is None:
            aggregate["missed_detection_count"] += 1.0
            bucket["missed_detection_count"] += 1.0
        elif gt_box is None and chosen_detection is not None:
            aggregate["no_gt_predicted"] += 1.0
            bucket["no_gt_predicted"] += 1.0
        else:
            aggregate["tn"] += 1.0
            bucket["tn"] += 1.0

    per_sequence = [
        _finalize_bucket(sequence, bucket)
        for sequence, bucket in sorted(per_sequence_buckets.items())
    ]
    aggregate_row = _finalize_bucket("ALL", aggregate)
    return {
        "stage": stage,
        "iou_threshold": float(iou_threshold),
        "score_threshold": float(score_threshold),
        "aggregate": aggregate_row,
        "per_sequence": per_sequence,
    }


def count_predictions_matched_by_category(
    detections: Sequence[dict],
    annotations_path: Path,
    alt_category_id: int,
    iou_threshold: float,
    score_threshold: float,
    stage: str,
    allowed_image_ids: Optional[Sequence[int]] = None,
) -> int:
    """Count frames where the top prediction (by score) matches any GT box of alt_category_id.

    Used to measure how many model predictions landed on ball_out boxes (IoU >= iou_threshold)
    even though the model was trained to detect only ball. Each frame is counted at most once.
    """
    alt_gt = load_coco_ball_ground_truth(annotations_path, ball_category_id=alt_category_id)
    if allowed_image_ids is not None:
        allowed_set = {int(i) for i in allowed_image_ids}
        alt_gt = {k: v for k, v in alt_gt.items() if k in allowed_set}
    alt_gt = {k: v for k, v in alt_gt.items() if v.boxes_xyxy}

    image_id_by_file_name = {img.file_name: img_id for img_id, img in alt_gt.items()}

    detections_by_image: Dict[int, List[dict]] = defaultdict(list)
    for detection in detections:
        if detection.get("stage") != stage:
            continue
        if float(detection.get("score", 0.0)) < score_threshold:
            continue
        image_id = detection.get("image_id")
        if image_id is None:
            image_id = image_id_by_file_name.get(str(detection.get("file_name", "")))
        if image_id is None or int(image_id) not in alt_gt:
            continue
        detections_by_image[int(image_id)].append(detection)

    count = 0
    for image_id, image in alt_gt.items():
        image_dets = sorted(
            detections_by_image.get(image_id, []),
            key=lambda d: float(d.get("score", 0.0)),
            reverse=True,
        )
        if not image_dets:
            continue
        top_box = xywh_to_xyxy(image_dets[0]["bbox_xywh"])
        for gt_box in image.boxes_xyxy:
            if iou_xyxy(top_box, gt_box) >= iou_threshold:
                count += 1
                break
    return count


def evaluate_detection_export(
    detections_path: Path,
    annotations_path: Path,
    stage: str,
    iou_threshold: float,
    score_threshold: float,
    ball_category_id: int = BALL_CATEGORY_ID,
    allowed_image_ids: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    payload = load_detection_export(detections_path)
    return evaluate_detections(
        detections=payload.get("detections", []),
        annotations_path=annotations_path,
        stage=stage,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        ball_category_id=ball_category_id,
        allowed_image_ids=allowed_image_ids,
    )


def write_metrics_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate exported ball detections against a COCO annotation file."
    )
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--stage", type=str, choices=["raw", "final"], default="final")
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--ball_category_id", type=int, default=BALL_CATEGORY_ID)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate_detection_export(
        detections_path=args.detections.expanduser().resolve(),
        annotations_path=args.annotations.expanduser().resolve(),
        stage=args.stage,
        iou_threshold=args.iou,
        score_threshold=args.score_threshold,
        ball_category_id=args.ball_category_id,
    )

    if args.output_json is not None:
        args.output_json.expanduser().resolve().write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
    if args.output_csv is not None:
        write_metrics_csv(
            args.output_csv.expanduser().resolve(),
            list(result["per_sequence"]) + [result["aggregate"]],
        )

    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
