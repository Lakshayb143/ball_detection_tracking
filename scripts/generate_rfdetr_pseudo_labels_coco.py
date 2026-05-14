#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _eval_rfdetr_ball_only_common import RFDetrBallOnlyDetector  # noqa: E402
from ball_detection_metrics import BALL_CATEGORY_ID, xyxy_to_xywh  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL_PATH = REPO_ROOT / "checkpoints" / "ball_1120.pth"
DEFAULT_RESOLUTION = 1120
DEFAULT_CONFIDENCE_THRESHOLD = 0.01


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {value!r}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def mean_or_zero(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def candidate_image_paths(data_root: Path, ignored_roots: Sequence[Path]) -> List[Path]:
    resolved_ignored = [path.resolve() for path in ignored_roots]

    def is_ignored(path: Path) -> bool:
        resolved = path.resolve()
        for ignored_root in resolved_ignored:
            try:
                resolved.relative_to(ignored_root)
                return True
            except ValueError:
                continue
        return False

    return sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.name != "_annotations.coco.json"
        and not is_ignored(path)
    )


def relative_file_name(data_root: Path, image_path: Path) -> str:
    return image_path.relative_to(data_root).as_posix()


def read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def clip_xyxy(xyxy: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(0.0, min(x2, width - 1))
    y2 = max(0.0, min(y2, height - 1))
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return [x1, y1, x2, y2]


def csv_write_dicts(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_prediction_preview(
    image_path: Path,
    detections: Sequence[dict],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
        score = float(detection["score"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{score:.3f}",
            (max(0, x1), max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), image)


def coco_info(args: argparse.Namespace) -> dict:
    return {
        "description": "RF-DETR pseudo labels for manual correction",
        "version": "pseudo_v1",
        "model_path": str(args.model_path),
        "resolution": int(args.resolution) if int(args.resolution) > 0 else None,
        "confidence_threshold": float(args.confidence_threshold),
        "max_detections_per_image": int(args.max_detections_per_image),
    }


def coco_categories(ball_category_id: int) -> list[dict]:
    return [
        {
            "id": int(ball_category_id),
            "name": "ball",
            "supercategory": "sports",
        }
    ]


def build_coco_payload(
    *,
    info: dict,
    ball_category_id: int,
    images: Sequence[dict],
    annotations: Sequence[dict],
) -> dict:
    return {
        "info": info,
        "licenses": [],
        "categories": coco_categories(ball_category_id),
        "images": list(images),
        "annotations": list(annotations),
    }


def write_per_clip_annotations(
    *,
    data_root: Path,
    image_rows: Sequence[dict],
    annotation_rows: Sequence[dict],
    args: argparse.Namespace,
    file_name: str,
) -> list[str]:
    image_row_by_id = {int(row["id"]): row for row in image_rows}
    annotations_by_image_id: Dict[int, List[dict]] = defaultdict(list)
    for annotation in annotation_rows:
        annotations_by_image_id[int(annotation["image_id"])].append(annotation)

    clip_to_images: Dict[str, List[dict]] = defaultdict(list)
    for image_row in image_rows:
        file_path = Path(str(image_row["file_name"]))
        clip_key = file_path.parent.as_posix() if len(file_path.parts) > 1 else "."
        clip_to_images[clip_key].append(image_row)

    written_paths: list[str] = []
    for clip_key, clip_images in sorted(clip_to_images.items()):
        clip_dir = data_root if clip_key == "." else data_root / clip_key
        clip_annotations_path = clip_dir / file_name

        if clip_annotations_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Clip annotations already exist: {clip_annotations_path}. Use --overwrite true to replace them."
            )

        clip_images_sorted = sorted(clip_images, key=lambda row: int(row["id"]))
        clip_image_id_map: Dict[int, int] = {}
        clip_payload_images: List[dict] = []
        clip_payload_annotations: List[dict] = []
        next_image_id = 1
        next_annotation_id = 1

        for global_image_row in clip_images_sorted:
            global_image_id = int(global_image_row["id"])
            clip_image_id_map[global_image_id] = next_image_id
            file_path = Path(str(global_image_row["file_name"]))
            clip_relative_file_name = (
                file_path.relative_to(Path(clip_key)).as_posix() if clip_key != "." else file_path.as_posix()
            )
            clip_payload_images.append(
                {
                    "id": next_image_id,
                    "license": int(global_image_row.get("license", 0)),
                    "file_name": clip_relative_file_name,
                    "width": int(global_image_row["width"]),
                    "height": int(global_image_row["height"]),
                }
            )
            next_image_id += 1

        for global_image_row in clip_images_sorted:
            global_image_id = int(global_image_row["id"])
            for global_annotation in annotations_by_image_id.get(global_image_id, []):
                clip_payload_annotations.append(
                    {
                        "id": next_annotation_id,
                        "image_id": clip_image_id_map[global_image_id],
                        "category_id": int(global_annotation["category_id"]),
                        "bbox": list(global_annotation["bbox"]),
                        "area": float(global_annotation["area"]),
                        "iscrowd": int(global_annotation.get("iscrowd", 0)),
                    }
                )
                next_annotation_id += 1

        ensure_dir(clip_annotations_path.parent)
        clip_annotations_path.write_text(
            json.dumps(
                build_coco_payload(
                    info=coco_info(args),
                    ball_category_id=args.ball_category_id,
                    images=clip_payload_images,
                    annotations=clip_payload_annotations,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        written_paths.append(str(clip_annotations_path))

    return written_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate COCO pseudo labels for a folder of extracted frames using RF-DETR."
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_annotations", type=Path, default=None)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--confidence_threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--ball_class_id", type=int, default=0)
    parser.add_argument("--ball_category_id", type=int, default=BALL_CATEGORY_ID)
    parser.add_argument("--max_detections_per_image", type=int, default=1)
    parser.add_argument("--optimize_for_inference", type=str2bool, default=True)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--per_clip_json", type=str2bool, default=False)
    parser.add_argument("--per_clip_annotations_name", type=str, default="_annotations.coco.json")
    parser.add_argument("--vis_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.data_root = args.data_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    output_annotations = (
        args.output_annotations.expanduser().resolve()
        if args.output_annotations is not None
        else args.data_root / "_annotations.coco.json"
    )
    vis_dir = args.vis_dir.expanduser().resolve() if args.vis_dir is not None else None

    if output_annotations.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output annotations already exist: {output_annotations}. Use --overwrite true to replace them."
        )

    ignored_roots: List[Path] = []
    if vis_dir is not None:
        try:
            vis_dir.relative_to(args.data_root)
            ignored_roots.append(vis_dir)
        except ValueError:
            pass

    image_paths = candidate_image_paths(args.data_root, ignored_roots=ignored_roots)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under {args.data_root}")

    print(f"[INFO] Data root: {args.data_root}")
    print(f"[INFO] Images found: {len(image_paths)}")
    print(f"[INFO] Model path: {args.model_path}")
    print(f"[INFO] Resolution: {args.resolution if int(args.resolution) > 0 else 'default'}")
    print(f"[INFO] Confidence threshold: {args.confidence_threshold}")
    print(f"[INFO] Max detections per image: {args.max_detections_per_image}")
    print(f"[INFO] Output annotations: {output_annotations}")
    print(f"[INFO] Per-clip JSON: {args.per_clip_json}")
    if vis_dir is not None:
        print(f"[INFO] Preview dir: {vis_dir}")

    detector = RFDetrBallOnlyDetector(
        model_path=args.model_path,
        ball_class_id=args.ball_class_id,
        confidence_threshold=args.confidence_threshold,
        resolution=int(args.resolution) if int(args.resolution) > 0 else None,
        optimize_for_inference=args.optimize_for_inference,
    )

    images: List[dict] = []
    annotations: List[dict] = []
    manifest_rows: List[dict] = []
    source_summary: Dict[str, dict] = defaultdict(
        lambda: {
            "images": 0,
            "images_with_predictions": 0,
            "annotations": 0,
            "score_sum": 0.0,
        }
    )
    inference_times_ms: List[float] = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        file_name = relative_file_name(args.data_root, image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        clip_name = image_path.parent.relative_to(args.data_root).as_posix() if image_path.parent != args.data_root else "."

        started_at = time.perf_counter()
        raw_detections = detector.predict_ball_detections(image)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        inference_times_ms.append(elapsed_ms)

        detections = sorted(raw_detections, key=lambda item: float(item["score"]), reverse=True)
        if args.max_detections_per_image > 0:
            detections = detections[: args.max_detections_per_image]

        images.append(
            {
                "id": image_id,
                "license": 0,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        preview_rows: List[dict] = []
        for detection_index, detection in enumerate(detections, start=1):
            bbox_xyxy = clip_xyxy(detection["xyxy"], width=width, height=height)
            bbox_xywh = xyxy_to_xywh(bbox_xyxy)
            score = float(detection["score"])
            area = float(bbox_xywh[2] * bbox_xywh[3])
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(args.ball_category_id),
                    "bbox": bbox_xywh,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            preview_rows.append(
                {
                    "bbox_xyxy": bbox_xyxy,
                    "score": score,
                    "rank": detection_index,
                }
            )
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "clip": clip_name,
                    "width": width,
                    "height": height,
                    "prediction_rank": detection_index,
                    "score": score,
                    "bbox_x": bbox_xywh[0],
                    "bbox_y": bbox_xywh[1],
                    "bbox_w": bbox_xywh[2],
                    "bbox_h": bbox_xywh[3],
                    "area": area,
                    "inference_ms": elapsed_ms,
                }
            )
            annotation_id += 1

        if not detections:
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "clip": clip_name,
                    "width": width,
                    "height": height,
                    "prediction_rank": 0,
                    "score": "",
                    "bbox_x": "",
                    "bbox_y": "",
                    "bbox_w": "",
                    "bbox_h": "",
                    "area": "",
                    "inference_ms": elapsed_ms,
                }
            )

        if vis_dir is not None:
            draw_prediction_preview(
                image_path=image_path,
                detections=preview_rows,
                output_path=vis_dir / file_name,
            )

        source_summary[clip_name]["images"] += 1
        if detections:
            source_summary[clip_name]["images_with_predictions"] += 1
            source_summary[clip_name]["annotations"] += len(detections)
            source_summary[clip_name]["score_sum"] += sum(float(item["score"]) for item in detections)

    coco = build_coco_payload(
        info=coco_info(args),
        ball_category_id=args.ball_category_id,
        images=images,
        annotations=annotations,
    )
    ensure_dir(output_annotations.parent)
    output_annotations.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    per_clip_annotation_paths: list[str] = []
    if args.per_clip_json:
        per_clip_annotation_paths = write_per_clip_annotations(
            data_root=args.data_root,
            image_rows=images,
            annotation_rows=annotations,
            args=args,
            file_name=args.per_clip_annotations_name,
        )

    manifest_path = output_annotations.parent / "pseudo_label_manifest.csv"
    csv_write_dicts(manifest_path, manifest_rows)

    clip_rows = []
    for clip_name in sorted(source_summary.keys()):
        row = source_summary[clip_name]
        annotation_count = int(row["annotations"])
        clip_rows.append(
            {
                "clip": clip_name,
                "images": int(row["images"]),
                "images_with_predictions": int(row["images_with_predictions"]),
                "annotations": annotation_count,
                "mean_score": float(row["score_sum"] / annotation_count) if annotation_count > 0 else 0.0,
            }
        )
    clip_summary_path = output_annotations.parent / "pseudo_label_clip_summary.csv"
    if clip_rows:
        csv_write_dicts(clip_summary_path, clip_rows)

    summary = {
        "data_root": str(args.data_root),
        "output_annotations": str(output_annotations),
        "manifest_path": str(manifest_path),
        "clip_summary_path": str(clip_summary_path) if clip_rows else None,
        "preview_dir": str(vis_dir) if vis_dir is not None else None,
        "per_clip_json": bool(args.per_clip_json),
        "per_clip_annotations_name": args.per_clip_annotations_name,
        "per_clip_annotation_paths": per_clip_annotation_paths,
        "images_total": len(images),
        "annotations_total": len(annotations),
        "images_with_predictions": sum(1 for row in manifest_rows if row["prediction_rank"] == 1),
        "images_without_predictions": sum(1 for row in manifest_rows if row["prediction_rank"] == 0),
        "mean_inference_ms": mean_or_zero(inference_times_ms),
        "runtime_fps": (1000.0 / mean_or_zero(inference_times_ms)) if inference_times_ms else 0.0,
        "model_path": str(args.model_path),
        "resolution": int(args.resolution) if int(args.resolution) > 0 else None,
        "confidence_threshold": float(args.confidence_threshold),
        "max_detections_per_image": int(args.max_detections_per_image),
    }
    summary_path = output_annotations.parent / "pseudo_label_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] COCO annotations written to {output_annotations}")
    if per_clip_annotation_paths:
        print(f"[INFO] Wrote {len(per_clip_annotation_paths)} per-clip annotation files")
    print(f"[INFO] Manifest written to {manifest_path}")
    if clip_rows:
        print(f"[INFO] Clip summary written to {clip_summary_path}")
    if vis_dir is not None:
        print(f"[INFO] Preview images written to {vis_dir}")
    print(f"[INFO] Summary written to {summary_path}")
    print(
        f"[INFO] Images={summary['images_total']}, annotations={summary['annotations_total']}, "
        f"images_without_predictions={summary['images_without_predictions']}, fps={summary['runtime_fps']:.2f}"
    )


if __name__ == "__main__":
    main()
