#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
WASB_ROOT = REPO_ROOT / "WASB-SBDT"
WASB_SRC = WASB_ROOT / "src"

for path in (REPO_ROOT, SCRIPTS_ROOT, WASB_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ball_detection_metrics import (  # noqa: E402
    build_detection_record,
    evaluate_detections,
    safe_div,
    write_benchmark_metrics_csv,
    write_detection_export,
)
from benchmark_dataset import default_annotation_path, resolve_annotated_image_path  # noqa: E402
from infer_frames import build_cfg as build_wasb_cfg  # noqa: E402
from infer_frames import build_detector_for_inference, preprocess_window  # noqa: E402
from dataloaders import build_img_transforms  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "benchmark_sets" / "central_test_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "central_test_v1_wasb_sbdt"
DEFAULT_ANNOTATIONS_NAME = "_annotations.coco.json"
DEFAULT_TRACKING_TEST_IMAGE_ROOT = Path("/home/lakshay/SoccerNet_tracking/dataset/test/images")
MODEL_WEIGHT_NAMES = {
    "wasb": "wasb_soccer_best.pth.tar",
    "tracknetv2": "tracknetv2_soccer_best.pth.tar",
    "restracknetv2": "restracknetv2_soccer_best.pth.tar",
    "monotrack": "monotrack_soccer_best.pth.tar",
    "ballseg": "ballseg_soccer_best.pth.tar",
    "deepball": "deepball_soccer_best.pth.tar",
    "deepball_large": "deepball-large_soccer_best.pth.tar",
}
MODEL_CHOICES = tuple(MODEL_WEIGHT_NAMES.keys())
CONTEXT_CHOICES = (
    "repeat",
    "tracking_source_else_repeat",
)
DEFAULT_EVAL_IOU = 0.01
DEFAULT_EVAL_SCORE_THRESHOLD = 0.01
DEFAULT_BOX_SIZE = 20.0


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


def csv_write_dicts(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_or_zero(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def resolve_dataset_layout(
    data_root: Path,
    annotations_override: Optional[Path],
) -> tuple[Path, Path]:
    if annotations_override is not None:
        if not annotations_override.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotations_override}")
        return data_root, annotations_override

    annotation_path = default_annotation_path(data_root)
    if annotation_path is not None and annotation_path.exists():
        return data_root, annotation_path

    nested_images_dir = data_root / "images"
    nested_annotation_path = nested_images_dir / DEFAULT_ANNOTATIONS_NAME
    if nested_annotation_path.exists():
        return nested_images_dir, nested_annotation_path

    raise FileNotFoundError(f"Could not find {DEFAULT_ANNOTATIONS_NAME} under {data_root}")


def resolve_default_weights(model_name: str) -> Path:
    return WASB_ROOT / "pretrained_weights" / MODEL_WEIGHT_NAMES[model_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark WASB-SBDT detector outputs on the central_test_v1 benchmark."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="deepball_large")
    parser.add_argument("--tracker", type=str, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--random_init", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--box_size", type=float, default=DEFAULT_BOX_SIZE)
    parser.add_argument("--eval_iou", type=float, default=DEFAULT_EVAL_IOU)
    parser.add_argument("--eval_score_threshold", type=float, default=DEFAULT_EVAL_SCORE_THRESHOLD)
    parser.add_argument("--context_mode", choices=CONTEXT_CHOICES, default="tracking_source_else_repeat")
    parser.add_argument("--tracking_test_image_root", type=Path, default=DEFAULT_TRACKING_TEST_IMAGE_ROOT)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--write_visualizations", type=str2bool, default=False)
    return parser.parse_args()


def load_image_rows(
    image_root: Path,
    annotations_path: Path,
    source_filters: Sequence[str],
    max_images: int,
) -> List[dict]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    wanted_sources = {item.strip() for item in source_filters if item.strip()}

    rows: List[dict] = []
    for image in sorted(payload.get("images", []), key=lambda row: int(row["id"])):
        source = str(image.get("source", "")).strip()
        if wanted_sources and source not in wanted_sources:
            continue

        file_name = str(image["file_name"])
        image_path = resolve_annotated_image_path(image_root, file_name)
        if image_path is None:
            raise FileNotFoundError(f"Could not resolve image path for {file_name} under {image_root}")

        rows.append(
            {
                "image_id": int(image["id"]),
                "file_name": file_name,
                "image_path": Path(image_path),
                "width": int(image.get("width", 0) or 0),
                "height": int(image.get("height", 0) or 0),
                "source": source or "unknown",
                "source_split": str(image.get("source_split", "") or ""),
                "source_file_name": str(image.get("source_file_name", "") or ""),
            }
        )

    if max_images > 0:
        rows = rows[:max_images]
    return rows


def parse_tracking_source_file_name(source_file_name: str) -> Optional[tuple[str, int, str]]:
    stem = Path(source_file_name).stem
    if "_" not in stem:
        return None
    sequence_name, frame_token = stem.rsplit("_", 1)
    if not sequence_name.upper().startswith("SNMOT-") or not frame_token.isdigit():
        return None
    suffix = Path(source_file_name).suffix or ".jpg"
    return sequence_name.upper(), int(frame_token), suffix


def centered_window_frame_numbers(frame_number: int, frames_in: int) -> List[int]:
    if frames_in <= 1:
        return [frame_number]
    if frames_in % 2 == 1:
        half = frames_in // 2
        return [frame_number + offset for offset in range(-half, half + 1)]
    start = frame_number - frames_in + 1
    return list(range(start, frame_number + 1))


def build_context_paths(
    row: dict,
    frames_in: int,
    context_mode: str,
    tracking_test_image_root: Path,
) -> tuple[List[Path], str]:
    current_path = Path(row["image_path"])
    if frames_in <= 1:
        return [current_path], "single_frame"

    if context_mode == "tracking_source_else_repeat" and row["source"] == "tracking_test":
        parsed = parse_tracking_source_file_name(str(row["source_file_name"]))
        if parsed is not None and tracking_test_image_root.exists():
            sequence_name, frame_number, suffix = parsed
            frame_numbers = centered_window_frame_numbers(frame_number, frames_in)
            context_paths: List[Path] = []
            current_source_path = tracking_test_image_root / f"{sequence_name}_{frame_number:06d}{suffix}"
            for candidate_frame in frame_numbers:
                candidate_path = tracking_test_image_root / f"{sequence_name}_{candidate_frame:06d}{suffix}"
                if candidate_path.exists():
                    context_paths.append(candidate_path)
                elif current_source_path.exists():
                    context_paths.append(current_source_path)
                else:
                    context_paths.append(current_path)
            return context_paths, "tracking_source"

    return [current_path] * frames_in, "repeat"


def select_output_index(frames_in: int, frames_out: int) -> int:
    if frames_out == 1:
        return 0
    if frames_out == frames_in and frames_in % 2 == 1:
        return frames_in // 2
    if frames_out == frames_in:
        return frames_out - 1
    raise ValueError(f"Unsupported frames_in/frames_out combination: {frames_in}/{frames_out}")


def select_best_prediction(predictions: Sequence[dict]) -> Optional[dict]:
    if not predictions:
        return None
    return max(predictions, key=lambda item: float(item.get("score", 0.0)))


def center_to_square_xyxy(
    center_xy: np.ndarray,
    width: int,
    height: int,
    box_size: float,
) -> np.ndarray:
    half = float(box_size) / 2.0
    x = float(center_xy[0])
    y = float(center_xy[1])
    x1 = max(0.0, x - half)
    y1 = max(0.0, y - half)
    x2 = min(float(width), x + half)
    y2 = min(float(height), y + half)
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def render_visualization(path: Path, row: dict, prediction: Optional[dict], box_xyxy: Optional[np.ndarray]) -> None:
    import cv2

    image = cv2.imread(str(row["image_path"]))
    if image is None:
        return

    if prediction is not None:
        x, y = int(round(float(prediction["xy"][0]))), int(round(float(prediction["xy"][1])))
        cv2.circle(image, (x, y), 6, (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{float(prediction['score']):.3f}",
            (max(0, x - 30), max(18, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if box_xyxy is not None:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box_xyxy]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 220, 0), 2)

    cv2.imwrite(str(path), image)


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise ValueError("WASB-SBDT only supports --device cuda in the current adapter.")

    requested_data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.tracking_test_image_root = args.tracking_test_image_root.expanduser().resolve()
    if args.annotations is not None:
        args.annotations = args.annotations.expanduser().resolve()

    if args.weights is None and not args.random_init:
        args.weights = resolve_default_weights(args.model)
    if args.weights is not None:
        args.weights = args.weights.expanduser().resolve()

    if not args.run_name.strip():
        args.run_name = f"{args.model}__central_test_v1"

    image_root, annotations_path = resolve_dataset_layout(requested_data_root, args.annotations)
    run_dir = ensure_dir(args.output_root / args.run_name)
    if args.write_visualizations:
        vis_dir = ensure_dir(run_dir / "visualizations")
    else:
        vis_dir = None

    image_rows = load_image_rows(
        image_root=image_root,
        annotations_path=annotations_path,
        source_filters=args.source,
        max_images=args.max_images,
    )
    if not image_rows:
        raise RuntimeError("No benchmark images matched the provided filters.")

    cfg = build_wasb_cfg(args)
    detector = build_detector_for_inference(cfg, args)
    _, transform_test = build_img_transforms(cfg)

    frames_in = int(detector.frames_in)
    frames_out = int(detector.frames_out)
    current_output_index = select_output_index(frames_in, frames_out)

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Requested data root: {requested_data_root}")
    print(f"[INFO] Resolved image root: {image_root}")
    print(f"[INFO] Annotation path: {annotations_path}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Frames in/out: {frames_in}/{frames_out}")
    print(f"[INFO] Box size: {args.box_size}")
    print(f"[INFO] Context mode: {args.context_mode}")
    print(f"[INFO] Images to process: {len(image_rows)}")
    if args.weights is not None:
        print(f"[INFO] Weights: {args.weights}")
    if args.source:
        print(f"[INFO] Source filter: {', '.join(args.source)}")

    all_detections: List[dict] = []
    center_rows: List[dict] = []
    source_runtime_ms: Dict[str, List[float]] = defaultdict(list)
    source_detection_count: Dict[str, int] = defaultdict(int)
    source_context_modes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    processed_image_ids: List[int] = []
    total_inference_ms = 0.0

    batch_size = max(1, int(args.batch_size))

    try:
        inference_context = torch.inference_mode()
    except Exception:
        inference_context = contextlib.nullcontext()

    with inference_context:
        for batch_start in range(0, len(image_rows), batch_size):
            batch_rows = image_rows[batch_start : batch_start + batch_size]
            batch_imgs: List[torch.Tensor] = []
            batch_affine_mats: Dict[int, List[torch.Tensor]] = defaultdict(list)
            batch_context_labels: List[str] = []
            batch_started_at = time.perf_counter()

            for row in batch_rows:
                context_paths, context_label = build_context_paths(
                    row=row,
                    frames_in=frames_in,
                    context_mode=args.context_mode,
                    tracking_test_image_root=args.tracking_test_image_root,
                )
                batch_context_labels.append(context_label)
                imgs, affine_mats = preprocess_window(context_paths, cfg, transform_test)
                batch_imgs.append(imgs)
                for scale, mat in affine_mats.items():
                    batch_affine_mats[scale].append(mat)

            imgs_tensor = torch.stack(batch_imgs, dim=0)
            affine_mats_tensor = {
                scale: torch.stack(mats, dim=0)
                for scale, mats in batch_affine_mats.items()
            }
            batch_results, _ = detector.run_tensor(imgs_tensor, affine_mats_tensor)
            batch_elapsed_ms = (time.perf_counter() - batch_started_at) * 1000.0

            per_item_ms = batch_elapsed_ms / max(1, len(batch_rows))
            total_inference_ms += batch_elapsed_ms

            for local_index, row in enumerate(batch_rows):
                processed_image_ids.append(int(row["image_id"]))
                source = str(row["source"])
                source_runtime_ms[source].append(per_item_ms)
                source_context_modes[source][batch_context_labels[local_index]] += 1

                predictions = batch_results.get(local_index, {}).get(current_output_index, [])
                best_prediction = select_best_prediction(predictions)
                box_xyxy = None
                score = 0.0
                if best_prediction is not None:
                    box_xyxy = center_to_square_xyxy(
                        center_xy=np.asarray(best_prediction["xy"], dtype=np.float32),
                        width=int(row["width"]),
                        height=int(row["height"]),
                        box_size=float(args.box_size),
                    )
                    score = float(best_prediction["score"])
                    source_detection_count[source] += 1
                    all_detections.append(
                        build_detection_record(
                            sequence=source,
                            frame_index=int(row["image_id"]),
                            original_frame=int(row["image_id"]),
                            file_name=str(row["file_name"]),
                            image_id=int(row["image_id"]),
                            bbox_xyxy=box_xyxy,
                            score=score,
                            stage="final",
                            source=f"wasb_sbdt:{args.model}",
                        )
                    )

                center_rows.append(
                    {
                        "image_id": int(row["image_id"]),
                        "file_name": str(row["file_name"]),
                        "source": source,
                        "context_mode": batch_context_labels[local_index],
                        "num_candidates": len(predictions),
                        "visible": bool(best_prediction is not None),
                        "score": score if best_prediction is not None else None,
                        "x": float(best_prediction["xy"][0]) if best_prediction is not None else None,
                        "y": float(best_prediction["xy"][1]) if best_prediction is not None else None,
                    }
                )

                if vis_dir is not None:
                    render_visualization(
                        path=vis_dir / str(row["file_name"]),
                        row=row,
                        prediction=best_prediction,
                        box_xyxy=box_xyxy,
                    )

    detections_path = run_dir / "detections.json"
    write_detection_export(
        path=detections_path,
        run_name=args.run_name,
        data_root=image_root,
        detections=all_detections,
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        annotations_path=annotations_path,
    )
    csv_write_dicts(run_dir / "center_predictions.csv", center_rows)

    evaluation = evaluate_detections(
        detections=all_detections,
        annotations_path=annotations_path,
        stage="final",
        iou_threshold=args.eval_iou,
        score_threshold=args.eval_score_threshold,
        allowed_image_ids=processed_image_ids,
    )
    aggregate_metrics = dict(evaluation["aggregate"])
    aggregate_metrics.update(
        {
            "frames_total": len(image_rows),
            "detections": len(all_detections),
            "inference_ms_avg": safe_div(total_inference_ms, len(image_rows)),
            "runtime_fps": safe_div(1000.0 * len(image_rows), total_inference_ms),
        }
    )

    source_rows: List[dict] = []
    for source in sorted({row["source"] for row in image_rows}):
        source_image_ids = [int(row["image_id"]) for row in image_rows if row["source"] == source]
        source_eval = evaluate_detections(
            detections=all_detections,
            annotations_path=annotations_path,
            stage="final",
            iou_threshold=args.eval_iou,
            score_threshold=args.eval_score_threshold,
            allowed_image_ids=source_image_ids,
        )["aggregate"]
        context_counts = source_context_modes[source]
        source_rows.append(
            {
                "source": source,
                "images_total": int(source_eval["images_total"]),
                "gt_count": int(source_eval["gt_count"]),
                "detections": int(source_eval["detection_count"]),
                "tp": int(source_eval["tp"]),
                "fp": int(source_eval["fp"]),
                "fn": int(source_eval["fn"]),
                "precision": float(source_eval["precision"]),
                "recall": float(source_eval["recall"]),
                "mean_matched_iou": float(source_eval["mean_matched_iou"]),
                "inference_ms_avg": mean_or_zero(source_runtime_ms[source]),
                "runtime_fps": safe_div(1000.0, mean_or_zero(source_runtime_ms[source])),
                "context_single_frame": int(context_counts.get("single_frame", 0)),
                "context_repeat": int(context_counts.get("repeat", 0)),
                "context_tracking_source": int(context_counts.get("tracking_source", 0)),
            }
        )

    csv_write_dicts(run_dir / "source_summary.csv", source_rows)
    write_benchmark_metrics_csv(
        run_dir / "benchmark_metrics.csv",
        precision=aggregate_metrics.get("precision", 0.0),
        recall=aggregate_metrics.get("recall", 0.0),
        latency_ms=aggregate_metrics.get("inference_ms_avg", 0.0),
    )
    fresh_metrics_path = args.output_root / f"{args.run_name}_benchmark_metrics.csv"
    write_benchmark_metrics_csv(
        fresh_metrics_path,
        precision=aggregate_metrics.get("precision", 0.0),
        recall=aggregate_metrics.get("recall", 0.0),
        latency_ms=aggregate_metrics.get("inference_ms_avg", 0.0),
    )

    experiment_summary = {
        "run_name": args.run_name,
        "requested_data_root": str(requested_data_root),
        "resolved_data_root": str(image_root),
        "annotations_path": str(annotations_path),
        "weights": str(args.weights) if args.weights is not None else None,
        "model": args.model,
        "frames_in": frames_in,
        "frames_out": frames_out,
        "context_mode": args.context_mode,
        "box_size": float(args.box_size),
        "source_filter": list(args.source),
        "random_init": bool(args.random_init),
        "detections_path": str(detections_path),
        "evaluation": {
            "status": "ok",
            "final": evaluation,
            "iou_threshold": float(args.eval_iou),
            "score_threshold": float(args.eval_score_threshold),
        },
        "aggregate": aggregate_metrics,
    }
    with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_summary, handle, indent=2)

    print(f"[INFO] Detection export written to {detections_path}")
    print(f"[INFO] Center predictions written to {run_dir / 'center_predictions.csv'}")
    print(f"[INFO] Source summary written to {run_dir / 'source_summary.csv'}")
    print(f"[INFO] Benchmark metrics written to {run_dir / 'benchmark_metrics.csv'}")
    print(f"[INFO] Fresh metrics CSV written to {fresh_metrics_path}")
    print(f"[INFO] Experiment summary written to {run_dir / 'experiment_summary.json'}")
    print(
        "[INFO] Aggregate: "
        f"precision={aggregate_metrics.get('precision', 0.0):.4f}, "
        f"recall={aggregate_metrics.get('recall', 0.0):.4f}, "
        f"fps={aggregate_metrics.get('runtime_fps', 0.0):.2f}"
    )


if __name__ == "__main__":
    main()
