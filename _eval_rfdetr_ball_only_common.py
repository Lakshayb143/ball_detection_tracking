#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ball_detection_metrics import (  # noqa: E402
    BALL_CATEGORY_ID,
    build_detection_record,
    evaluate_detections,
    infer_category_id_by_name,
    safe_div,
    write_benchmark_metrics_csv,
    write_detection_export,
)
from benchmark_dataset import BenchmarkSequence, default_annotation_path, resolve_sequences  # noqa: E402
from distance_rule_adjustment import (  # noqa: E402
    DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
    attach_distance_rule_summary,
    resolve_center_distance_bucket_threshold_px,
    write_single_stage_distance_rule_metrics_csv,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "test"
DEFAULT_MODEL_PATH = REPO_ROOT / "checkpoints" / "ball.pth"
DEFAULT_SEQ_START = 0
DEFAULT_SEQ_END = 999
DEFAULT_SEQ_LIST = ""
DEFAULT_MAX_FRAMES_PER_SEQ = 0
DEFAULT_BALL_CLASS_ID = 0
DEFAULT_CONFIDENCE_THRESHOLD = 0.01
DEFAULT_EVAL_IOU = 0.01
DEFAULT_EVAL_SCORE_THRESHOLD = 0.01
DEFAULT_OPTIMIZE_FOR_INFERENCE = True


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


def mean_or_zero(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def csv_write_dicts(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_dataset_layout(
    data_root: Path,
    annotations_override: Optional[Path],
) -> Tuple[Path, Optional[Path]]:
    if annotations_override is not None:
        if not annotations_override.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotations_override}")
        nested_images_dir = data_root / "images"
        if nested_images_dir.exists() and annotations_override.parent == nested_images_dir:
            return nested_images_dir, annotations_override
        return data_root, annotations_override

    flat_annotations = default_annotation_path(data_root)
    if flat_annotations is not None:
        return data_root, flat_annotations

    nested_images_dir = data_root / "images"
    nested_annotations = nested_images_dir / "_annotations.coco.json"
    if nested_annotations.exists():
        return nested_images_dir, nested_annotations

    return data_root, None


def resolve_ball_category_id(
    annotations_path: Optional[Path],
    explicit_ball_category_id: Optional[int],
) -> int:
    if explicit_ball_category_id is not None:
        return int(explicit_ball_category_id)
    if annotations_path is not None and annotations_path.exists():
        inferred = infer_category_id_by_name(
            annotations_path=annotations_path,
            category_name="ball",
            fallback_category_id=BALL_CATEGORY_ID,
        )
        if inferred is not None:
            return int(inferred)
    return int(BALL_CATEGORY_ID)


class RFDetrBallOnlyDetector:
    def __init__(
        self,
        model_path: Path,
        ball_class_id: int,
        confidence_threshold: float,
        resolution: Optional[int],
        optimize_for_inference: bool,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"RF-DETR checkpoint not found: {model_path}")

        self.ball_class_id = ball_class_id
        self.confidence_threshold = confidence_threshold

        from rfdetr import RFDETRMedium

        model_kwargs = {"pretrain_weights": str(model_path)}
        if resolution is not None and int(resolution) > 0:
            model_kwargs["resolution"] = int(resolution)
        self.model = RFDETRMedium(**model_kwargs)

        if optimize_for_inference:
            try:
                self.model.optimize_for_inference()
                print("[INFO] Optimized RF-DETR model for inference")
            except Exception as exc:
                print(f"[WARN] Could not optimize RF-DETR model for inference: {exc}")

    def predict_ball_detections(self, image_bgr: np.ndarray) -> List[dict]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(image_rgb, confidence=self.confidence_threshold)
        if detections is None or len(detections) == 0:
            return []

        xyxy = np.asarray(detections.xyxy)
        confidence = np.asarray(detections.confidence).reshape(-1)
        class_ids = np.asarray(detections.class_id).reshape(-1)

        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            return []

        num_boxes = xyxy.shape[0]
        if confidence.size == 1 and num_boxes > 1:
            confidence = np.repeat(confidence, num_boxes)
        if class_ids.size == 1 and num_boxes > 1:
            class_ids = np.repeat(class_ids, num_boxes)

        num_items = min(num_boxes, confidence.size, class_ids.size)
        if num_items <= 0:
            return []

        output: List[dict] = []
        for index in range(num_items):
            if int(class_ids[index]) != self.ball_class_id:
                continue
            score = float(confidence[index])
            if score < float(self.confidence_threshold):
                continue
            output.append(
                {
                    "xyxy": np.asarray(xyxy[index], dtype=np.float32),
                    "score": score,
                }
            )
        return output


def build_arg_parser(
    *,
    default_run_name: str,
    default_output_root: Path,
    default_model_path: Path,
    default_resolution: Optional[int],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the RF-DETR ball detector without any tracking logic."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=default_output_root)
    parser.add_argument("--run_name", type=str, default=default_run_name)
    parser.add_argument("--seq_start", type=int, default=DEFAULT_SEQ_START)
    parser.add_argument("--seq_end", type=int, default=DEFAULT_SEQ_END)
    parser.add_argument("--seq_list", type=str, default=DEFAULT_SEQ_LIST)
    parser.add_argument("--max_frames_per_seq", type=int, default=DEFAULT_MAX_FRAMES_PER_SEQ)

    parser.add_argument("--model_path", type=Path, default=default_model_path)
    parser.add_argument("--ball_class_id", type=int, default=DEFAULT_BALL_CLASS_ID)
    parser.add_argument("--ball_category_id", type=int, default=None)
    parser.add_argument("--confidence_threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--eval_iou", type=float, default=DEFAULT_EVAL_IOU)
    parser.add_argument("--eval_score_threshold", type=float, default=DEFAULT_EVAL_SCORE_THRESHOLD)
    parser.add_argument("--resolution", type=int, default=default_resolution or 0)
    parser.add_argument("--optimize_for_inference", type=str2bool, default=DEFAULT_OPTIMIZE_FOR_INFERENCE)
    return parser


def main(
    *,
    default_run_name: str,
    default_output_root: Path,
    default_model_path: Path,
    default_resolution: Optional[int],
) -> None:
    args = build_arg_parser(
        default_run_name=default_run_name,
        default_output_root=default_output_root,
        default_model_path=default_model_path,
        default_resolution=default_resolution,
    ).parse_args()

    requested_data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    if args.annotations is not None:
        args.annotations = args.annotations.expanduser().resolve()

    resolved_data_root, annotations_path = resolve_dataset_layout(requested_data_root, args.annotations)
    run_dir = ensure_dir(args.output_root / args.run_name)
    sequences = resolve_sequences(
        data_root=resolved_data_root,
        seq_start=args.seq_start,
        seq_end=args.seq_end,
        seq_list=args.seq_list,
        max_frames_per_seq=args.max_frames_per_seq,
        annotations_path=annotations_path,
    )
    resolved_ball_category_id = resolve_ball_category_id(annotations_path, args.ball_category_id)

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Requested data root: {requested_data_root}")
    if resolved_data_root != requested_data_root:
        print(f"[INFO] Resolved image root: {resolved_data_root}")
    print(f"[INFO] Sequences: {', '.join(sequence.name for sequence in sequences)}")
    print(f"[INFO] Model path: {args.model_path}")
    print(
        "[INFO] Inference resolution: "
        f"{args.resolution if int(args.resolution) > 0 else 'default'}"
    )
    if annotations_path is not None:
        print(f"[INFO] Evaluation annotations: {annotations_path}")
    print(f"[INFO] Evaluation ball category id: {resolved_ball_category_id}")

    detector = RFDetrBallOnlyDetector(
        model_path=args.model_path,
        ball_class_id=args.ball_class_id,
        confidence_threshold=args.confidence_threshold,
        resolution=int(args.resolution) if int(args.resolution) > 0 else None,
        optimize_for_inference=args.optimize_for_inference,
    )

    all_detections: List[dict] = []
    per_sequence_runtime: Dict[str, dict] = {}
    processed_image_ids: List[int] = []
    total_frames = 0
    total_inference_ms = 0.0

    try:
        import torch

        inference_context = torch.inference_mode()
    except Exception:
        inference_context = contextlib.nullcontext()

    with inference_context:
        for sequence in sequences:
            print(f"[INFO] Processing {sequence.name}")
            inference_times_ms: List[float] = []
            exported_count = 0

            for frame_index, image_path in enumerate(sequence.image_paths, start=1):
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Could not read frame {image_path}")

                image_id = sequence.image_ids_by_frame.get(frame_index)
                if image_id is not None:
                    processed_image_ids.append(int(image_id))

                started_at = time.perf_counter()
                detections = detector.predict_ball_detections(image)
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0

                inference_times_ms.append(elapsed_ms)
                total_inference_ms += elapsed_ms
                total_frames += 1

                for detection in detections:
                    exported_count += 1
                    all_detections.append(
                        build_detection_record(
                            sequence=sequence.name,
                            frame_index=frame_index,
                            original_frame=sequence.original_frames_by_frame[frame_index],
                            file_name=sequence.file_names_by_frame[frame_index],
                            image_id=image_id,
                            bbox_xyxy=detection["xyxy"],
                            score=detection["score"],
                            stage="final",
                            source="rfdetr",
                        )
                    )

            per_sequence_runtime[sequence.name] = {
                "sequence": sequence.name,
                "frames_total": len(sequence.image_paths),
                "exported_detections": exported_count,
                "inference_ms_avg": mean_or_zero(inference_times_ms),
                "runtime_fps": safe_div(1000.0, mean_or_zero(inference_times_ms)),
            }
            print(
                f"[INFO] {sequence.name}: detections={exported_count}, "
                f"fps={per_sequence_runtime[sequence.name]['runtime_fps']:.2f}"
            )

    detections_path = run_dir / "detections.json"
    write_detection_export(
        path=detections_path,
        run_name=args.run_name,
        data_root=resolved_data_root,
        detections=all_detections,
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        annotations_path=annotations_path,
    )

    evaluation_summary: Dict[str, object] = {"status": "skipped", "reason": "annotations not found"}
    sequence_rows: List[dict] = []

    if annotations_path is not None and annotations_path.exists():
        evaluation = evaluate_detections(
            detections=all_detections,
            annotations_path=annotations_path,
            stage="final",
            iou_threshold=args.eval_iou,
            score_threshold=args.eval_score_threshold,
            ball_category_id=resolved_ball_category_id,
            allowed_image_ids=processed_image_ids,
        )
        per_sequence_eval = {row["sequence"]: row for row in evaluation["per_sequence"]}

        for sequence in sequences:
            runtime_row = per_sequence_runtime[sequence.name]
            eval_row = per_sequence_eval.get(
                sequence.name,
                {
                    "images_total": float(runtime_row["frames_total"]),
                    "gt_count": 0.0,
                    "detection_count": float(runtime_row["exported_detections"]),
                    "tp": 0.0,
                    "fp": 0.0,
                    "fn": 0.0,
                    "gt_frames": 0.0,
                    "no_gt_frames": float(runtime_row["frames_total"]),
                    "predicted_frames": float(runtime_row["exported_detections"] > 0),
                    "event_tp": 0.0,
                    "missed_detection_count": 0.0,
                    "false_positive_count": 0.0,
                    "tn": 0.0,
                    "no_gt_predicted": 0.0,
                    "avg_false_positive_center_distance_px": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "mean_matched_iou": 0.0,
                },
            )
            sequence_rows.append(
                {
                    "sequence": sequence.name,
                    "frames_total": runtime_row["frames_total"],
                    "images_total": int(eval_row["images_total"]),
                    "gt_count": int(eval_row["gt_count"]),
                    "detections": int(eval_row["detection_count"]),
                    "tp": int(eval_row["tp"]),
                    "fp": int(eval_row["fp"]),
                    "fn": int(eval_row["fn"]),
                    "gt_frames": int(eval_row.get("gt_frames", 0.0)),
                    "no_gt_frames": int(eval_row.get("no_gt_frames", 0.0)),
                    "predicted_frames": int(eval_row.get("predicted_frames", 0.0)),
                    "event_tp": int(eval_row.get("event_tp", 0.0)),
                    "missed_detection_count": int(eval_row.get("missed_detection_count", 0.0)),
                    "false_positive_count": int(eval_row.get("false_positive_count", 0.0)),
                    "tn": int(eval_row.get("tn", 0.0)),
                    "no_gt_predicted": int(eval_row.get("no_gt_predicted", 0.0)),
                    "avg_false_positive_center_distance_px": float(
                        eval_row.get("avg_false_positive_center_distance_px", 0.0)
                    ),
                    "precision": float(eval_row["precision"]),
                    "recall": float(eval_row["recall"]),
                    "mean_matched_iou": float(eval_row["mean_matched_iou"]),
                    "inference_ms_avg": runtime_row["inference_ms_avg"],
                    "runtime_fps": runtime_row["runtime_fps"],
                }
            )

        evaluation_summary = {
            "status": "ok",
            "annotations_path": str(annotations_path),
            "ball_category_id": resolved_ball_category_id,
            "iou_threshold": args.eval_iou,
            "score_threshold": args.eval_score_threshold,
            "final": evaluation,
        }
    else:
        for sequence in sequences:
            runtime_row = per_sequence_runtime[sequence.name]
            sequence_rows.append(
                {
                    "sequence": sequence.name,
                    "frames_total": runtime_row["frames_total"],
                    "images_total": runtime_row["frames_total"],
                    "gt_count": 0,
                    "detections": runtime_row["exported_detections"],
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "gt_frames": 0,
                    "no_gt_frames": runtime_row["frames_total"],
                    "predicted_frames": 1 if runtime_row["exported_detections"] > 0 else 0,
                    "event_tp": 0,
                    "missed_detection_count": 0,
                    "false_positive_count": 0,
                    "tn": 0,
                    "no_gt_predicted": 0,
                    "avg_false_positive_center_distance_px": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "mean_matched_iou": 0.0,
                    "inference_ms_avg": runtime_row["inference_ms_avg"],
                    "runtime_fps": runtime_row["runtime_fps"],
                }
            )

    aggregate_metrics = {}
    if evaluation_summary.get("status") == "ok":
        aggregate_metrics.update(evaluation_summary["final"]["aggregate"])
    aggregate_metrics.update(
        {
            "frames_total": total_frames,
            "detections": len(all_detections),
            "inference_ms_avg": safe_div(total_inference_ms, total_frames),
            "runtime_fps": safe_div(1000.0 * total_frames, total_inference_ms),
        }
    )

    csv_write_dicts(run_dir / "sequence_summary.csv", sequence_rows)
    original_metrics_path = run_dir / "benchmark_metrics_original.csv"
    write_benchmark_metrics_csv(
        original_metrics_path,
        latency_ms=aggregate_metrics.get("inference_ms_avg", 0.0),
        tp=int(aggregate_metrics.get("event_tp", 0.0)),
        missed_detection_count=int(aggregate_metrics.get("missed_detection_count", 0.0)),
        false_positive_count=int(aggregate_metrics.get("false_positive_count", 0.0)),
        tn=int(aggregate_metrics.get("tn", 0.0)),
        no_gt_predicted=int(aggregate_metrics.get("no_gt_predicted", 0.0)),
        avg_false_positive_center_distance_px=float(
            aggregate_metrics.get("avg_false_positive_center_distance_px", 0.0)
        ),
        iou_threshold=float(args.eval_iou),
        score_threshold=float(args.eval_score_threshold),
        gt_frames=int(aggregate_metrics.get("gt_frames", 0.0)),
        no_gt_frames=int(aggregate_metrics.get("no_gt_frames", 0.0)),
        predicted_frames=int(aggregate_metrics.get("predicted_frames", 0.0)),
    )
    original_fresh_metrics_path = args.output_root / f"{args.run_name}_benchmark_metrics_original.csv"
    write_benchmark_metrics_csv(
        original_fresh_metrics_path,
        latency_ms=aggregate_metrics.get("inference_ms_avg", 0.0),
        tp=int(aggregate_metrics.get("event_tp", 0.0)),
        missed_detection_count=int(aggregate_metrics.get("missed_detection_count", 0.0)),
        false_positive_count=int(aggregate_metrics.get("false_positive_count", 0.0)),
        tn=int(aggregate_metrics.get("tn", 0.0)),
        no_gt_predicted=int(aggregate_metrics.get("no_gt_predicted", 0.0)),
        avg_false_positive_center_distance_px=float(
            aggregate_metrics.get("avg_false_positive_center_distance_px", 0.0)
        ),
        iou_threshold=float(args.eval_iou),
        score_threshold=float(args.eval_score_threshold),
        gt_frames=int(aggregate_metrics.get("gt_frames", 0.0)),
        no_gt_frames=int(aggregate_metrics.get("no_gt_frames", 0.0)),
        predicted_frames=int(aggregate_metrics.get("predicted_frames", 0.0)),
    )

    experiment_summary = {
        "run_name": args.run_name,
        "requested_data_root": str(requested_data_root),
        "resolved_data_root": str(resolved_data_root),
        "model_path": str(args.model_path),
        "resolution": int(args.resolution) if int(args.resolution) > 0 else None,
        "sequences": [sequence.name for sequence in sequences],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "detections_path": str(detections_path),
        "evaluation": evaluation_summary,
        "aggregate": aggregate_metrics,
    }
    center_distance_bucket_threshold_px = resolve_center_distance_bucket_threshold_px(
        annotations_path=annotations_path if annotations_path is not None and annotations_path.exists() else None,
        ball_category_id=int(args.eval_ball_category_id),
        fallback_threshold_px=DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
    )
    distance_rule_summary = attach_distance_rule_summary(
        experiment_summary,
        center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
        include_raw_stage=False,
    )

    adjusted_metrics_path = run_dir / "benchmark_metrics.csv"
    adjusted_metrics = write_single_stage_distance_rule_metrics_csv(
        adjusted_metrics_path,
        latency_ms=aggregate_metrics.get("inference_ms_avg", 0.0),
        iou_threshold=float(args.eval_iou),
        score_threshold=float(args.eval_score_threshold),
        aggregate=evaluation_summary.get("final", {}).get("aggregate", {})
        if isinstance(evaluation_summary, dict)
        else {},
        center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
    )
    fresh_metrics_path = args.output_root / f"{args.run_name}_benchmark_metrics.csv"
    write_single_stage_distance_rule_metrics_csv(
        fresh_metrics_path,
        latency_ms=aggregate_metrics.get("inference_ms_avg", 0.0),
        iou_threshold=float(args.eval_iou),
        score_threshold=float(args.eval_score_threshold),
        aggregate=evaluation_summary.get("final", {}).get("aggregate", {})
        if isinstance(evaluation_summary, dict)
        else {},
        center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
    )

    with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_summary, handle, indent=2)

    print(f"[INFO] Detection export written to {detections_path}")
    print(f"[INFO] Sequence summary written to {run_dir / 'sequence_summary.csv'}")
    print(f"[INFO] Benchmark metrics written to {run_dir / 'benchmark_metrics.csv'}")
    print(f"[INFO] Original metrics preserved at {original_metrics_path}")
    print(f"[INFO] Fresh metrics CSV written to {fresh_metrics_path}")
    print(f"[INFO] Original fresh metrics CSV written to {original_fresh_metrics_path}")
    print(f"[INFO] Experiment summary written to {run_dir / 'experiment_summary.json'}")
    final_adjusted = distance_rule_summary.get("final", {}) if isinstance(distance_rule_summary, dict) else {}
    if final_adjusted:
        print(
            "[INFO] Metric aggregate (distance-threshold bucket kept separate): "
            f"tp={int(final_adjusted.get('tp', 0))}, "
            f"missed={int(final_adjusted.get('missed_detection_count', 0))}, "
            f"fp={int(final_adjusted.get('false_positive_count', 0))}, "
            f"tn={int(final_adjusted.get('tn', 0))}, "
            f"no_gt_count={int(final_adjusted.get('no_gt_count', 0))}, "
            f"distance_threshold_px={float(final_adjusted.get('center_distance_bucket_threshold_px', 0.0)):.2f}, "
            f"distance_le_threshold_px_count={int(final_adjusted.get('distance_le_threshold_px_count', 0))}"
        )
    if aggregate_metrics:
        print(
            "[INFO] Aggregate: "
            f"tp={int(aggregate_metrics.get('event_tp', 0.0))}, "
            f"missed={int(aggregate_metrics.get('missed_detection_count', 0.0))}, "
            f"fp={int(aggregate_metrics.get('false_positive_count', 0.0))}, "
            f"tn={int(aggregate_metrics.get('tn', 0.0))}, "
            f"no_gt_predicted={int(aggregate_metrics.get('no_gt_predicted', 0.0))}, "
            f"fps={aggregate_metrics.get('runtime_fps', 0.0):.2f}"
        )
