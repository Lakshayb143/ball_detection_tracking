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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _eval_rfdetr_ball_only_common import (  # noqa: E402
    resolve_ball_category_id,
    resolve_dataset_layout,
    str2bool,
)
from ball_detection_metrics import (  # noqa: E402
    build_detection_record,
    evaluate_detections,
    safe_div,
    write_benchmark_metrics_csv,
    write_detection_export,
)
from benchmark_dataset import resolve_sequences  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "benchmark_sets" / "central_test_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "central_test_v1_rfdetr_1120_sahi_fresh"
DEFAULT_RUN_NAME = "rfdetr_1120_sahi_central_test_v1_slice640_overlap20"
DEFAULT_MODEL_PATH = REPO_ROOT / "checkpoints" / "ball_1120.pth"
DEFAULT_BALL_CLASS_ID = 0
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_EVAL_IOU = 0.01
DEFAULT_EVAL_SCORE_THRESHOLD = 0.5
DEFAULT_RESOLUTION = 1120
DEFAULT_SLICE_SIZE = 640
DEFAULT_OVERLAP_RATIO = 0.2


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


def score_value(score: object) -> float:
    value = getattr(score, "value", score)
    return float(value)


class RFDetrSahiDetector:
    def __init__(
        self,
        model_path: Path,
        ball_class_id: int,
        confidence_threshold: float,
        resolution: int,
        device: str,
        optimize_for_inference: bool,
        slice_height: int,
        slice_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
        perform_standard_pred: bool,
        postprocess_type: str,
        postprocess_match_metric: str,
        postprocess_match_threshold: float,
        postprocess_class_agnostic: bool,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"RF-DETR checkpoint not found: {model_path}")

        self.ball_class_id = int(ball_class_id)
        self.confidence_threshold = float(confidence_threshold)
        self.slice_height = int(slice_height)
        self.slice_width = int(slice_width)
        self.overlap_height_ratio = float(overlap_height_ratio)
        self.overlap_width_ratio = float(overlap_width_ratio)
        self.perform_standard_pred = bool(perform_standard_pred)
        self.postprocess_type = postprocess_type
        self.postprocess_match_metric = postprocess_match_metric
        self.postprocess_match_threshold = float(postprocess_match_threshold)
        self.postprocess_class_agnostic = bool(postprocess_class_agnostic)

        from rfdetr.detr import RFDETRMedium
        from sahi import AutoDetectionModel

        rfdetr_model = RFDETRMedium(
            pretrain_weights=str(model_path),
            resolution=int(resolution),
            device=device,
        )
        class_names = list(getattr(rfdetr_model, "class_names", None) or [])
        category_mapping = {idx: name for idx, name in enumerate(class_names)}
        if self.ball_class_id not in category_mapping:
            fallback_name = "ball" if not category_mapping else f"class_{self.ball_class_id}"
            category_mapping[self.ball_class_id] = fallback_name

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="roboflow",
            model=rfdetr_model,
            device=device,
            confidence_threshold=self.confidence_threshold,
            category_mapping=category_mapping,
        )

        if optimize_for_inference and hasattr(self.detection_model.model, "optimize_for_inference"):
            try:
                self.detection_model.model.optimize_for_inference()
                print("[INFO] Optimized RF-DETR model for inference")
            except Exception as exc:
                print(f"[WARN] Could not optimize RF-DETR model for inference: {exc}")

    def predict_ball_detections(self, image_bgr: np.ndarray) -> Tuple[List[dict], Dict[str, float]]:
        from sahi.predict import get_sliced_prediction

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = get_sliced_prediction(
            image=image_rgb,
            detection_model=self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            perform_standard_pred=self.perform_standard_pred,
            postprocess_type=self.postprocess_type,
            postprocess_match_metric=self.postprocess_match_metric,
            postprocess_match_threshold=self.postprocess_match_threshold,
            postprocess_class_agnostic=self.postprocess_class_agnostic,
            auto_slice_resolution=False,
            verbose=0,
        )

        output: List[dict] = []
        for object_prediction in result.object_prediction_list:
            if object_prediction is None:
                continue
            if int(object_prediction.category.id) != self.ball_class_id:
                continue
            output.append(
                {
                    "xyxy": np.asarray(object_prediction.bbox.to_xyxy(), dtype=np.float32),
                    "score": score_value(object_prediction.score),
                }
            )
        return output, {key: float(value) * 1000.0 for key, value in result.durations_in_seconds.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark ball_1120.pth with SAHI sliced inference and custom precision/recall logic."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--seq_start", type=int, default=0)
    parser.add_argument("--seq_end", type=int, default=999)
    parser.add_argument("--seq_list", type=str, default="")
    parser.add_argument("--max_frames_per_seq", type=int, default=0)

    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ball_class_id", type=int, default=DEFAULT_BALL_CLASS_ID)
    parser.add_argument("--ball_category_id", type=int, default=None)
    parser.add_argument("--confidence_threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--eval_iou", type=float, default=DEFAULT_EVAL_IOU)
    parser.add_argument("--eval_score_threshold", type=float, default=DEFAULT_EVAL_SCORE_THRESHOLD)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--optimize_for_inference", type=str2bool, default=True)

    parser.add_argument("--slice_height", type=int, default=DEFAULT_SLICE_SIZE)
    parser.add_argument("--slice_width", type=int, default=DEFAULT_SLICE_SIZE)
    parser.add_argument("--overlap_height_ratio", type=float, default=DEFAULT_OVERLAP_RATIO)
    parser.add_argument("--overlap_width_ratio", type=float, default=DEFAULT_OVERLAP_RATIO)
    parser.add_argument("--perform_standard_pred", type=str2bool, default=True)
    parser.add_argument("--postprocess_type", type=str, default="GREEDYNMM")
    parser.add_argument("--postprocess_match_metric", type=str, default="IOS")
    parser.add_argument("--postprocess_match_threshold", type=float, default=0.5)
    parser.add_argument("--postprocess_class_agnostic", type=str2bool, default=True)
    return parser


def prepare_run_dir(output_root: Path, run_name: str, overwrite: bool) -> Path:
    run_dir = output_root / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}. "
            "Use --overwrite true or choose a new --run_name."
        )
    ensure_dir(run_dir)
    return run_dir


def write_fresh_metrics_csv(path: Path, row: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = build_arg_parser().parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    if args.annotations is not None:
        args.annotations = args.annotations.expanduser().resolve()

    args.data_root, annotations_path = resolve_dataset_layout(args.data_root, args.annotations)
    resolved_ball_category_id = resolve_ball_category_id(annotations_path, args.ball_category_id)
    run_dir = prepare_run_dir(args.output_root, args.run_name, args.overwrite)

    sequences = resolve_sequences(
        data_root=args.data_root,
        seq_start=args.seq_start,
        seq_end=args.seq_end,
        seq_list=args.seq_list,
        max_frames_per_seq=args.max_frames_per_seq,
    )

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Fresh metrics CSV: {args.output_root / 'sahi_benchmark_metrics.csv'}")
    print(f"[INFO] Data root: {args.data_root}")
    print(f"[INFO] Sequences: {', '.join(sequence.name for sequence in sequences)}")
    print(f"[INFO] Model path: {args.model_path}")
    print(f"[INFO] RF-DETR resolution: {args.resolution}")
    print(
        "[INFO] SAHI: "
        f"slice={args.slice_width}x{args.slice_height}, "
        f"overlap=({args.overlap_width_ratio}, {args.overlap_height_ratio}), "
        f"standard_pred={args.perform_standard_pred}"
    )
    if annotations_path is not None:
        print(f"[INFO] Evaluation annotations: {annotations_path}")
    print(f"[INFO] Evaluation ball category id: {resolved_ball_category_id}")

    detector = RFDetrSahiDetector(
        model_path=args.model_path,
        ball_class_id=args.ball_class_id,
        confidence_threshold=args.confidence_threshold,
        resolution=args.resolution,
        device=args.device,
        optimize_for_inference=args.optimize_for_inference,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        perform_standard_pred=args.perform_standard_pred,
        postprocess_type=args.postprocess_type,
        postprocess_match_metric=args.postprocess_match_metric,
        postprocess_match_threshold=args.postprocess_match_threshold,
        postprocess_class_agnostic=args.postprocess_class_agnostic,
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
            sahi_slice_times_ms: List[float] = []
            sahi_prediction_times_ms: List[float] = []
            sahi_postprocess_times_ms: List[float] = []
            exported_count = 0

            for frame_index, image_path in enumerate(sequence.image_paths, start=1):
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Could not read frame {image_path}")

                image_id = sequence.image_ids_by_frame.get(frame_index)
                if image_id is not None:
                    processed_image_ids.append(int(image_id))

                started_at = time.perf_counter()
                detections, sahi_durations_ms = detector.predict_ball_detections(image)
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0

                inference_times_ms.append(elapsed_ms)
                sahi_slice_times_ms.append(sahi_durations_ms.get("slice", 0.0))
                sahi_prediction_times_ms.append(sahi_durations_ms.get("prediction", 0.0))
                sahi_postprocess_times_ms.append(sahi_durations_ms.get("postprocess", 0.0))
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
                            source="rfdetr_1120_sahi",
                        )
                    )

            per_sequence_runtime[sequence.name] = {
                "sequence": sequence.name,
                "frames_total": len(sequence.image_paths),
                "exported_detections": exported_count,
                "inference_ms_avg": mean_or_zero(inference_times_ms),
                "sahi_slice_ms_avg": mean_or_zero(sahi_slice_times_ms),
                "sahi_prediction_ms_avg": mean_or_zero(sahi_prediction_times_ms),
                "sahi_postprocess_ms_avg": mean_or_zero(sahi_postprocess_times_ms),
                "runtime_fps": safe_div(1000.0, mean_or_zero(inference_times_ms)),
            }
            print(
                f"[INFO] {sequence.name}: detections={exported_count}, "
                f"fps={per_sequence_runtime[sequence.name]['runtime_fps']:.2f}"
            )

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["resolved_data_root"] = str(args.data_root)
    config["resolved_annotations"] = str(annotations_path) if annotations_path is not None else None
    config["resolved_ball_category_id"] = resolved_ball_category_id

    detections_path = run_dir / "detections.json"
    write_detection_export(
        path=detections_path,
        run_name=args.run_name,
        data_root=args.data_root,
        detections=all_detections,
        config=config,
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
                    "precision": float(eval_row["precision"]),
                    "recall": float(eval_row["recall"]),
                    "mean_matched_iou": float(eval_row["mean_matched_iou"]),
                    "inference_ms_avg": runtime_row["inference_ms_avg"],
                    "sahi_slice_ms_avg": runtime_row["sahi_slice_ms_avg"],
                    "sahi_prediction_ms_avg": runtime_row["sahi_prediction_ms_avg"],
                    "sahi_postprocess_ms_avg": runtime_row["sahi_postprocess_ms_avg"],
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
                    "precision": 0.0,
                    "recall": 0.0,
                    "mean_matched_iou": 0.0,
                    "inference_ms_avg": runtime_row["inference_ms_avg"],
                    "sahi_slice_ms_avg": runtime_row["sahi_slice_ms_avg"],
                    "sahi_prediction_ms_avg": runtime_row["sahi_prediction_ms_avg"],
                    "sahi_postprocess_ms_avg": runtime_row["sahi_postprocess_ms_avg"],
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
            "sahi_slice_ms_avg": mean_or_zero(
                [row["sahi_slice_ms_avg"] for row in per_sequence_runtime.values()]
            ),
            "sahi_prediction_ms_avg": mean_or_zero(
                [row["sahi_prediction_ms_avg"] for row in per_sequence_runtime.values()]
            ),
            "sahi_postprocess_ms_avg": mean_or_zero(
                [row["sahi_postprocess_ms_avg"] for row in per_sequence_runtime.values()]
            ),
        }
    )

    csv_write_dicts(run_dir / "sequence_summary.csv", sequence_rows)
    write_benchmark_metrics_csv(
        run_dir / "benchmark_metrics.csv",
        precision=aggregate_metrics.get("precision", 0.0),
        recall=aggregate_metrics.get("recall", 0.0),
        latency_ms=aggregate_metrics.get("inference_ms_avg", 0.0),
    )

    experiment_summary = {
        "run_name": args.run_name,
        "data_root": str(args.data_root),
        "model_path": str(args.model_path),
        "resolution": int(args.resolution),
        "sequences": [sequence.name for sequence in sequences],
        "config": config,
        "detections_path": str(detections_path),
        "evaluation": evaluation_summary,
        "aggregate": aggregate_metrics,
    }
    with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_summary, handle, indent=2)

    fresh_metrics_row = {
        "benchmark_name": args.output_root.name,
        "run_name": args.run_name,
        "output_dir": str(run_dir),
        "data_root": str(args.data_root),
        "frames_total": aggregate_metrics.get("frames_total", 0),
        "gt_count": aggregate_metrics.get("gt_count", 0),
        "detections": aggregate_metrics.get("detections", 0),
        "tp": aggregate_metrics.get("tp", 0),
        "fp": aggregate_metrics.get("fp", 0),
        "fn": aggregate_metrics.get("fn", 0),
        "precision": aggregate_metrics.get("precision", 0.0),
        "recall": aggregate_metrics.get("recall", 0.0),
        "latency_ms": aggregate_metrics.get("inference_ms_avg", 0.0),
        "runtime_fps": aggregate_metrics.get("runtime_fps", 0.0),
        "model_path": str(args.model_path),
        "resolution": args.resolution,
        "confidence_threshold": args.confidence_threshold,
        "eval_iou": args.eval_iou,
        "eval_score_threshold": args.eval_score_threshold,
        "slice_width": args.slice_width,
        "slice_height": args.slice_height,
        "overlap_width_ratio": args.overlap_width_ratio,
        "overlap_height_ratio": args.overlap_height_ratio,
        "perform_standard_pred": args.perform_standard_pred,
        "postprocess_type": args.postprocess_type,
        "postprocess_match_metric": args.postprocess_match_metric,
        "postprocess_match_threshold": args.postprocess_match_threshold,
        "postprocess_class_agnostic": args.postprocess_class_agnostic,
    }
    write_fresh_metrics_csv(args.output_root / "sahi_benchmark_metrics.csv", fresh_metrics_row)

    print(f"[INFO] Detection export written to {detections_path}")
    print(f"[INFO] Sequence summary written to {run_dir / 'sequence_summary.csv'}")
    print(f"[INFO] Benchmark metrics written to {run_dir / 'benchmark_metrics.csv'}")
    print(f"[INFO] Fresh metrics CSV written to {args.output_root / 'sahi_benchmark_metrics.csv'}")
    print(f"[INFO] Experiment summary written to {run_dir / 'experiment_summary.json'}")
    if aggregate_metrics:
        print(
            "[INFO] Aggregate: "
            f"precision={aggregate_metrics.get('precision', 0.0):.4f}, "
            f"recall={aggregate_metrics.get('recall', 0.0):.4f}, "
            f"latency_ms={aggregate_metrics.get('inference_ms_avg', 0.0):.2f}, "
            f"fps={aggregate_metrics.get('runtime_fps', 0.0):.2f}"
        )


if __name__ == "__main__":
    main()
