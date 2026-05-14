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
from benchmark_dataset import default_annotation_path, resolve_sequences  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "test"
DEFAULT_MODEL_PATH = REPO_ROOT / "samy_models" / "best_combined_v2.pt"
DEFAULT_SEQ_START = 0
DEFAULT_SEQ_END = 999
DEFAULT_SEQ_LIST = ""
DEFAULT_MAX_FRAMES_PER_SEQ = 0
DEFAULT_BALL_CLASS_ID = 0
DEFAULT_CONFIDENCE_THRESHOLD = 0.01
DEFAULT_EVAL_IOU = 0.01
DEFAULT_EVAL_SCORE_THRESHOLD = 0.01
DEFAULT_IMAGE_SIZE = 1280
DEFAULT_DEVICE = ""


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


def patch_ultralytics_numpy_bridge() -> None:
    from ultralytics.engine.predictor import BasePredictor

    if getattr(BasePredictor, "_codex_numpy_bridge_patch", False):
        return

    def patched_preprocess(self, im):
        import torch

        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)
            try:
                im = torch.from_numpy(im)
            except RuntimeError as exc:
                if "Numpy is not available" not in str(exc):
                    raise
                # Fall back to a pure-Python conversion when torch's NumPy bridge is unavailable.
                im = torch.tensor(im.tolist(), dtype=torch.uint8)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        if not_tensor:
            im /= 255
        return im

    BasePredictor.preprocess = patched_preprocess
    BasePredictor._codex_numpy_bridge_patch = True


class YoloBallOnlyDetector:
    def __init__(
        self,
        model_path: Path,
        ball_class_id: int,
        confidence_threshold: float,
        image_size: Optional[int],
        device: str,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")

        from ultralytics import YOLO

        patch_ultralytics_numpy_bridge()

        self.ball_class_id = int(ball_class_id)
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size) if image_size is not None and int(image_size) > 0 else None
        self.device = str(device).strip()
        self.model = YOLO(str(model_path))

        class_names = getattr(self.model, "names", {}) or {}
        print(f"[INFO] Loaded YOLO model: {model_path}")
        print(f"[INFO] Model classes: {class_names}")

    def predict_ball_detections(self, image_bgr: np.ndarray) -> List[dict]:
        predict_kwargs = {
            "source": image_bgr,
            "conf": self.confidence_threshold,
            "classes": [self.ball_class_id],
            "verbose": False,
        }
        if self.image_size is not None:
            predict_kwargs["imgsz"] = self.image_size
        if self.device:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidence = boxes.conf.detach().cpu().tolist()
        class_ids = [int(value) for value in boxes.cls.detach().cpu().tolist()]

        num_items = min(len(xyxy), len(confidence), len(class_ids))
        if num_items <= 0:
            return []

        output: List[dict] = []
        for index in range(num_items):
            if int(class_ids[index]) != self.ball_class_id:
                continue
            output.append(
                {
                    "xyxy": np.asarray(xyxy[index], dtype=np.float32),
                    "score": float(confidence[index]),
                }
            )
        return output


def build_arg_parser(
    *,
    default_run_name: str,
    default_output_root: Path,
    default_model_path: Path,
    default_image_size: Optional[int],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the YOLO ball detector without any tracking logic."
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
    parser.add_argument("--imgsz", type=int, default=default_image_size or 0)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    return parser


def main(
    *,
    default_run_name: str,
    default_output_root: Path,
    default_model_path: Path,
    default_image_size: Optional[int],
) -> None:
    args = build_arg_parser(
        default_run_name=default_run_name,
        default_output_root=default_output_root,
        default_model_path=default_model_path,
        default_image_size=default_image_size,
    ).parse_args()

    requested_data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    if args.annotations is not None:
        args.annotations = args.annotations.expanduser().resolve()

    resolved_data_root, annotations_path = resolve_dataset_layout(requested_data_root, args.annotations)
    resolved_ball_category_id = resolve_ball_category_id(annotations_path, args.ball_category_id)
    run_dir = ensure_dir(args.output_root / args.run_name)

    sequences = resolve_sequences(
        data_root=resolved_data_root,
        seq_start=args.seq_start,
        seq_end=args.seq_end,
        seq_list=args.seq_list,
        max_frames_per_seq=args.max_frames_per_seq,
        annotations_path=annotations_path,
    )

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Requested data root: {requested_data_root}")
    if resolved_data_root != requested_data_root:
        print(f"[INFO] Resolved image root: {resolved_data_root}")
    print(f"[INFO] Sequences: {', '.join(sequence.name for sequence in sequences)}")
    print(f"[INFO] Model path: {args.model_path}")
    print(
        "[INFO] Inference image size: "
        f"{args.imgsz if int(args.imgsz) > 0 else 'default'}"
    )
    if args.device.strip():
        print(f"[INFO] Inference device: {args.device}")
    if annotations_path is not None:
        print(f"[INFO] Evaluation annotations: {annotations_path}")
    print(f"[INFO] Evaluation ball category id: {resolved_ball_category_id}")

    detector = YoloBallOnlyDetector(
        model_path=args.model_path,
        ball_class_id=args.ball_class_id,
        confidence_threshold=args.confidence_threshold,
        image_size=int(args.imgsz) if int(args.imgsz) > 0 else None,
        device=args.device,
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
                            source="yolo",
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
    export_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    export_config["requested_data_root"] = str(requested_data_root)
    export_config["resolved_data_root"] = str(resolved_data_root)

    write_detection_export(
        path=detections_path,
        run_name=args.run_name,
        data_root=resolved_data_root,
        detections=all_detections,
        config=export_config,
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
    write_benchmark_metrics_csv(
        run_dir / "benchmark_metrics.csv",
        precision=aggregate_metrics.get("precision", 0.0),
        recall=aggregate_metrics.get("recall", 0.0),
        latency_ms=aggregate_metrics.get("inference_ms_avg", 0.0),
    )

    experiment_summary = {
        "run_name": args.run_name,
        "requested_data_root": str(requested_data_root),
        "resolved_data_root": str(resolved_data_root),
        "model_path": str(args.model_path),
        "imgsz": int(args.imgsz) if int(args.imgsz) > 0 else None,
        "device": args.device or None,
        "sequences": [sequence.name for sequence in sequences],
        "config": export_config,
        "detections_path": str(detections_path),
        "evaluation": evaluation_summary,
        "aggregate": aggregate_metrics,
    }
    with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_summary, handle, indent=2)

    print(f"[INFO] Detection export written to {detections_path}")
    print(f"[INFO] Sequence summary written to {run_dir / 'sequence_summary.csv'}")
    print(f"[INFO] Benchmark metrics written to {run_dir / 'benchmark_metrics.csv'}")
    print(f"[INFO] Experiment summary written to {run_dir / 'experiment_summary.json'}")
    if aggregate_metrics:
        print(
            "[INFO] Aggregate: "
            f"precision={aggregate_metrics.get('precision', 0.0):.4f}, "
            f"recall={aggregate_metrics.get('recall', 0.0):.4f}, "
            f"fps={aggregate_metrics.get('runtime_fps', 0.0):.2f}"
        )
