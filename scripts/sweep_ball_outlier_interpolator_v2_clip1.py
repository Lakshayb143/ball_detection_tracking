#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ball_outlier_interpolator_v2 as v2  # noqa: E402
import benchmark_ball_outlier_interpolator_v2_clip1 as bench  # noqa: E402
from ball_detection_metrics import (  # noqa: E402
    build_detection_record,
    evaluate_detections,
    write_benchmark_metrics_csv,
    write_detection_export,
)
from benchmark_dataset import default_annotation_path, resolve_sequences  # noqa: E402
from benchmark_rfdetr_outlier_gdino_fallback_tracking import (  # noqa: E402
    center_wh_to_xyxy,
    clip_xyxy,
    csv_write_dicts,
    ensure_dir,
    mean_or_zero,
    safe_div,
    str2bool,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "clip1_fresh_runs" / "v2_hparam_sweep"
DEFAULT_SWEEP_NAME = "ball_outlier_interpolator_v2_hparam_sweep"
DEFAULT_HISTORY_FRAMES = [10, 15, 20, 25, 30, 40]
DEFAULT_MAX_GAP_FRAMES = [10, 15, 20, 25, 30]
DEFAULT_OUTLIER_CONFIRM_FRAMES = [5, 10, 15]

_CACHE: List[dict] = []
_BASE_CONFIG: Dict[str, object] = {}


def parse_int_list(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return [int(item) for item in items]


def int_list_text(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def run_name_for(history_frames: int, max_gap_frames: int, outlier_confirm_frames: int) -> str:
    return (
        f"v2_h{int(history_frames):02d}"
        f"_gap{int(max_gap_frames):02d}"
        f"_confirm{int(outlier_confirm_frames):02d}"
    )


def detections_from_frame(frame: dict) -> sv.Detections:
    xyxy = np.asarray(frame["xyxy"], dtype=np.float32)
    if xyxy.size == 0:
        xyxy = np.empty((0, 4), dtype=np.float32)
    else:
        xyxy = xyxy.reshape((-1, 4))

    confidence = np.asarray(frame["confidence"], dtype=np.float32)
    class_id = np.asarray(frame["class_id"], dtype=np.int32)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def serialize_detections(detections: sv.Detections) -> dict:
    xyxy = np.asarray(detections.xyxy, dtype=np.float32).reshape((-1, 4))
    confidence = (
        np.asarray(detections.confidence, dtype=np.float32)
        if detections.confidence is not None
        else np.zeros((len(xyxy),), dtype=np.float32)
    )
    class_id = (
        np.asarray(detections.class_id, dtype=np.int32)
        if detections.class_id is not None
        else np.zeros((len(xyxy),), dtype=np.int32)
    )
    return {
        "xyxy": xyxy.tolist(),
        "confidence": confidence.tolist(),
        "class_id": class_id.tolist(),
    }


def event_metric_stats(aggregate: Dict[str, object], prefix: str) -> Dict[str, float]:
    tp = float(aggregate.get(f"{prefix}_event_tp", 0.0))
    missed = float(aggregate.get(f"{prefix}_missed_detection_count", 0.0))
    fp = float(aggregate.get(f"{prefix}_false_positive_count", 0.0))
    no_gt_predicted = float(aggregate.get(f"{prefix}_no_gt_predicted", 0.0))
    event_fp_total = fp + no_gt_predicted
    precision = safe_div(tp, tp + event_fp_total)
    recall = safe_div(tp, tp + missed)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        f"{prefix}_event_precision": precision,
        f"{prefix}_event_recall": recall,
        f"{prefix}_event_f1": f1,
        f"{prefix}_event_fp_total": event_fp_total,
    }


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep v2 tracker hyperparameters for clip1 while caching RF-DETR detections once."
        )
    )
    parser.add_argument("--data_root", type=Path, default=bench.DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sweep_name", type=str, default=DEFAULT_SWEEP_NAME)
    parser.add_argument("--seq_start", type=int, default=bench.DEFAULT_SEQ_START)
    parser.add_argument("--seq_end", type=int, default=bench.DEFAULT_SEQ_END)
    parser.add_argument("--seq_list", type=str, default=bench.DEFAULT_SEQ_LIST)
    parser.add_argument("--max_frames_per_seq", type=int, default=bench.DEFAULT_MAX_FRAMES_PER_SEQ)

    parser.add_argument("--ball_model_path", type=Path, default=bench.DEFAULT_BALL_MODEL_PATH)
    parser.add_argument("--ball_model_resolution", type=int, default=bench.DEFAULT_BALL_MODEL_RESOLUTION)
    parser.add_argument("--ball_confidence", type=float, default=bench.DEFAULT_BALL_CONFIDENCE)
    parser.add_argument("--ball_class_id", type=int, default=bench.DEFAULT_BALL_CLASS_ID)
    parser.add_argument("--enable_rfdetr_optimize", type=str2bool, default=bench.DEFAULT_ENABLE_RFDETR_OPTIMIZE)

    parser.add_argument("--fps_assumption", type=float, default=bench.DEFAULT_FPS_ASSUMPTION)
    parser.add_argument("--mahalanobis_gate", type=float, default=float(v2.MAHALANOBIS_GATE))
    parser.add_argument("--interpolated_box_size", type=float, default=bench.DEFAULT_INTERPOLATED_BOX_SIZE)

    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--match_iou", type=float, default=bench.DEFAULT_MATCH_IOU)
    parser.add_argument("--eval_score_threshold", type=float, default=bench.DEFAULT_EVAL_SCORE_THRESHOLD)
    parser.add_argument("--ball_category_id", type=int, default=bench.DEFAULT_BALL_CATEGORY_ID)

    parser.add_argument("--history_frames", type=parse_int_list, default=DEFAULT_HISTORY_FRAMES)
    parser.add_argument("--max_gap_frames", type=parse_int_list, default=DEFAULT_MAX_GAP_FRAMES)
    parser.add_argument("--outlier_confirm_frames", type=parse_int_list, default=DEFAULT_OUTLIER_CONFIRM_FRAMES)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    return parser


def cache_metadata(args: argparse.Namespace, sequences: Sequence[object], annotations_path: Optional[Path]) -> dict:
    return {
        "data_root": str(args.data_root),
        "sequences": [sequence.name for sequence in sequences],
        "max_frames_per_seq": int(args.max_frames_per_seq),
        "ball_model_path": str(args.ball_model_path),
        "ball_model_resolution": int(args.ball_model_resolution),
        "ball_confidence": float(args.ball_confidence),
        "ball_class_id": int(args.ball_class_id),
        "annotations_path": str(annotations_path) if annotations_path is not None else None,
    }


def cache_matches(path: Path, metadata: dict) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("metadata") == metadata


def build_detection_cache(
    args: argparse.Namespace,
    sequences: Sequence[object],
    annotations_path: Optional[Path],
    cache_path: Path,
) -> List[dict]:
    metadata = cache_metadata(args, sequences, annotations_path)
    if not args.rebuild_cache and cache_matches(cache_path, metadata):
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"[INFO] Reusing detection cache: {cache_path}")
        return list(payload["sequences"])

    print("[INFO] Building RF-DETR detection cache once for the full sweep")
    model_kwargs = {"pretrain_weights": str(args.ball_model_path)}
    if int(args.ball_model_resolution) > 0:
        model_kwargs["resolution"] = int(args.ball_model_resolution)
    model = RFDETRMedium(**model_kwargs)
    if args.enable_rfdetr_optimize:
        try:
            model.optimize_for_inference()
            print("[INFO] Optimized RF-DETR model for inference")
        except Exception as exc:
            print(f"[WARN] Could not optimize RF-DETR model: {exc}")

    cached_sequences: List[dict] = []
    for sequence in sequences:
        frames: List[dict] = []
        print(f"[INFO] Caching detections for {sequence.name} ({len(sequence.image_paths)} frames)")
        for frame_id, image_path in enumerate(sequence.image_paths, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read frame {image_path}")

            started_at = time.perf_counter()
            detections = model.predict(image, confidence=args.ball_confidence)
            primary_ms = (time.perf_counter() - started_at) * 1000.0
            ball_detections = detections[detections.class_id == args.ball_class_id]
            serialized = serialize_detections(ball_detections)
            frames.append(
                {
                    "frame": int(frame_id),
                    "image_path": str(image_path),
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "image_id": sequence.image_ids_by_frame.get(frame_id),
                    "file_name": sequence.file_names_by_frame[frame_id],
                    "original_frame": int(sequence.original_frames_by_frame[frame_id]),
                    "primary_ms": float(primary_ms),
                    **serialized,
                }
            )
        cached_sequences.append({"name": sequence.name, "frames": frames})

    payload = {"metadata": metadata, "sequences": cached_sequences}
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] Detection cache written to {cache_path}")
    return cached_sequences


def process_cached_sequence(
    sequence: dict,
    config: Dict[str, object],
    run_dir: Path,
) -> Tuple[bench.SequenceSummary, List[dict], List[dict], List[int]]:
    seq_output_dir = ensure_dir(run_dir / str(sequence["name"]))
    tracker = bench.V2TrackerAdapter(
        fps_assumption=float(config["fps_assumption"]),
        confidence_threshold=float(config["ball_confidence"]),
        mahalanobis_gate=float(config["mahalanobis_gate"]),
        max_gap_frames=int(config["max_gap_frames"]),
        outlier_confirm_frames=int(config["outlier_confirm_frames"]),
    )

    frame_records: List[bench.FrameRecord] = []
    point_records: List[dict] = []
    detection_records: List[dict] = []
    processed_image_ids: List[int] = []
    raw_pred_frames = 0
    final_pred_frames = 0
    interpolated_frames = 0

    for frame in sequence["frames"]:
        tracking_start = time.perf_counter()
        frame_id = int(frame["frame"])
        image_id = frame.get("image_id")
        if image_id is not None:
            processed_image_ids.append(int(image_id))

        ball_detections = detections_from_frame(frame)
        raw_best_detection = bench._top_confidence_detection(ball_detections)
        if raw_best_detection is not None:
            raw_pred_frames += 1
            raw_box = clip_xyxy(
                np.asarray(raw_best_detection.xyxy[0], dtype=np.float32),
                width=int(frame["width"]),
                height=int(frame["height"]),
            )
            detection_records.append(
                build_detection_record(
                    sequence=str(sequence["name"]),
                    frame_index=frame_id,
                    original_frame=int(frame["original_frame"]),
                    file_name=str(frame["file_name"]),
                    image_id=image_id,
                    bbox_xyxy=raw_box,
                    score=float(raw_best_detection.confidence[0]),
                    stage="raw",
                    source="rfdetr",
                )
            )

        best_detection = tracker.select_best_detection(ball_detections)
        output_position, is_interpolated, accepted_detection = tracker.step(best_detection, frame_id)
        tracking_ms = (time.perf_counter() - tracking_start) * 1000.0

        selected_score = float(best_detection["sv"].confidence[0]) if best_detection is not None else 0.0
        selected_in_gate = bool(best_detection["in_gate"]) if best_detection is not None else False
        output_score = 0.0
        output_source = "none"
        if accepted_detection is not None and len(accepted_detection.xyxy) > 0:
            final_box = clip_xyxy(
                np.asarray(accepted_detection.xyxy[0], dtype=np.float32),
                width=int(frame["width"]),
                height=int(frame["height"]),
            )
            output_score = float(accepted_detection.confidence[0])
            output_source = "rfdetr"
            final_pred_frames += 1
            detection_records.append(
                build_detection_record(
                    sequence=str(sequence["name"]),
                    frame_index=frame_id,
                    original_frame=int(frame["original_frame"]),
                    file_name=str(frame["file_name"]),
                    image_id=image_id,
                    bbox_xyxy=final_box,
                    score=output_score,
                    stage="final",
                    source=output_source,
                )
            )
        elif output_position is not None and is_interpolated:
            final_box = clip_xyxy(
                center_wh_to_xyxy(
                    np.asarray(output_position, dtype=np.float32),
                    np.array(
                        [float(config["interpolated_box_size"]), float(config["interpolated_box_size"])],
                        dtype=np.float32,
                    ),
                ),
                width=int(frame["width"]),
                height=int(frame["height"]),
            )
            output_score = 0.0
            output_source = "interpolation"
            final_pred_frames += 1
            interpolated_frames += 1
            detection_records.append(
                build_detection_record(
                    sequence=str(sequence["name"]),
                    frame_index=frame_id,
                    original_frame=int(frame["original_frame"]),
                    file_name=str(frame["file_name"]),
                    image_id=image_id,
                    bbox_xyxy=final_box,
                    score=output_score,
                    stage="final",
                    source=output_source,
                )
            )

        point_records.append(
            bench._record_point_output(
                frame_idx=frame_id,
                output_position=output_position,
                accepted_detection=accepted_detection,
                is_interpolated=is_interpolated,
            )
        )

        primary_ms = float(frame.get("primary_ms", 0.0))
        total_ms = primary_ms + tracking_ms
        frame_records.append(
            bench.FrameRecord(
                frame=frame_id,
                raw_candidate_count=int(len(ball_detections.xyxy)),
                raw_best_score=float(raw_best_detection.confidence[0]) if raw_best_detection is not None else 0.0,
                selected_score=selected_score,
                selected_in_gate=selected_in_gate,
                output_score=output_score,
                output_source=output_source,
                tracker_active=tracker.track_initialized,
                gap_frames=int(tracker.frames_since_detection),
                hit_streak=int(tracker.hit_streak),
                interpolated_output=bool(output_source == "interpolation"),
                primary_ms=primary_ms,
                tracking_ms=tracking_ms,
                total_ms=total_ms,
            )
        )

    csv_write_dicts(seq_output_dir / "frame_trace.csv", [asdict(record) for record in frame_records])
    (seq_output_dir / "point_outputs.json").write_text(json.dumps(point_records, indent=2), encoding="utf-8")
    summary = bench.SequenceSummary(
        sequence=str(sequence["name"]),
        frames_total=len(frame_records),
        raw_pred_frames=raw_pred_frames,
        final_pred_frames=final_pred_frames,
        interpolated_frames=interpolated_frames,
        raw_eval_detections=0,
        raw_tp=0,
        raw_fp=0,
        raw_fn=0,
        raw_precision=0.0,
        raw_recall=0.0,
        raw_mean_iou=0.0,
        final_eval_detections=0,
        final_tp=0,
        final_fp=0,
        final_fn=0,
        final_precision=0.0,
        final_recall=0.0,
        final_mean_iou=0.0,
        primary_ms_avg=mean_or_zero([item.primary_ms for item in frame_records]),
        tracking_ms_avg=mean_or_zero([item.tracking_ms for item in frame_records]),
        total_ms_avg=mean_or_zero([item.total_ms for item in frame_records]),
        runtime_fps=safe_div(1000.0, mean_or_zero([item.total_ms for item in frame_records])),
    )
    return summary, detection_records, point_records, processed_image_ids


def run_combo(combo: Dict[str, int]) -> Dict[str, object]:
    config = dict(_BASE_CONFIG)
    config.update(combo)
    run_name = run_name_for(
        history_frames=int(combo["history_frames"]),
        max_gap_frames=int(combo["max_gap_frames"]),
        outlier_confirm_frames=int(combo["outlier_confirm_frames"]),
    )
    config["run_name"] = run_name

    output_root = Path(str(config["output_root"]))
    run_dir = ensure_dir(output_root / run_name)
    summary_path = run_dir / "experiment_summary.json"
    if bool(config.get("skip_existing")) and summary_path.exists():
        experiment = json.loads(summary_path.read_text(encoding="utf-8"))
        row = dict(experiment.get("sweep_row", {}))
        if row:
            row["status"] = "skipped"
            return row

    started_at = time.perf_counter()
    tracker_log_path = run_dir / "tracker.log"
    with tracker_log_path.open("w", encoding="utf-8") as log_handle, redirect_stdout(log_handle):
        summaries: List[bench.SequenceSummary] = []
        all_detections: List[dict] = []
        all_point_outputs: List[dict] = []
        processed_image_ids: List[int] = []
        for sequence in _CACHE:
            summary, detections, points, image_ids = process_cached_sequence(
                sequence=sequence,
                config=config,
                run_dir=run_dir,
            )
            summaries.append(summary)
            all_detections.extend(detections)
            all_point_outputs.extend(points)
            processed_image_ids.extend(image_ids)

    detections_path = run_dir / "detections.json"
    annotations_path = Path(str(config["annotations_path"])) if config.get("annotations_path") else None
    write_detection_export(
        path=detections_path,
        run_name=run_name,
        data_root=Path(str(config["data_root"])),
        detections=all_detections,
        config=config,
        annotations_path=annotations_path,
    )
    (run_dir / "point_outputs.json").write_text(json.dumps(all_point_outputs, indent=2), encoding="utf-8")

    evaluation_summary: Dict[str, object] = {"status": "skipped", "reason": "annotations not found"}
    raw_metric_aggregate: Dict[str, object] = {}
    final_metric_aggregate: Dict[str, object] = {}
    if annotations_path is not None and annotations_path.exists():
        raw_metrics = evaluate_detections(
            detections=all_detections,
            annotations_path=annotations_path,
            stage="raw",
            iou_threshold=float(config["match_iou"]),
            score_threshold=float(config["eval_score_threshold"]),
            ball_category_id=int(config["ball_category_id"]),
            allowed_image_ids=processed_image_ids,
        )
        final_metrics = evaluate_detections(
            detections=all_detections,
            annotations_path=annotations_path,
            stage="final",
            iou_threshold=float(config["match_iou"]),
            score_threshold=float(config["eval_score_threshold"]),
            ball_category_id=int(config["ball_category_id"]),
            allowed_image_ids=processed_image_ids,
        )
        raw_metric_aggregate = raw_metrics["aggregate"]
        final_metric_aggregate = final_metrics["aggregate"]
        raw_by_sequence = {row["sequence"]: row for row in raw_metrics["per_sequence"]}
        final_by_sequence = {row["sequence"]: row for row in final_metrics["per_sequence"]}
        for summary in summaries:
            raw_row = raw_by_sequence.get(summary.sequence, {})
            final_row = final_by_sequence.get(summary.sequence, {})
            summary.raw_eval_detections = int(raw_row.get("detection_count", 0.0))
            summary.raw_tp = int(raw_row.get("tp", 0.0))
            summary.raw_fp = int(raw_row.get("fp", 0.0))
            summary.raw_fn = int(raw_row.get("fn", 0.0))
            summary.raw_precision = float(raw_row.get("precision", 0.0))
            summary.raw_recall = float(raw_row.get("recall", 0.0))
            summary.raw_mean_iou = float(raw_row.get("mean_matched_iou", 0.0))
            summary.final_eval_detections = int(final_row.get("detection_count", 0.0))
            summary.final_tp = int(final_row.get("tp", 0.0))
            summary.final_fp = int(final_row.get("fp", 0.0))
            summary.final_fn = int(final_row.get("fn", 0.0))
            summary.final_precision = float(final_row.get("precision", 0.0))
            summary.final_recall = float(final_row.get("recall", 0.0))
            summary.final_mean_iou = float(final_row.get("mean_matched_iou", 0.0))

        evaluation_summary = {
            "status": "ok",
            "annotations_path": str(annotations_path),
            "ball_category_id": int(config["ball_category_id"]),
            "iou_threshold": float(config["match_iou"]),
            "score_threshold": float(config["eval_score_threshold"]),
            "raw": raw_metrics,
            "final": final_metrics,
        }

    csv_write_dicts(run_dir / "sequence_summary.csv", [asdict(summary) for summary in summaries])
    aggregate = bench.aggregate_sequence_summaries(summaries)
    if raw_metric_aggregate:
        aggregate.update(
            {
                "raw_event_tp": raw_metric_aggregate.get("event_tp", 0.0),
                "raw_missed_detection_count": raw_metric_aggregate.get("missed_detection_count", 0.0),
                "raw_false_positive_count": raw_metric_aggregate.get("false_positive_count", 0.0),
                "raw_tn": raw_metric_aggregate.get("tn", 0.0),
                "raw_no_gt_predicted": raw_metric_aggregate.get("no_gt_predicted", 0.0),
                "raw_avg_false_positive_center_distance_px": raw_metric_aggregate.get(
                    "avg_false_positive_center_distance_px", 0.0
                ),
                "final_event_tp": final_metric_aggregate.get("event_tp", 0.0),
                "final_missed_detection_count": final_metric_aggregate.get("missed_detection_count", 0.0),
                "final_false_positive_count": final_metric_aggregate.get("false_positive_count", 0.0),
                "final_tn": final_metric_aggregate.get("tn", 0.0),
                "final_no_gt_predicted": final_metric_aggregate.get("no_gt_predicted", 0.0),
                "final_avg_false_positive_center_distance_px": final_metric_aggregate.get(
                    "avg_false_positive_center_distance_px", 0.0
                ),
            }
        )

    aggregate.update(event_metric_stats(aggregate, "raw"))
    aggregate.update(event_metric_stats(aggregate, "final"))
    elapsed_s = time.perf_counter() - started_at

    write_benchmark_metrics_csv(
        run_dir / "benchmark_metrics.csv",
        latency_ms=aggregate.get("total_ms_avg", 0.0),
        iou_threshold=float(config["match_iou"]),
        score_threshold=float(config["eval_score_threshold"]),
        raw_tp=int(aggregate.get("raw_event_tp", 0.0)),
        raw_missed_detection_count=int(aggregate.get("raw_missed_detection_count", 0.0)),
        raw_false_positive_count=int(aggregate.get("raw_false_positive_count", 0.0)),
        raw_tn=int(aggregate.get("raw_tn", 0.0)),
        raw_no_gt_predicted=int(aggregate.get("raw_no_gt_predicted", 0.0)),
        raw_avg_false_positive_center_distance_px=float(
            aggregate.get("raw_avg_false_positive_center_distance_px", 0.0)
        ),
        final_tp=int(aggregate.get("final_event_tp", 0.0)),
        final_missed_detection_count=int(aggregate.get("final_missed_detection_count", 0.0)),
        final_false_positive_count=int(aggregate.get("final_false_positive_count", 0.0)),
        final_tn=int(aggregate.get("final_tn", 0.0)),
        final_no_gt_predicted=int(aggregate.get("final_no_gt_predicted", 0.0)),
        final_avg_false_positive_center_distance_px=float(
            aggregate.get("final_avg_false_positive_center_distance_px", 0.0)
        ),
        final_event_precision=float(aggregate.get("final_event_precision", 0.0)),
        final_event_recall=float(aggregate.get("final_event_recall", 0.0)),
        final_event_f1=float(aggregate.get("final_event_f1", 0.0)),
    )

    row = {
        "run_name": run_name,
        "history_frames": int(combo["history_frames"]),
        "history_frames_active": False,
        "max_gap_frames": int(combo["max_gap_frames"]),
        "outlier_confirm_frames": int(combo["outlier_confirm_frames"]),
        "final_event_f1": float(aggregate.get("final_event_f1", 0.0)),
        "final_event_precision": float(aggregate.get("final_event_precision", 0.0)),
        "final_event_recall": float(aggregate.get("final_event_recall", 0.0)),
        "final_event_tp": int(aggregate.get("final_event_tp", 0.0)),
        "final_missed_detection_count": int(aggregate.get("final_missed_detection_count", 0.0)),
        "final_false_positive_count": int(aggregate.get("final_false_positive_count", 0.0)),
        "final_no_gt_predicted": int(aggregate.get("final_no_gt_predicted", 0.0)),
        "final_avg_false_positive_center_distance_px": float(
            aggregate.get("final_avg_false_positive_center_distance_px", 0.0)
        ),
        "interpolated_frames": int(aggregate.get("interpolated_frames", 0.0)),
        "final_pred_frames": int(aggregate.get("final_pred_frames", 0.0)),
        "raw_event_tp": int(aggregate.get("raw_event_tp", 0.0)),
        "raw_missed_detection_count": int(aggregate.get("raw_missed_detection_count", 0.0)),
        "raw_false_positive_count": int(aggregate.get("raw_false_positive_count", 0.0)),
        "raw_no_gt_predicted": int(aggregate.get("raw_no_gt_predicted", 0.0)),
        "runtime_fps": float(aggregate.get("runtime_fps", 0.0)),
        "latency_ms": float(aggregate.get("total_ms_avg", 0.0)),
        "elapsed_s": elapsed_s,
        "run_dir": str(run_dir),
        "tracker_log_path": str(tracker_log_path),
        "status": "ok",
    }

    experiment_summary = {
        "run_name": run_name,
        "data_root": str(config["data_root"]),
        "sequences": [sequence["name"] for sequence in _CACHE],
        "config": config,
        "detections_path": str(detections_path),
        "point_outputs_path": str(run_dir / "point_outputs.json"),
        "evaluation": evaluation_summary,
        "aggregate": aggregate,
        "sweep_row": row,
        "logic_reference": str((REPO_ROOT / "ball_outlier_interpolator_v2.py").resolve()),
        "logic_notes": {
            "history_frames": (
                "Recorded for this sweep, but ball_outlier_interpolator_v2.py defines "
                "HISTORY_FRAMES without using it in the active v2 Kalman/outlier logic."
            ),
            "detection_cache": (
                "RF-DETR detections are cached once; each run replays the v2 tracker against "
                "the same candidate detections."
            ),
        },
    }
    summary_path.write_text(json.dumps(experiment_summary, indent=2), encoding="utf-8")
    return row


def _init_worker(cache: List[dict], base_config: Dict[str, object]) -> None:
    global _CACHE, _BASE_CONFIG
    _CACHE = cache
    _BASE_CONFIG = base_config


def sorted_results(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("final_event_f1", 0.0)),
            int(row.get("final_missed_detection_count", 10**9)),
            int(row.get("final_false_positive_count", 10**9)) + int(row.get("final_no_gt_predicted", 10**9)),
            float(row.get("final_avg_false_positive_center_distance_px", 10**9)),
            int(row.get("max_gap_frames", 10**9)),
            int(row.get("outlier_confirm_frames", 10**9)),
            int(row.get("history_frames", 10**9)),
        ),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.ball_model_path = args.ball_model_path.expanduser().resolve()
    if args.annotations is not None:
        args.annotations = args.annotations.expanduser().resolve()

    sweep_dir = ensure_dir(args.output_root / args.sweep_name)
    cache_path = sweep_dir / "detection_cache.json"
    summary_path = sweep_dir / "sweep_results.csv"
    best_path = sweep_dir / "sweep_results_ranked.csv"

    sequences = resolve_sequences(
        data_root=args.data_root,
        seq_start=args.seq_start,
        seq_end=args.seq_end,
        seq_list=args.seq_list,
        max_frames_per_seq=args.max_frames_per_seq,
    )
    annotations_path = args.annotations or default_annotation_path(args.data_root)

    combos = [
        {
            "history_frames": history_frames,
            "max_gap_frames": max_gap_frames,
            "outlier_confirm_frames": outlier_confirm_frames,
        }
        for history_frames in args.history_frames
        for max_gap_frames in args.max_gap_frames
        for outlier_confirm_frames in args.outlier_confirm_frames
    ]

    print(f"[INFO] Sweep dir: {sweep_dir}")
    print(f"[INFO] Sequences: {', '.join(sequence.name for sequence in sequences)}")
    print(f"[INFO] Planned combinations: {len(combos)}")
    print(f"[INFO] Parallel tracker jobs: {max(1, int(args.jobs))}")
    print(
        "[WARN] HISTORY_FRAMES is currently not active in ball_outlier_interpolator_v2.py; "
        "it is recorded in run names/results, but v2 scores will only change with "
        "max_gap_frames and outlier_confirm_frames unless you wire history into the tracker."
    )

    cache = build_detection_cache(args, sequences, annotations_path, cache_path)
    base_config: Dict[str, object] = {
        "data_root": str(args.data_root),
        "output_root": str(sweep_dir),
        "ball_model_path": str(args.ball_model_path),
        "ball_model_resolution": int(args.ball_model_resolution),
        "ball_confidence": float(args.ball_confidence),
        "ball_class_id": int(args.ball_class_id),
        "enable_rfdetr_optimize": bool(args.enable_rfdetr_optimize),
        "fps_assumption": float(args.fps_assumption),
        "mahalanobis_gate": float(args.mahalanobis_gate),
        "interpolated_box_size": float(args.interpolated_box_size),
        "annotations_path": str(annotations_path) if annotations_path is not None else "",
        "match_iou": float(args.match_iou),
        "eval_score_threshold": float(args.eval_score_threshold),
        "ball_category_id": int(args.ball_category_id),
        "skip_existing": bool(args.skip_existing),
    }

    rows: List[Dict[str, object]] = []
    started_at = time.perf_counter()
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        _init_worker(cache, base_config)
        for index, combo in enumerate(combos, start=1):
            row = run_combo(combo)
            rows.append(row)
            write_csv_rows(summary_path, rows)
            write_csv_rows(best_path, sorted_results(rows))
            print(
                f"[INFO] ({index}/{len(combos)}) {row['run_name']}: "
                f"f1={float(row['final_event_f1']):.4f}, "
                f"missed={row['final_missed_detection_count']}, "
                f"fp={row['final_false_positive_count']}, "
                f"no_gt={row['final_no_gt_predicted']}"
            )
    else:
        worker_count = min(jobs, len(combos), max(1, os.cpu_count() or 1))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(cache, base_config),
        ) as executor:
            futures = {executor.submit(run_combo, combo): combo for combo in combos}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                write_csv_rows(summary_path, rows)
                write_csv_rows(best_path, sorted_results(rows))
                print(
                    f"[INFO] ({index}/{len(combos)}) {row['run_name']}: "
                    f"f1={float(row['final_event_f1']):.4f}, "
                    f"missed={row['final_missed_detection_count']}, "
                    f"fp={row['final_false_positive_count']}, "
                    f"no_gt={row['final_no_gt_predicted']}"
                )

    ranked = sorted_results(rows)
    write_csv_rows(summary_path, rows)
    write_csv_rows(best_path, ranked)
    elapsed_s = time.perf_counter() - started_at

    summary_payload = {
        "sweep_name": args.sweep_name,
        "sweep_dir": str(sweep_dir),
        "planned_combinations": len(combos),
        "completed_combinations": len(rows),
        "elapsed_s": elapsed_s,
        "history_frames_active": False,
        "best": ranked[0] if ranked else None,
        "results_csv": str(summary_path),
        "ranked_results_csv": str(best_path),
        "detection_cache_path": str(cache_path),
    }
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if ranked:
        best = ranked[0]
        print(
            "[INFO] Best combo: "
            f"history={best['history_frames']} (inactive), "
            f"max_gap={best['max_gap_frames']}, "
            f"outlier_confirm={best['outlier_confirm_frames']}, "
            f"f1={float(best['final_event_f1']):.4f}, "
            f"missed={best['final_missed_detection_count']}, "
            f"fp={best['final_false_positive_count']}, "
            f"no_gt={best['final_no_gt_predicted']}"
        )
    print(f"[INFO] Ranked results written to {best_path}")


if __name__ == "__main__":
    main()
