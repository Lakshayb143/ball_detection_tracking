#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ball_detection_metrics import (  # noqa: E402
    BALL_CATEGORY_ID,
    build_detection_record,
    evaluate_detections,
    write_benchmark_metrics_csv,
    write_detection_export,
)
from benchmark_dataset import BenchmarkSequence, default_annotation_path, resolve_sequences  # noqa: E402
from benchmark_rfdetr_outlier_gdino_fallback_tracking import (  # noqa: E402
    center_wh_to_xyxy,
    clip_xyxy,
    csv_write_dicts,
    ensure_dir,
    mean_or_zero,
    safe_div,
    str2bool,
)
from distance_rule_adjustment import (  # noqa: E402
    DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
    attach_distance_rule_summary,
    resolve_center_distance_bucket_threshold_px,
    write_dual_stage_distance_rule_metrics_csv,
)

import ball_outlier_interpolator_v2 as v2  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "train"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "clip1_fresh_runs"
DEFAULT_RUN_NAME = "ball_outlier_interpolator_v2__clip1"
DEFAULT_SEQ_START = 0
DEFAULT_SEQ_END = 0
DEFAULT_SEQ_LIST = ""
DEFAULT_MAX_FRAMES_PER_SEQ = 0

DEFAULT_BALL_MODEL_PATH = (REPO_ROOT / v2.MODEL_PATH).resolve()
DEFAULT_BALL_MODEL_RESOLUTION = int(v2.MODEL_RESOLUTION)
DEFAULT_BALL_CONFIDENCE = float(v2.CONFIDENCE)
DEFAULT_BALL_CLASS_ID = int(v2.BALL_CLASS_ID)
DEFAULT_ENABLE_RFDETR_OPTIMIZE = bool(v2.ENABLE_RFDETR_OPTIMIZE)

DEFAULT_FPS_ASSUMPTION = float(v2.DEFAULT_FPS_IF_MISSING)
DEFAULT_INTERPOLATED_BOX_SIZE = float(v2.BENCHMARK_INTERPOLATED_BOX_SIZE)
DEFAULT_MATCH_IOU = float(v2.BENCHMARK_MATCH_IOU)
DEFAULT_EVAL_SCORE_THRESHOLD = float(v2.BENCHMARK_EVAL_SCORE_THRESHOLD)
DEFAULT_BALL_CATEGORY_ID = int(v2.BENCHMARK_BALL_CATEGORY_ID)


@dataclass
class FrameRecord:
    frame: int
    raw_candidate_count: int
    raw_best_score: float
    selected_score: float
    selected_in_gate: bool
    output_score: float
    output_source: str
    tracker_active: bool
    gap_frames: int
    hit_streak: int
    interpolated_output: bool
    primary_ms: float
    tracking_ms: float
    total_ms: float


@dataclass
class SequenceSummary:
    sequence: str
    frames_total: int
    raw_pred_frames: int
    final_pred_frames: int
    interpolated_frames: int
    raw_eval_detections: int
    raw_tp: int
    raw_fp: int
    raw_fn: int
    raw_precision: float
    raw_recall: float
    raw_mean_iou: float
    final_eval_detections: int
    final_tp: int
    final_fp: int
    final_fn: int
    final_precision: float
    final_recall: float
    final_mean_iou: float
    primary_ms_avg: float
    tracking_ms_avg: float
    total_ms_avg: float
    runtime_fps: float


class V2TrackerAdapter:
    """Keeps v2 tracking behavior intact while letting us benchmark image folders."""

    def __init__(
        self,
        fps_assumption: float,
        confidence_threshold: float,
        mahalanobis_gate: float,
        max_gap_frames: int,
        outlier_confirm_frames: int,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.mahalanobis_gate = float(mahalanobis_gate)
        self.max_gap_frames = int(max_gap_frames)
        self.kf = v2.BallKalmanFilter(dt=1.0 / max(1e-6, float(fps_assumption)))
        self.outlier_confirmer = v2.OutlierConfirmer(max_consecutive=int(outlier_confirm_frames))

        self.track_initialized = False
        self.frames_since_detection = 0
        self.hit_streak = 0

    def select_best_detection(self, ball_detections: sv.Detections) -> Optional[dict]:
        if len(ball_detections.xyxy) == 0:
            return None

        conf_mask = ball_detections.confidence >= self.confidence_threshold
        if not np.any(conf_mask):
            return None
        filtered = ball_detections[conf_mask]
        centers = filtered.get_anchors_coordinates(sv.Position.CENTER)

        if not self.track_initialized:
            idx = int(np.argmax(filtered.confidence))
            return {"center": centers[idx], "sv": filtered[idx : idx + 1], "in_gate": False}

        m_dists = np.array([self.kf.mahalanobis(center) for center in centers], dtype=np.float32)
        in_gate_mask = m_dists < self.mahalanobis_gate

        if np.any(in_gate_mask):
            in_gate_indices = np.where(in_gate_mask)[0]
            best_in_gate = int(in_gate_indices[np.argmin(m_dists[in_gate_indices])])
            return {
                "center": centers[best_in_gate],
                "sv": filtered[best_in_gate : best_in_gate + 1],
                "in_gate": True,
            }

        idx = int(np.argmax(filtered.confidence))
        return {"center": centers[idx], "sv": filtered[idx : idx + 1], "in_gate": False}

    def step(
        self,
        best_detection: Optional[dict],
        frame_count: int,
    ) -> Tuple[Optional[np.ndarray], bool, Optional[sv.Detections]]:
        if self.track_initialized:
            predicted_pos, _ = self.kf.predict()
        else:
            predicted_pos = None

        if best_detection is None:
            if not self.track_initialized:
                return None, False, None
            self.frames_since_detection += 1
            self.hit_streak = 0
            if self.frames_since_detection > self.max_gap_frames:
                self.track_initialized = False
                self.outlier_confirmer.reset()
                print(
                    f"Frame {frame_count}: track lost - gap exceeded {self.max_gap_frames} frames"
                )
                return None, False, None
            print(
                f"Frame {frame_count}: INTERPOLATED at {predicted_pos.round(1)} "
                f"(gap={self.frames_since_detection})"
            )
            return predicted_pos, True, None

        center = best_detection["center"]
        in_gate = bool(best_detection["in_gate"])

        if not self.track_initialized:
            self.kf.initialize(center)
            self.track_initialized = True
            self.frames_since_detection = 0
            self.hit_streak = 1
            self.outlier_confirmer.reset()
            print(f"Frame {frame_count}: ACCEPTED (init) at {center.round(1)}")
            return center, False, best_detection["sv"]

        if in_gate:
            self.kf.update(center)
            self.frames_since_detection = 0
            self.hit_streak += 1
            self.outlier_confirmer.reset()
            print(f"Frame {frame_count}: ACCEPTED at {center.round(1)} m_dist passed gate")
            return self.kf.position(), False, best_detection["sv"]

        should_reset = self.outlier_confirmer.report(was_outlier=True)
        self.frames_since_detection += 1
        self.hit_streak = 0

        if should_reset:
            self.kf.initialize(center)
            self.track_initialized = True
            self.frames_since_detection = 0
            self.hit_streak = 1
            self.outlier_confirmer.reset()
            print(
                "Frame "
                f"{frame_count}: RESET - re-initializing on outlier-streak detection at "
                f"{center.round(1)}"
            )
            return center, False, best_detection["sv"]

        if self.frames_since_detection > self.max_gap_frames:
            self.track_initialized = False
            self.outlier_confirmer.reset()
            print(
                "Frame "
                f"{frame_count}: track lost - gap exceeded {self.max_gap_frames} frames "
                "during outlier streak"
            )
            return None, False, None

        print(
            "Frame "
            f"{frame_count}: REJECTED detection (outside gate) at {center.round(1)}; interpolating"
        )
        return predicted_pos, True, None


def aggregate_sequence_summaries(summaries: Sequence[SequenceSummary]) -> Dict[str, float]:
    if not summaries:
        return {}

    totals = {
        "frames_total": 0.0,
        "raw_pred_frames": 0.0,
        "final_pred_frames": 0.0,
        "interpolated_frames": 0.0,
        "raw_eval_detections": 0.0,
        "raw_tp": 0.0,
        "raw_fp": 0.0,
        "raw_fn": 0.0,
        "final_eval_detections": 0.0,
        "final_tp": 0.0,
        "final_fp": 0.0,
        "final_fn": 0.0,
    }

    for summary in summaries:
        for key in totals:
            totals[key] += float(getattr(summary, key))

    totals["raw_precision"] = safe_div(totals["raw_tp"], totals["raw_tp"] + totals["raw_fp"])
    totals["raw_recall"] = safe_div(totals["raw_tp"], totals["raw_tp"] + totals["raw_fn"])
    totals["raw_mean_iou"] = safe_div(
        sum(item.raw_mean_iou * item.raw_tp for item in summaries),
        totals["raw_tp"],
    )
    totals["final_precision"] = safe_div(totals["final_tp"], totals["final_tp"] + totals["final_fp"])
    totals["final_recall"] = safe_div(totals["final_tp"], totals["final_tp"] + totals["final_fn"])
    totals["final_mean_iou"] = safe_div(
        sum(item.final_mean_iou * item.final_tp for item in summaries),
        totals["final_tp"],
    )
    totals["primary_ms_avg"] = mean_or_zero([item.primary_ms_avg for item in summaries])
    totals["tracking_ms_avg"] = mean_or_zero([item.tracking_ms_avg for item in summaries])
    totals["total_ms_avg"] = mean_or_zero([item.total_ms_avg for item in summaries])
    totals["runtime_fps"] = mean_or_zero([item.runtime_fps for item in summaries])
    totals["sequence_count"] = float(len(summaries))
    return totals


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ball_outlier_interpolator_v2.py on clip1-style frame folders while "
            "keeping its tracker logic unchanged."
        )
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--seq_start", type=int, default=DEFAULT_SEQ_START)
    parser.add_argument("--seq_end", type=int, default=DEFAULT_SEQ_END)
    parser.add_argument("--seq_list", type=str, default=DEFAULT_SEQ_LIST)
    parser.add_argument("--max_frames_per_seq", type=int, default=DEFAULT_MAX_FRAMES_PER_SEQ)

    parser.add_argument("--ball_model_path", type=Path, default=DEFAULT_BALL_MODEL_PATH)
    parser.add_argument("--ball_model_resolution", type=int, default=DEFAULT_BALL_MODEL_RESOLUTION)
    parser.add_argument("--ball_confidence", type=float, default=DEFAULT_BALL_CONFIDENCE)
    parser.add_argument("--ball_class_id", type=int, default=DEFAULT_BALL_CLASS_ID)
    parser.add_argument("--enable_rfdetr_optimize", type=str2bool, default=DEFAULT_ENABLE_RFDETR_OPTIMIZE)

    parser.add_argument("--fps_assumption", type=float, default=DEFAULT_FPS_ASSUMPTION)
    parser.add_argument("--mahalanobis_gate", type=float, default=float(v2.MAHALANOBIS_GATE))
    parser.add_argument("--max_gap_frames", type=int, default=int(v2.MAX_GAP_FRAMES))
    parser.add_argument("--outlier_confirm_frames", type=int, default=int(v2.OUTLIER_CONFIRM_FRAMES))
    parser.add_argument("--interpolated_box_size", type=float, default=DEFAULT_INTERPOLATED_BOX_SIZE)

    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--match_iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--eval_score_threshold", type=float, default=DEFAULT_EVAL_SCORE_THRESHOLD)
    parser.add_argument("--ball_category_id", type=int, default=DEFAULT_BALL_CATEGORY_ID)
    return parser


def _top_confidence_detection(ball_detections: sv.Detections) -> Optional[sv.Detections]:
    if len(ball_detections.xyxy) == 0:
        return None
    idx = int(np.argmax(ball_detections.confidence))
    return ball_detections[idx : idx + 1]


def _record_point_output(
    frame_idx: int,
    output_position: Optional[np.ndarray],
    accepted_detection: Optional[sv.Detections],
    is_interpolated: bool,
) -> dict:
    if accepted_detection is not None and len(accepted_detection.xyxy) > 0:
        center = accepted_detection.get_anchors_coordinates(sv.Position.CENTER)[0]
        return {
            "frame_idx": int(frame_idx),
            "x": float(center[0]),
            "y": float(center[1]),
            "confidence": float(accepted_detection.confidence[0]),
            "source": "accepted_detection",
            "interpolated": False,
        }
    if output_position is not None and is_interpolated:
        return {
            "frame_idx": int(frame_idx),
            "x": float(output_position[0]),
            "y": float(output_position[1]),
            "confidence": None,
            "source": "interpolation",
            "interpolated": True,
        }
    return {
        "frame_idx": int(frame_idx),
        "x": None,
        "y": None,
        "confidence": None,
        "source": "none",
        "interpolated": False,
    }


def process_sequence(
    sequence: BenchmarkSequence,
    model: RFDETRMedium,
    args: argparse.Namespace,
    run_dir: Path,
) -> Tuple[SequenceSummary, List[dict], List[dict], List[int]]:
    if not sequence.image_paths:
        raise FileNotFoundError(f"No frames found for {sequence.name}")

    seq_output_dir = ensure_dir(run_dir / sequence.name)
    frame_trace_path = seq_output_dir / "frame_trace.csv"
    point_output_path = seq_output_dir / "point_outputs.json"
    tracker = V2TrackerAdapter(
        fps_assumption=args.fps_assumption,
        confidence_threshold=args.ball_confidence,
        mahalanobis_gate=args.mahalanobis_gate,
        max_gap_frames=args.max_gap_frames,
        outlier_confirm_frames=args.outlier_confirm_frames,
    )

    frame_records: List[FrameRecord] = []
    point_records: List[dict] = []
    detection_records: List[dict] = []
    processed_image_ids: List[int] = []
    raw_pred_frames = 0
    final_pred_frames = 0
    interpolated_frames = 0

    for frame_id, image_path in enumerate(sequence.image_paths, start=1):
        total_start = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read frame {image_path}")

        image_id = sequence.image_ids_by_frame.get(frame_id)
        if image_id is not None:
            processed_image_ids.append(int(image_id))

        primary_start = time.perf_counter()
        detections = model.predict(image, confidence=args.ball_confidence)
        primary_ms = (time.perf_counter() - primary_start) * 1000.0
        ball_detections = detections[detections.class_id == args.ball_class_id]
        raw_best_detection = _top_confidence_detection(ball_detections)
        if raw_best_detection is not None:
            raw_pred_frames += 1
            raw_box = clip_xyxy(
                np.asarray(raw_best_detection.xyxy[0], dtype=np.float32),
                width=image.shape[1],
                height=image.shape[0],
            )
            detection_records.append(
                build_detection_record(
                    sequence=sequence.name,
                    frame_index=frame_id,
                    original_frame=sequence.original_frames_by_frame[frame_id],
                    file_name=sequence.file_names_by_frame[frame_id],
                    image_id=image_id,
                    bbox_xyxy=raw_box,
                    score=float(raw_best_detection.confidence[0]),
                    stage="raw",
                    source="rfdetr",
                )
            )

        tracking_start = time.perf_counter()
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
                width=image.shape[1],
                height=image.shape[0],
            )
            output_score = float(accepted_detection.confidence[0])
            output_source = "rfdetr"
            final_pred_frames += 1
            detection_records.append(
                build_detection_record(
                    sequence=sequence.name,
                    frame_index=frame_id,
                    original_frame=sequence.original_frames_by_frame[frame_id],
                    file_name=sequence.file_names_by_frame[frame_id],
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
                    np.array([args.interpolated_box_size, args.interpolated_box_size], dtype=np.float32),
                ),
                width=image.shape[1],
                height=image.shape[0],
            )
            output_score = 0.0
            output_source = "interpolation"
            final_pred_frames += 1
            interpolated_frames += 1
            detection_records.append(
                build_detection_record(
                    sequence=sequence.name,
                    frame_index=frame_id,
                    original_frame=sequence.original_frames_by_frame[frame_id],
                    file_name=sequence.file_names_by_frame[frame_id],
                    image_id=image_id,
                    bbox_xyxy=final_box,
                    score=output_score,
                    stage="final",
                    source=output_source,
                )
            )

        point_records.append(
            _record_point_output(
                frame_idx=frame_id,
                output_position=output_position,
                accepted_detection=accepted_detection,
                is_interpolated=is_interpolated,
            )
        )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        frame_records.append(
            FrameRecord(
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

    csv_write_dicts(frame_trace_path, [asdict(record) for record in frame_records])
    point_output_path.write_text(json.dumps(point_records, indent=2), encoding="utf-8")
    summary = SequenceSummary(
        sequence=sequence.name,
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


def main() -> None:
    args = build_arg_parser().parse_args()

    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.ball_model_path = args.ball_model_path.expanduser().resolve()
    if args.annotations is not None:
        args.annotations = args.annotations.expanduser().resolve()

    run_dir = ensure_dir(args.output_root / args.run_name)
    sequences = resolve_sequences(
        data_root=args.data_root,
        seq_start=args.seq_start,
        seq_end=args.seq_end,
        seq_list=args.seq_list,
        max_frames_per_seq=args.max_frames_per_seq,
    )
    annotations_path = args.annotations or default_annotation_path(args.data_root)

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Data root: {args.data_root}")
    print(f"[INFO] Sequences: {', '.join(sequence.name for sequence in sequences)}")
    print(f"[INFO] Model path: {args.ball_model_path}")
    print(f"[INFO] Model resolution: {args.ball_model_resolution}")
    print(f"[INFO] Confidence threshold: {args.ball_confidence}")
    print(
        f"[INFO] v2 tracker config: gate={args.mahalanobis_gate}, "
        f"max_gap_frames={args.max_gap_frames}, outlier_confirm_frames={args.outlier_confirm_frames}"
    )
    print(f"[INFO] FPS assumption: {args.fps_assumption}")
    print(f"[INFO] Interpolated box size: {args.interpolated_box_size}")
    if annotations_path is not None:
        print(f"[INFO] Evaluation annotations: {annotations_path}")

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

    summaries: List[SequenceSummary] = []
    all_detections: List[dict] = []
    all_point_outputs: List[dict] = []
    processed_image_ids: List[int] = []
    for sequence in sequences:
        print(f"[INFO] Processing {sequence.name}")
        summary, sequence_detections, sequence_points, sequence_image_ids = process_sequence(
            sequence=sequence,
            model=model,
            args=args,
            run_dir=run_dir,
        )
        summaries.append(summary)
        all_detections.extend(sequence_detections)
        all_point_outputs.extend(sequence_points)
        processed_image_ids.extend(sequence_image_ids)
        print(
            f"[INFO] {sequence.name}: raw_predictions={summary.raw_pred_frames}, "
            f"final_predictions={summary.final_pred_frames}, interpolated_frames={summary.interpolated_frames}, "
            f"fps={summary.runtime_fps:.2f}"
        )

    detections_path = run_dir / "detections.json"
    write_detection_export(
        path=detections_path,
        run_name=args.run_name,
        data_root=args.data_root,
        detections=all_detections,
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
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
            iou_threshold=args.match_iou,
            score_threshold=args.eval_score_threshold,
            ball_category_id=args.ball_category_id,
            allowed_image_ids=processed_image_ids,
        )
        final_metrics = evaluate_detections(
            detections=all_detections,
            annotations_path=annotations_path,
            stage="final",
            iou_threshold=args.match_iou,
            score_threshold=args.eval_score_threshold,
            ball_category_id=args.ball_category_id,
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
            "ball_category_id": args.ball_category_id,
            "iou_threshold": args.match_iou,
            "score_threshold": args.eval_score_threshold,
            "raw": raw_metrics,
            "final": final_metrics,
        }

    csv_write_dicts(run_dir / "sequence_summary.csv", [asdict(summary) for summary in summaries])
    aggregate = aggregate_sequence_summaries(summaries)
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
                "raw_false_positive_center_distance_by_frame_px": raw_metric_aggregate.get(
                    "false_positive_center_distance_by_frame_px", {}
                ),
                "final_event_tp": final_metric_aggregate.get("event_tp", 0.0),
                "final_missed_detection_count": final_metric_aggregate.get("missed_detection_count", 0.0),
                "final_false_positive_count": final_metric_aggregate.get("false_positive_count", 0.0),
                "final_tn": final_metric_aggregate.get("tn", 0.0),
                "final_no_gt_predicted": final_metric_aggregate.get("no_gt_predicted", 0.0),
                "final_avg_false_positive_center_distance_px": final_metric_aggregate.get(
                    "avg_false_positive_center_distance_px", 0.0
                ),
                "final_false_positive_center_distance_by_frame_px": final_metric_aggregate.get(
                    "false_positive_center_distance_by_frame_px", {}
                ),
            }
        )

    original_metrics_path = run_dir / "benchmark_metrics_original.csv"
    write_benchmark_metrics_csv(
        original_metrics_path,
        latency_ms=aggregate.get("total_ms_avg", 0.0),
        iou_threshold=float(args.match_iou),
        score_threshold=float(args.eval_score_threshold),
        raw_tp=int(raw_metric_aggregate.get("event_tp", 0.0)),
        raw_missed_detection_count=int(raw_metric_aggregate.get("missed_detection_count", 0.0)),
        raw_false_positive_count=int(raw_metric_aggregate.get("false_positive_count", 0.0)),
        raw_tn=int(raw_metric_aggregate.get("tn", 0.0)),
        raw_no_gt_predicted=int(raw_metric_aggregate.get("no_gt_predicted", 0.0)),
        raw_avg_false_positive_center_distance_px=float(
            raw_metric_aggregate.get("avg_false_positive_center_distance_px", 0.0)
        ),
        final_tp=int(final_metric_aggregate.get("event_tp", 0.0)),
        final_missed_detection_count=int(final_metric_aggregate.get("missed_detection_count", 0.0)),
        final_false_positive_count=int(final_metric_aggregate.get("false_positive_count", 0.0)),
        final_tn=int(final_metric_aggregate.get("tn", 0.0)),
        final_no_gt_predicted=int(final_metric_aggregate.get("no_gt_predicted", 0.0)),
        final_avg_false_positive_center_distance_px=float(
            final_metric_aggregate.get("avg_false_positive_center_distance_px", 0.0)
        ),
    )
    original_fresh_metrics_path = args.output_root / f"{args.run_name}_benchmark_metrics_original.csv"
    write_benchmark_metrics_csv(
        original_fresh_metrics_path,
        latency_ms=aggregate.get("total_ms_avg", 0.0),
        iou_threshold=float(args.match_iou),
        score_threshold=float(args.eval_score_threshold),
        raw_tp=int(raw_metric_aggregate.get("event_tp", 0.0)),
        raw_missed_detection_count=int(raw_metric_aggregate.get("missed_detection_count", 0.0)),
        raw_false_positive_count=int(raw_metric_aggregate.get("false_positive_count", 0.0)),
        raw_tn=int(raw_metric_aggregate.get("tn", 0.0)),
        raw_no_gt_predicted=int(raw_metric_aggregate.get("no_gt_predicted", 0.0)),
        raw_avg_false_positive_center_distance_px=float(
            raw_metric_aggregate.get("avg_false_positive_center_distance_px", 0.0)
        ),
        final_tp=int(final_metric_aggregate.get("event_tp", 0.0)),
        final_missed_detection_count=int(final_metric_aggregate.get("missed_detection_count", 0.0)),
        final_false_positive_count=int(final_metric_aggregate.get("false_positive_count", 0.0)),
        final_tn=int(final_metric_aggregate.get("tn", 0.0)),
        final_no_gt_predicted=int(final_metric_aggregate.get("no_gt_predicted", 0.0)),
        final_avg_false_positive_center_distance_px=float(
            final_metric_aggregate.get("avg_false_positive_center_distance_px", 0.0)
        ),
    )

    experiment_summary = {
        "run_name": args.run_name,
        "data_root": str(args.data_root),
        "sequences": [sequence.name for sequence in sequences],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "detections_path": str(detections_path),
        "point_outputs_path": str(run_dir / "point_outputs.json"),
        "evaluation": evaluation_summary,
        "aggregate": aggregate,
        "logic_reference": str((REPO_ROOT / "ball_outlier_interpolator_v2.py").resolve()),
        "logic_notes": {
            "core_tracking_logic": "Copied from ball_outlier_interpolator_v2.py without intended behavior changes.",
            "serialization_only_change": (
                "Interpolated point outputs are converted to fixed-size boxes for metric scoring."
            ),
            "interpolated_box_size": float(args.interpolated_box_size),
            "fps_assumption": float(args.fps_assumption),
        },
    }
    center_distance_bucket_threshold_px = resolve_center_distance_bucket_threshold_px(
        annotations_path=annotations_path if annotations_path is not None and annotations_path.exists() else None,
        ball_category_id=int(args.ball_category_id),
        fallback_threshold_px=DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
    )
    distance_rule_summary = attach_distance_rule_summary(
        experiment_summary,
        center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
        include_raw_stage=True,
    )
    adjusted_metrics_path = run_dir / "benchmark_metrics.csv"
    write_dual_stage_distance_rule_metrics_csv(
        adjusted_metrics_path,
        latency_ms=aggregate.get("total_ms_avg", 0.0),
        iou_threshold=float(args.match_iou),
        score_threshold=float(args.eval_score_threshold),
        raw_aggregate=raw_metric_aggregate,
        final_aggregate=final_metric_aggregate,
        center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
    )
    fresh_metrics_path = args.output_root / f"{args.run_name}_benchmark_metrics.csv"
    write_dual_stage_distance_rule_metrics_csv(
        fresh_metrics_path,
        latency_ms=aggregate.get("total_ms_avg", 0.0),
        iou_threshold=float(args.match_iou),
        score_threshold=float(args.eval_score_threshold),
        raw_aggregate=raw_metric_aggregate,
        final_aggregate=final_metric_aggregate,
        center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
    )
    with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_summary, handle, indent=2)

    print(f"[INFO] Detection export written to {detections_path}")
    print(f"[INFO] Point outputs written to {run_dir / 'point_outputs.json'}")
    print(f"[INFO] Sequence summary written to {run_dir / 'sequence_summary.csv'}")
    print(f"[INFO] Benchmark metrics written to {run_dir / 'benchmark_metrics.csv'}")
    print(f"[INFO] Original metrics preserved at {original_metrics_path}")
    print(f"[INFO] Fresh metrics CSV written to {fresh_metrics_path}")
    print(f"[INFO] Original fresh metrics CSV written to {original_fresh_metrics_path}")
    print(f"[INFO] Experiment summary written to {run_dir / 'experiment_summary.json'}")
    final_adjusted = distance_rule_summary.get("final", {}) if isinstance(distance_rule_summary, dict) else {}
    if final_adjusted:
        print(
            "[INFO] Final metric aggregate (distance-threshold bucket kept separate): "
            f"tp={int(final_adjusted.get('tp', 0))}, "
            f"missed={int(final_adjusted.get('missed_detection_count', 0))}, "
            f"fp={int(final_adjusted.get('false_positive_count', 0))}, "
            f"tn={int(final_adjusted.get('tn', 0))}, "
            f"no_gt_count={int(final_adjusted.get('no_gt_count', 0))}, "
            f"distance_threshold_px={float(final_adjusted.get('center_distance_bucket_threshold_px', 0.0)):.2f}, "
            f"distance_le_threshold_px_count={int(final_adjusted.get('distance_le_threshold_px_count', 0))}"
        )


if __name__ == "__main__":
    main()
