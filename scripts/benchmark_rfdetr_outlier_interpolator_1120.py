#!/usr/bin/env python3
"""
Benchmark the standalone RF-DETR + outlier rejection + interpolation tracker on a dataset.

This mirrors the behavior of /home/lakshay/lx/ball_outlier_interpolator.py as closely
as practical for image-sequence benchmarking:
- RF-DETR Medium primary detections at 1120 resolution
- Kalman-gated candidate selection
- outlier rejection using recent accepted positions / velocities
- interpolated outputs on missing detections
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ball_detection_metrics import (
    BALL_CATEGORY_ID,
    build_detection_record,
    evaluate_detections,
    write_benchmark_metrics_csv,
    write_detection_export,
)
from benchmark_dataset import BenchmarkSequence, default_annotation_path
from benchmark_rfdetr_outlier_gdino_fallback_tracking import (
    DEFAULT_BALL_CLASS_ID,
    DEFAULT_ENABLE_RFDETR_OPTIMIZE,
    DEFAULT_EVAL_SCORE_THRESHOLD,
    DEFAULT_MATCH_IOU,
    DEFAULT_MAX_FRAMES_PER_SEQ,
    DEFAULT_SEQ_END,
    DEFAULT_SEQ_LIST,
    DEFAULT_SEQ_START,
    DetectionCandidate,
    OutlierDetector,
    RFDetrBallDetector,
    center_wh_to_xyxy,
    clip_xyxy,
    csv_write_dicts,
    ensure_dir,
    mean_or_zero,
    resolve_sequences,
    safe_div,
    str2bool,
    xyxy_center,
)
from distance_rule_adjustment import (
    DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
    attach_distance_rule_summary,
    resolve_center_distance_bucket_threshold_px,
    write_dual_stage_distance_rule_metrics_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = REPO_ROOT / "benchmark_sets" / "central_test_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "central_test_v1_best_models"
DEFAULT_RUN_NAME = "rfdetr_1120_outlier_interpolator__central_test_v1"

DEFAULT_BALL_MODEL_PATH = REPO_ROOT / "checkpoints" / "ball_1120.pth"
DEFAULT_BALL_MODEL_RESOLUTION = 1120
DEFAULT_BALL_CONFIDENCE = 0.8

DEFAULT_POSITION_THRESHOLD = 50.0
DEFAULT_VELOCITY_THRESHOLD = 100.0
DEFAULT_OUTLIER_HISTORY_FRAMES = 3
DEFAULT_OUTLIER_WAIT_FRAMES = 4
DEFAULT_OUTLIER_RESET_FRAMES = 3

DEFAULT_FPS_ASSUMPTION = 30.0
DEFAULT_MAX_LOST_SECONDS = 0.75
DEFAULT_STABLE_TRACK_THRESHOLD = 5
DEFAULT_VALIDATION_GATE_THRESHOLD = 25.0
DEFAULT_INTERPOLATION_VELOCITY_THRESHOLD = 15.0
DEFAULT_MAX_INTERPOLATION_FRAMES = 4
DEFAULT_INTERPOLATED_BOX_SIZE = 20.0


@dataclass
class FrameRecord:
    frame: int
    raw_candidate_count: int
    raw_best_score: float
    output_score: float
    output_source: str
    tracker_active: bool
    tracker_gap: int
    interpolation_active: bool
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


class AdaptiveKalmanFilter:
    """Constant-acceleration Kalman filter over [x, y, vx, vy, ax, ay]."""

    def __init__(self, dt: float = 1.0) -> None:
        self.dt = dt
        dt2 = 0.5 * dt**2
        self.A = np.array(
            [
                [1, 0, dt, 0, dt2, 0],
                [0, 1, 0, dt, 0, dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        self.Q = np.eye(6, dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * 5.0
        self.x_hat = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0
        self.set_process_noise(10.0)

    def set_process_noise(self, accel_noise: float) -> None:
        self.Q[4, 4] = float(accel_noise)
        self.Q[5, 5] = float(accel_noise)

    def predict(self) -> np.ndarray:
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_hat[:2].reshape(-1).astype(np.float32)

    def update(self, measurement: np.ndarray) -> None:
        measurement = np.asarray(measurement, dtype=np.float32).reshape(2, 1)
        residual = measurement - self.H @ self.x_hat
        innovation = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation)
        self.x_hat = self.x_hat + kalman_gain @ residual
        self.P = (np.eye(6, dtype=np.float32) - kalman_gain @ self.H) @ self.P

    def initialize_state(self, measurement: np.ndarray) -> None:
        measurement = np.asarray(measurement, dtype=np.float32).reshape(2, 1)
        self.x_hat.fill(0.0)
        self.x_hat[:2] = measurement
        self.P = np.eye(6, dtype=np.float32) * 100.0


class InterpolationTracker:
    def __init__(self, velocity_threshold: float, max_gap_frames: int, dt: float) -> None:
        self.velocity_threshold = velocity_threshold
        self.max_gap_frames = max_gap_frames
        self.interpolation_kf = AdaptiveKalmanFilter(dt=dt)
        self.accepted_positions: List[np.ndarray] = []
        self.interpolation_active = False
        self.interpolation_frames_remaining = 0
        self.last_accepted_position: Optional[np.ndarray] = None

    def add_accepted_prediction(self, position: np.ndarray) -> None:
        self.accepted_positions.append(position.copy())
        self.accepted_positions = self.accepted_positions[-10:]
        self.last_accepted_position = position.copy()
        self.interpolation_active = False
        self.interpolation_frames_remaining = 0

    def should_interpolate(self, current_velocity: Optional[np.ndarray]) -> bool:
        if len(self.accepted_positions) < 2:
            return False
        if current_velocity is not None and np.linalg.norm(current_velocity) > self.velocity_threshold:
            return False
        return True

    def start_interpolation(self) -> bool:
        if len(self.accepted_positions) < 2 or self.last_accepted_position is None:
            return False
        self.interpolation_kf.initialize_state(self.last_accepted_position)
        self.interpolation_active = True
        self.interpolation_frames_remaining = self.max_gap_frames
        return True

    def get_interpolated_position(self) -> Optional[np.ndarray]:
        if not self.interpolation_active or self.interpolation_frames_remaining <= 0:
            return None
        predicted_position = self.interpolation_kf.predict()
        self.interpolation_frames_remaining -= 1
        if self.interpolation_frames_remaining <= 0:
            self.interpolation_active = False
        return predicted_position

    def stop_interpolation(self) -> None:
        self.interpolation_active = False
        self.interpolation_frames_remaining = 0


class OutlierInterpolationTracker:
    def __init__(self, args: argparse.Namespace) -> None:
        dt = 1.0 / max(1e-6, float(args.fps_assumption))
        self.kf = AdaptiveKalmanFilter(dt=dt)
        self.outlier_detector = OutlierDetector(
            position_threshold=args.position_threshold,
            velocity_threshold=args.velocity_threshold,
            max_frames=args.outlier_history_frames,
            outlier_wait_frames=args.outlier_wait_frames,
            max_suspension_frames=args.outlier_reset_frames,
        )
        self.interpolation_tracker = InterpolationTracker(
            velocity_threshold=args.interpolation_velocity_threshold,
            max_gap_frames=args.max_interpolation_frames,
            dt=dt,
        )
        self.track_initialized = False
        self.track_lost_count = 0
        self.track_hit_streak = 0
        self.prev_position: Optional[np.ndarray] = None
        self.last_accepted_score = 0.0
        self.max_lost_frames = max(1, int(round(float(args.fps_assumption) * float(args.max_lost_seconds))))
        self.stable_track_threshold = int(args.stable_track_threshold)
        self.validation_gate_threshold = float(args.validation_gate_threshold)
        self.interpolated_box_size = float(args.interpolated_box_size)

    def predict_position(self) -> Optional[np.ndarray]:
        if not self.track_initialized:
            return None
        return self.kf.predict()

    def select_best_candidate(
        self,
        candidates: Sequence[DetectionCandidate],
        predicted_position: Optional[np.ndarray],
    ) -> Optional[DetectionCandidate]:
        if not candidates:
            return None
        if not self.track_initialized or predicted_position is None:
            return max(candidates, key=lambda item: item.score)

        centers = np.stack([xyxy_center(item.xyxy) for item in candidates], axis=0)
        distances = np.linalg.norm(centers - predicted_position.reshape(1, 2), axis=1)
        valid_indices = np.where(distances < self.validation_gate_threshold)[0]
        if len(valid_indices) > 0:
            best_index = int(valid_indices[np.argmin(distances[valid_indices])])
            return candidates[best_index]
        return max(candidates, key=lambda item: item.score)

    def process_detection(
        self,
        best_detection: Optional[DetectionCandidate],
    ) -> Tuple[Optional[DetectionCandidate], bool]:
        if best_detection is None:
            return None, False

        current_position = xyxy_center(best_detection.xyxy)
        current_velocity = current_position - self.prev_position if self.prev_position is not None else None
        _is_outlier, should_use_detection = self.outlier_detector.is_outlier(current_position, current_velocity)

        if should_use_detection:
            self.track_hit_streak += 1
            self.track_lost_count = 0
            self.kf.set_process_noise(1.0 if self.track_hit_streak > self.stable_track_threshold else 10.0)
            if not self.track_initialized:
                self.kf.initialize_state(current_position)
                self.track_initialized = True
            else:
                self.kf.update(current_position)
            self.interpolation_tracker.add_accepted_prediction(current_position)
            self.outlier_detector.add_frame(current_position, current_velocity)
            self.prev_position = current_position.copy()
            self.last_accepted_score = float(best_detection.score)
            return best_detection, False

        self.track_lost_count += 1
        self.track_hit_streak = 0
        self.kf.set_process_noise(10.0)
        if self.outlier_detector.should_reset_tracking():
            self.track_initialized = False
            self.track_lost_count = 0
            self.track_hit_streak = 0
            self.interpolation_tracker.stop_interpolation()
        if self.track_lost_count >= self.max_lost_frames and self.track_initialized:
            self.track_initialized = False
            self.track_hit_streak = 0
            self.track_lost_count = 0
            self.interpolation_tracker.stop_interpolation()
        return None, False

    def process_miss(self) -> Optional[DetectionCandidate]:
        if not self.track_initialized:
            return None

        current_velocity = None
        if self.prev_position is not None and self.interpolation_tracker.accepted_positions:
            last_accepted = self.interpolation_tracker.accepted_positions[-1]
            current_velocity = last_accepted - self.prev_position

        interpolated_candidate: Optional[DetectionCandidate] = None
        if self.interpolation_tracker.should_interpolate(current_velocity):
            if not self.interpolation_tracker.interpolation_active:
                self.interpolation_tracker.start_interpolation()
            interpolated_position = self.interpolation_tracker.get_interpolated_position()
            if interpolated_position is not None:
                box_wh = np.array(
                    [self.interpolated_box_size, self.interpolated_box_size],
                    dtype=np.float32,
                )
                interpolated_candidate = DetectionCandidate(
                    xyxy=center_wh_to_xyxy(interpolated_position, box_wh),
                    score=float(self.last_accepted_score),
                    phrase="ball",
                    source="interpolation",
                )
        else:
            self.interpolation_tracker.stop_interpolation()

        self.track_lost_count += 1
        self.track_hit_streak = 0
        self.kf.set_process_noise(10.0)
        if self.track_lost_count >= self.max_lost_frames and self.track_initialized:
            self.track_initialized = False
            self.track_hit_streak = 0
            self.track_lost_count = 0
            self.interpolation_tracker.stop_interpolation()

        return interpolated_candidate


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


def process_sequence(
    sequence: BenchmarkSequence,
    detector: RFDetrBallDetector,
    args: argparse.Namespace,
    run_dir: Path,
) -> Tuple[SequenceSummary, List[dict], List[int]]:
    if not sequence.image_paths:
        raise FileNotFoundError(f"No frames found for {sequence.name}")

    seq_output_dir = ensure_dir(run_dir / sequence.name)
    frame_trace_path = seq_output_dir / "frame_trace.csv"
    tracker = OutlierInterpolationTracker(args=args)

    frame_records: List[FrameRecord] = []
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
        raw_candidates = detector.predict_ball_candidates(image)
        primary_ms = (time.perf_counter() - primary_start) * 1000.0
        raw_best = raw_candidates[0] if raw_candidates else None
        if raw_best is not None:
            raw_pred_frames += 1
            detection_records.append(
                build_detection_record(
                    sequence=sequence.name,
                    frame_index=frame_id,
                    original_frame=sequence.original_frames_by_frame[frame_id],
                    file_name=sequence.file_names_by_frame[frame_id],
                    image_id=image_id,
                    bbox_xyxy=raw_best.xyxy,
                    score=raw_best.score,
                    stage="raw",
                    source=raw_best.source,
                )
            )

        tracking_start = time.perf_counter()
        predicted_position = tracker.predict_position()
        best_detection = tracker.select_best_candidate(raw_candidates, predicted_position)
        accepted_candidate, _used_interpolation = tracker.process_detection(best_detection)
        output_candidate = accepted_candidate
        if best_detection is None and output_candidate is None:
            output_candidate = tracker.process_miss()
        tracking_ms = (time.perf_counter() - tracking_start) * 1000.0

        output_score = 0.0
        output_source = "none"
        interpolated_output = False
        if output_candidate is not None:
            output_box = clip_xyxy(
                output_candidate.xyxy,
                width=image.shape[1],
                height=image.shape[0],
            )
            output_score = float(output_candidate.score)
            output_source = str(output_candidate.source)
            interpolated_output = output_source == "interpolation"
            final_pred_frames += 1
            if interpolated_output:
                interpolated_frames += 1
            detection_records.append(
                build_detection_record(
                    sequence=sequence.name,
                    frame_index=frame_id,
                    original_frame=sequence.original_frames_by_frame[frame_id],
                    file_name=sequence.file_names_by_frame[frame_id],
                    image_id=image_id,
                    bbox_xyxy=output_box,
                    score=output_score,
                    stage="final",
                    source=output_source,
                )
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        frame_records.append(
            FrameRecord(
                frame=frame_id,
                raw_candidate_count=len(raw_candidates),
                raw_best_score=raw_best.score if raw_best is not None else 0.0,
                output_score=output_score,
                output_source=output_source,
                tracker_active=tracker.track_initialized,
                tracker_gap=tracker.track_lost_count,
                interpolation_active=tracker.interpolation_tracker.interpolation_active,
                interpolated_output=interpolated_output,
                primary_ms=primary_ms,
                tracking_ms=tracking_ms,
                total_ms=total_ms,
            )
        )

    csv_write_dicts(frame_trace_path, [asdict(record) for record in frame_records])
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
    return summary, detection_records, processed_image_ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the RF-DETR + outlier interpolation tracker on a dataset."
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
    parser.add_argument("--max_lost_seconds", type=float, default=DEFAULT_MAX_LOST_SECONDS)
    parser.add_argument("--stable_track_threshold", type=int, default=DEFAULT_STABLE_TRACK_THRESHOLD)
    parser.add_argument("--validation_gate_threshold", type=float, default=DEFAULT_VALIDATION_GATE_THRESHOLD)

    parser.add_argument("--position_threshold", type=float, default=DEFAULT_POSITION_THRESHOLD)
    parser.add_argument("--velocity_threshold", type=float, default=DEFAULT_VELOCITY_THRESHOLD)
    parser.add_argument("--outlier_history_frames", type=int, default=DEFAULT_OUTLIER_HISTORY_FRAMES)
    parser.add_argument("--outlier_wait_frames", type=int, default=DEFAULT_OUTLIER_WAIT_FRAMES)
    parser.add_argument("--outlier_reset_frames", type=int, default=DEFAULT_OUTLIER_RESET_FRAMES)

    parser.add_argument(
        "--interpolation_velocity_threshold",
        type=float,
        default=DEFAULT_INTERPOLATION_VELOCITY_THRESHOLD,
    )
    parser.add_argument("--max_interpolation_frames", type=int, default=DEFAULT_MAX_INTERPOLATION_FRAMES)
    parser.add_argument("--interpolated_box_size", type=float, default=DEFAULT_INTERPOLATED_BOX_SIZE)

    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--match_iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--eval_score_threshold", type=float, default=DEFAULT_EVAL_SCORE_THRESHOLD)
    parser.add_argument("--ball_category_id", type=int, default=BALL_CATEGORY_ID)
    return parser


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
        f"[INFO] Interpolation: velocity_threshold={args.interpolation_velocity_threshold}, "
        f"max_frames={args.max_interpolation_frames}, box_size={args.interpolated_box_size}"
    )
    if annotations_path is not None:
        print(f"[INFO] Evaluation annotations: {annotations_path}")

    detector = RFDetrBallDetector(
        ball_model_path=args.ball_model_path,
        ball_model_resolution=int(args.ball_model_resolution) if int(args.ball_model_resolution) > 0 else None,
        ball_confidence=args.ball_confidence,
        ball_class_id=args.ball_class_id,
        optimize_for_inference=args.enable_rfdetr_optimize,
    )

    summaries: List[SequenceSummary] = []
    all_detections: List[dict] = []
    processed_image_ids: List[int] = []
    for sequence in sequences:
        print(f"[INFO] Processing {sequence.name}")
        summary, sequence_detections, sequence_image_ids = process_sequence(
            sequence=sequence,
            detector=detector,
            args=args,
            run_dir=run_dir,
        )
        summaries.append(summary)
        all_detections.extend(sequence_detections)
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
        "evaluation": evaluation_summary,
        "aggregate": aggregate,
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
