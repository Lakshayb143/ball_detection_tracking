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
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
from rfdetr import RFDETRMedium

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ball_detection_metrics import (  # noqa: E402
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

import ball_tracking_lb as lb  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "train"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "clip1_fresh_runs"
DEFAULT_RUN_NAME = "ball_tracking_lb__clip1"
DEFAULT_SEQ_START = 0
DEFAULT_SEQ_END = 0
DEFAULT_SEQ_LIST = ""
DEFAULT_MAX_FRAMES_PER_SEQ = 0

DEFAULT_BALL_MODEL_PATH = (REPO_ROOT / lb.BALL_MODEL_PATH).resolve()
DEFAULT_PLAYER_MODEL_PATH = (REPO_ROOT / lb.PLAYER_MODEL_PATH).resolve()
DEFAULT_BALL_MODEL_RESOLUTION = int(lb.BALL_MODEL_RESOLUTION)
DEFAULT_PLAYER_MODEL_RESOLUTION = int(lb.PLAYER_MODEL_RESOLUTION)
DEFAULT_BALL_CONFIDENCE = float(lb.CONFIDENCE)
DEFAULT_PLAYER_CONFIDENCE = float(lb.PLAYER_CONFIDENCE)
DEFAULT_BALL_CLASS_ID = int(lb.BALL_CLASS_ID)
DEFAULT_PLAYER_CLASS_IDS = ",".join(str(x) for x in lb.PLAYER_CLASS_IDS)
DEFAULT_ENABLE_RFDETR_OPTIMIZE = bool(lb.ENABLE_RFDETR_OPTIMIZE)
DEFAULT_FPS_ASSUMPTION = float(lb.DEFAULT_FPS_IF_MISSING)
DEFAULT_BALL_TRACKER_BUFFER_SIZE = int(lb.BALL_TRACKER_BUFFER_SIZE)
DEFAULT_POSITION_THRESHOLD = float(lb.POSITION_THRESHOLD)
DEFAULT_VELOCITY_THRESHOLD = float(lb.VELOCITY_THRESHOLD)
DEFAULT_HISTORY_FRAMES = int(lb.HISTORY_FRAMES)
DEFAULT_MAX_LOST_SECONDS = float(lb.MAX_LOST_SECONDS)
DEFAULT_STABLE_TRACK_THRESHOLD = int(lb.STABLE_TRACK_THRESHOLD)
DEFAULT_VALIDATION_GATE_THRESHOLD = float(lb.VALIDATION_GATE_THRESHOLD)
DEFAULT_INTERPOLATION_VELOCITY_THRESHOLD = float(lb.INTERPOLATION_VELOCITY_THRESHOLD)
DEFAULT_MAX_INTERPOLATION_FRAMES = int(lb.MAX_INTERPOLATION_FRAMES)
DEFAULT_MAX_OPTICAL_FLOW_GAP = int(lb.MAX_OPTICAL_FLOW_GAP)
DEFAULT_OPTICAL_FLOW_ERROR_THRESHOLD = float(lb.OPTICAL_FLOW_ERROR_THRESHOLD)
DEFAULT_OPTICAL_FLOW_MAX_MOVEMENT = float(lb.OPTICAL_FLOW_MAX_MOVEMENT)
DEFAULT_OPTICAL_FLOW_WIN_SIZE = f"{lb.OPTICAL_FLOW_WIN_SIZE[0]},{lb.OPTICAL_FLOW_WIN_SIZE[1]}"
DEFAULT_OPTICAL_FLOW_MAX_LEVEL = int(lb.OPTICAL_FLOW_MAX_LEVEL)
DEFAULT_OPTICAL_FLOW_CRITERIA_EPS = float(lb.OPTICAL_FLOW_CRITERIA_EPS)
DEFAULT_OPTICAL_FLOW_CRITERIA_COUNT = int(lb.OPTICAL_FLOW_CRITERIA_COUNT)
DEFAULT_INTERPOLATED_BOX_SIZE = float(lb.BENCHMARK_INTERPOLATED_BOX_SIZE)
DEFAULT_MATCH_IOU = float(lb.BENCHMARK_MATCH_IOU)
DEFAULT_EVAL_SCORE_THRESHOLD = float(lb.BENCHMARK_EVAL_SCORE_THRESHOLD)
DEFAULT_BALL_CATEGORY_ID = int(lb.BENCHMARK_BALL_CATEGORY_ID)


def parse_int_list(value: str) -> List[int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return [int(part) for part in parts]


def parse_pair(value: str) -> Tuple[int, int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected pair formatted like '15,15', got {value!r}")
    return int(parts[0]), int(parts[1])


class IdentityCoordTransform:
    def rel_to_abs(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float32)

    def abs_to_rel(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float32)


def ensure_coord_transform(transform) -> IdentityCoordTransform:
    return transform if transform is not None else IdentityCoordTransform()


def top_confidence_detection(detections: sv.Detections) -> Optional[sv.Detections]:
    if len(detections.xyxy) == 0:
        return None
    idx = int(np.argmax(detections.confidence))
    return detections[idx : idx + 1]


@dataclass
class FrameRecord:
    frame: int
    player_count: int
    raw_candidate_count: int
    filtered_candidate_count: int
    raw_best_score: float
    filtered_best_score: float
    selected_score: float
    output_score: float
    output_source: str
    tracker_active: bool
    optical_flow_active: bool
    track_gap: int
    optical_flow_gap: int
    hit_streak: int
    primary_ms: float
    tracking_ms: float
    total_ms: float


@dataclass
class SequenceSummary:
    sequence: str
    frames_total: int
    raw_pred_frames: int
    final_pred_frames: int
    optical_flow_frames: int
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


class LBTrackingState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.fps = max(1e-6, float(args.fps_assumption))
        self.confidence_threshold = float(args.ball_confidence)
        self.kf = lb.AdaptiveKalmanFilter(dt=1.0 / self.fps)
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        self.outlier_detector = lb.OutlierDetector(
            position_threshold=float(args.position_threshold),
            velocity_threshold=float(args.velocity_threshold),
            max_frames=int(args.history_frames),
        )
        self.interpolation_tracker = lb.InterpolationTracker(
            velocity_threshold=float(args.interpolation_velocity_threshold),
            max_gap_frames=int(args.max_interpolation_frames),
        )
        self.ball_tracker = lb.BallTracker(buffer_size=int(args.ball_tracker_buffer_size))
        self.optical_flow_kf = lb.OpticalKalmanFilter(dt=1.0 / self.fps)

        self.track_initialized = False
        self.optical_kf_track_init = False
        self.prev_position_abs: Optional[np.ndarray] = None
        self.prev_gray_frame: Optional[np.ndarray] = None
        self.optical_flow_points_rel: Optional[np.ndarray] = None
        self.track_lost_count = 0
        self.track_hit_streak = 0
        self.optical_flow_gap_counter = 0
        self.max_lost_frames = max(1, int(round(self.fps * float(args.max_lost_seconds))))
        self.validation_gate_threshold = float(args.validation_gate_threshold)
        self.stable_track_threshold = int(args.stable_track_threshold)
        self.lk_params = dict(
            winSize=tuple(args.optical_flow_win_size),
            maxLevel=int(args.optical_flow_max_level),
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(args.optical_flow_criteria_count),
                float(args.optical_flow_criteria_eps),
            ),
        )

    def get_best_detection(
        self,
        ball_detections: sv.Detections,
        predicted_pos_abs: Optional[np.ndarray],
    ) -> Optional[dict]:
        if len(ball_detections.xyxy) == 0:
            return None

        high_conf_mask = ball_detections.confidence >= self.confidence_threshold
        if not np.any(high_conf_mask):
            return None

        filtered = ball_detections[high_conf_mask]
        centers = filtered.get_anchors_coordinates(sv.Position.CENTER)
        if self.track_initialized and predicted_pos_abs is not None:
            distances = np.linalg.norm(centers - predicted_pos_abs.reshape(1, 2), axis=1)
            valid_indices = np.where(distances < self.validation_gate_threshold)[0]
            if len(valid_indices) > 0:
                idx = int(valid_indices[np.argmin(distances[valid_indices])])
            else:
                idx = int(np.argmax(filtered.confidence))
        else:
            idx = int(np.argmax(filtered.confidence))
        return {"center": centers[idx], "sv": filtered[idx : idx + 1]}

    def process_detection(
        self,
        best_detection: Optional[dict],
        frame_count: int,
    ) -> Tuple[bool, Optional[dict]]:
        if not best_detection:
            return False, None

        current_position = best_detection["center"]
        current_velocity = (
            current_position - self.prev_position_abs
            if self.prev_position_abs is not None
            else None
        )
        _is_outlier, should_use_detection = self.outlier_detector.is_outlier(
            current_position,
            current_velocity,
        )

        if should_use_detection:
            self.track_hit_streak += 1
            self.track_lost_count = 0
            self.kf.set_process_noise(1.0 if self.track_hit_streak > self.stable_track_threshold else 10.0)
            if not self.track_initialized:
                self.kf.initialize_state(current_position)
                self.track_initialized = True
            else:
                self.kf.update(current_position)

            self.interpolation_tracker.add_accepted_prediction(current_position, current_velocity)
            self.outlier_detector.add_frame(current_position, current_velocity)
            self.prev_position_abs = current_position.copy()
            print(f"Frame {frame_count}: ACCEPTED detection at {current_position}")
            return True, best_detection

        self.track_lost_count += 1
        self.track_hit_streak = 0
        self.kf.set_process_noise(10.0)
        print(f"Frame {frame_count}: OUTLIER detected, rejecting prediction at {current_position}")

        if self.outlier_detector.should_reset_tracking():
            self.track_initialized = False
            self.track_lost_count = 0
            self.track_hit_streak = 0
            self.interpolation_tracker.stop_interpolation()
            print(f"Frame {frame_count}: Resetting tracking")

        return False, None


def aggregate_sequence_summaries(summaries: Sequence[SequenceSummary]) -> Dict[str, float]:
    if not summaries:
        return {}

    totals = {
        "frames_total": 0.0,
        "raw_pred_frames": 0.0,
        "final_pred_frames": 0.0,
        "optical_flow_frames": 0.0,
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
            "Benchmark the ball_tracking_lb pipeline on clip1-style frame folders with "
            "player exclusion, outlier rejection, and optical-flow fallback."
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
    parser.add_argument("--player_model_path", type=Path, default=DEFAULT_PLAYER_MODEL_PATH)
    parser.add_argument("--ball_model_resolution", type=int, default=DEFAULT_BALL_MODEL_RESOLUTION)
    parser.add_argument("--player_model_resolution", type=int, default=DEFAULT_PLAYER_MODEL_RESOLUTION)
    parser.add_argument("--ball_confidence", type=float, default=DEFAULT_BALL_CONFIDENCE)
    parser.add_argument("--player_confidence", type=float, default=DEFAULT_PLAYER_CONFIDENCE)
    parser.add_argument("--ball_class_id", type=int, default=DEFAULT_BALL_CLASS_ID)
    parser.add_argument("--player_class_ids", type=str, default=DEFAULT_PLAYER_CLASS_IDS)
    parser.add_argument("--enable_rfdetr_optimize", type=str2bool, default=DEFAULT_ENABLE_RFDETR_OPTIMIZE)

    parser.add_argument("--fps_assumption", type=float, default=DEFAULT_FPS_ASSUMPTION)
    parser.add_argument("--ball_tracker_buffer_size", type=int, default=DEFAULT_BALL_TRACKER_BUFFER_SIZE)
    parser.add_argument("--position_threshold", type=float, default=DEFAULT_POSITION_THRESHOLD)
    parser.add_argument("--velocity_threshold", type=float, default=DEFAULT_VELOCITY_THRESHOLD)
    parser.add_argument("--history_frames", type=int, default=DEFAULT_HISTORY_FRAMES)
    parser.add_argument("--max_lost_seconds", type=float, default=DEFAULT_MAX_LOST_SECONDS)
    parser.add_argument("--stable_track_threshold", type=int, default=DEFAULT_STABLE_TRACK_THRESHOLD)
    parser.add_argument("--validation_gate_threshold", type=float, default=DEFAULT_VALIDATION_GATE_THRESHOLD)

    parser.add_argument("--interpolation_velocity_threshold", type=float, default=DEFAULT_INTERPOLATION_VELOCITY_THRESHOLD)
    parser.add_argument("--max_interpolation_frames", type=int, default=DEFAULT_MAX_INTERPOLATION_FRAMES)
    parser.add_argument("--interpolated_box_size", type=float, default=DEFAULT_INTERPOLATED_BOX_SIZE)

    parser.add_argument("--max_optical_flow_gap", type=int, default=DEFAULT_MAX_OPTICAL_FLOW_GAP)
    parser.add_argument("--optical_flow_error_threshold", type=float, default=DEFAULT_OPTICAL_FLOW_ERROR_THRESHOLD)
    parser.add_argument("--optical_flow_max_movement", type=float, default=DEFAULT_OPTICAL_FLOW_MAX_MOVEMENT)
    parser.add_argument("--optical_flow_win_size", type=parse_pair, default=parse_pair(DEFAULT_OPTICAL_FLOW_WIN_SIZE))
    parser.add_argument("--optical_flow_max_level", type=int, default=DEFAULT_OPTICAL_FLOW_MAX_LEVEL)
    parser.add_argument("--optical_flow_criteria_eps", type=float, default=DEFAULT_OPTICAL_FLOW_CRITERIA_EPS)
    parser.add_argument("--optical_flow_criteria_count", type=int, default=DEFAULT_OPTICAL_FLOW_CRITERIA_COUNT)

    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--match_iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--eval_score_threshold", type=float, default=DEFAULT_EVAL_SCORE_THRESHOLD)
    parser.add_argument("--ball_category_id", type=int, default=DEFAULT_BALL_CATEGORY_ID)
    return parser


def process_sequence(
    sequence: BenchmarkSequence,
    ball_model: RFDETRMedium,
    player_model: RFDETRMedium,
    args: argparse.Namespace,
    run_dir: Path,
) -> Tuple[SequenceSummary, List[dict], List[dict], List[int]]:
    if not sequence.image_paths:
        raise FileNotFoundError(f"No frames found for {sequence.name}")

    seq_output_dir = ensure_dir(run_dir / sequence.name)
    frame_trace_path = seq_output_dir / "frame_trace.csv"
    point_output_path = seq_output_dir / "point_outputs.json"
    state = LBTrackingState(args)

    player_class_ids = np.asarray(parse_int_list(args.player_class_ids), dtype=np.int32)
    frame_records: List[FrameRecord] = []
    point_records: List[dict] = []
    detection_records: List[dict] = []
    processed_image_ids: List[int] = []
    raw_pred_frames = 0
    final_pred_frames = 0
    optical_flow_frames = 0

    for frame_id, image_path in enumerate(sequence.image_paths, start=1):
        total_start = time.perf_counter()
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read frame {image_path}")
        frame_h, frame_w = frame.shape[:2]

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        coord_transform = ensure_coord_transform(state.motion_estimator.update(frame))
        predicted_pos_abs = state.kf.predict() if state.track_initialized else None

        image_id = sequence.image_ids_by_frame.get(frame_id)
        if image_id is not None:
            processed_image_ids.append(int(image_id))

        primary_start = time.perf_counter()
        player_detections = player_model.predict(frame, confidence=args.player_confidence)
        if len(player_detections.xyxy) > 0:
            player_detections = player_detections[np.isin(player_detections.class_id, player_class_ids)]
        else:
            player_detections = sv.Detections.empty()

        ball_detections = ball_model.predict(frame, confidence=args.ball_confidence)
        if len(ball_detections.xyxy) > 0:
            ball_detections = ball_detections[ball_detections.class_id == args.ball_class_id]
        else:
            ball_detections = sv.Detections.empty()
        primary_ms = (time.perf_counter() - primary_start) * 1000.0

        raw_best_detection = top_confidence_detection(ball_detections)

        valid_ball_indices: List[int] = []
        if len(ball_detections.xyxy) > 0:
            ball_centers_rel = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
            for idx, center in enumerate(ball_centers_rel):
                if not lb.is_point_in_boxes(center, player_detections.xyxy):
                    valid_ball_indices.append(idx)
        filtered_ball_detections = (
            ball_detections[valid_ball_indices]
            if valid_ball_indices
            else sv.Detections.empty()
        )
        filtered_best_detection = top_confidence_detection(filtered_ball_detections)
        if filtered_best_detection is not None:
            raw_pred_frames += 1
            raw_box = clip_xyxy(
                np.asarray(filtered_best_detection.xyxy[0], dtype=np.float32),
                width=frame_w,
                height=frame_h,
            )
            detection_records.append(
                build_detection_record(
                    sequence=sequence.name,
                    frame_index=frame_id,
                    original_frame=sequence.original_frames_by_frame[frame_id],
                    file_name=sequence.file_names_by_frame[frame_id],
                    image_id=image_id,
                    bbox_xyxy=raw_box,
                    score=float(filtered_best_detection.confidence[0]),
                    stage="raw",
                    source="rfdetr_player_excluded",
                )
            )

        tracking_start = time.perf_counter()
        best_detection = state.get_best_detection(filtered_ball_detections, predicted_pos_abs)
        should_use_detection, accepted_detection = state.process_detection(best_detection, frame_id)

        detections_op = state.ball_tracker.update(filtered_ball_detections)
        measurement_abs_of = None
        if len(detections_op.xyxy) > 0:
            center_rel_of = detections_op.get_anchors_coordinates(sv.Position.CENTER)
            measurement_abs_of = coord_transform.rel_to_abs(center_rel_of).flatten()

        measurement_abs = None
        final_box: Optional[np.ndarray] = None
        final_center: Optional[np.ndarray] = None
        output_score = 0.0
        output_source = "none"
        if accepted_detection is not None and should_use_detection:
            center_rel = accepted_detection["center"]
            measurement_abs = coord_transform.rel_to_abs(center_rel.reshape(1, -1)).flatten()
            final_box = clip_xyxy(
                np.asarray(accepted_detection["sv"].xyxy[0], dtype=np.float32),
                width=frame_w,
                height=frame_h,
            )
            final_center = np.asarray(center_rel, dtype=np.float32)
            output_score = float(accepted_detection["sv"].confidence[0])
            output_source = "rfdetr"
            state.optical_flow_gap_counter = 0

        if (
            measurement_abs_of is None
            and state.optical_kf_track_init
            and state.optical_flow_gap_counter < int(args.max_optical_flow_gap)
            and state.optical_flow_points_rel is not None
            and state.prev_gray_frame is not None
        ):
            current_point_rel = state.optical_flow_points_rel[0][0]
            current_point_in_bounds = (
                0.0 <= float(current_point_rel[0]) < float(frame_w)
                and 0.0 <= float(current_point_rel[1]) < float(frame_h)
            )
            if current_point_in_bounds:
                new_points_rel, status, error = cv2.calcOpticalFlowPyrLK(
                    state.prev_gray_frame,
                    gray_frame,
                    state.optical_flow_points_rel,
                    None,
                    **state.lk_params,
                )
                if status is not None and int(status[0][0]) == 1:
                    optical_flow_error = (
                        float(error[0][0])
                        if error is not None and len(error) > 0
                        else 0.0
                    )
                    new_point_rel = new_points_rel[0][0]
                    new_point_in_bounds = (
                        0.0 <= float(new_point_rel[0]) < float(frame_w)
                        and 0.0 <= float(new_point_rel[1]) < float(frame_h)
                    )
                    movement_distance = float(
                        np.linalg.norm(new_point_rel.astype(np.float32) - current_point_rel.astype(np.float32))
                    )
                    inside_player_box = lb.is_point_in_boxes(new_point_rel, player_detections.xyxy)
                    if (
                        optical_flow_error <= float(args.optical_flow_error_threshold)
                        and new_point_in_bounds
                        and movement_distance <= float(args.optical_flow_max_movement)
                        and not inside_player_box
                    ):
                        measurement_abs_of = coord_transform.rel_to_abs(new_points_rel[0]).flatten()
                        state.optical_flow_gap_counter += 1
                        if final_box is None:
                            x_rel, y_rel = new_points_rel[0].ravel()
                            final_center = np.array([x_rel, y_rel], dtype=np.float32)
                            final_box = clip_xyxy(
                                center_wh_to_xyxy(
                                    final_center,
                                    np.array(
                                        [args.interpolated_box_size, args.interpolated_box_size],
                                        dtype=np.float32,
                                    ),
                                ),
                                width=frame_w,
                                height=frame_h,
                            )
                            output_source = "optical_flow"
                            output_score = 0.0

        if measurement_abs_of is not None:
            if not state.optical_kf_track_init:
                state.optical_flow_kf.initialize_state(measurement_abs_of)
                state.optical_kf_track_init = True
            else:
                state.optical_flow_kf.update(measurement_abs_of)
            state.prev_position_abs = state.optical_flow_kf.x_hat[:2].flatten()
            current_pos_rel_of = coord_transform.abs_to_rel(np.array([state.prev_position_abs]))[0]
            state.optical_flow_points_rel = np.array([[current_pos_rel_of]], dtype=np.float32)
        else:
            state.optical_kf_track_init = False
            state.optical_flow_points_rel = None
            state.prev_position_abs = None
            if state.track_lost_count >= state.max_lost_frames and state.track_initialized:
                state.track_hit_streak = 0
                state.track_lost_count = 0

        if measurement_abs is not None:
            if not state.track_initialized:
                state.kf.initialize_state(measurement_abs)
                state.track_initialized = True
            else:
                state.kf.update(measurement_abs)
        else:
            if state.track_lost_count >= state.max_lost_frames and state.track_initialized:
                state.track_initialized = False
                state.track_hit_streak = 0
                state.track_lost_count = 0

        if final_box is not None:
            final_pred_frames += 1
            if output_source == "optical_flow":
                optical_flow_frames += 1
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
            if final_center is None:
                x1, y1, x2, y2 = final_box
                final_center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            point_records.append(
                {
                    "frame_idx": int(frame_id),
                    "x": float(final_center[0]),
                    "y": float(final_center[1]),
                    "confidence": float(output_score),
                    "source": output_source,
                }
            )
        else:
            point_records.append(
                {
                    "frame_idx": int(frame_id),
                    "x": None,
                    "y": None,
                    "confidence": None,
                    "source": "none",
                }
            )

        tracking_ms = (time.perf_counter() - tracking_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        frame_records.append(
            FrameRecord(
                frame=frame_id,
                player_count=int(len(player_detections.xyxy)),
                raw_candidate_count=int(len(ball_detections.xyxy)),
                filtered_candidate_count=int(len(filtered_ball_detections.xyxy)),
                raw_best_score=float(raw_best_detection.confidence[0]) if raw_best_detection is not None else 0.0,
                filtered_best_score=float(filtered_best_detection.confidence[0]) if filtered_best_detection is not None else 0.0,
                selected_score=float(best_detection["sv"].confidence[0]) if best_detection is not None else 0.0,
                output_score=float(output_score),
                output_source=output_source,
                tracker_active=bool(state.track_initialized),
                optical_flow_active=bool(state.optical_kf_track_init),
                track_gap=int(state.track_lost_count),
                optical_flow_gap=int(state.optical_flow_gap_counter),
                hit_streak=int(state.track_hit_streak),
                primary_ms=primary_ms,
                tracking_ms=tracking_ms,
                total_ms=total_ms,
            )
        )
        state.prev_gray_frame = gray_frame.copy()

    csv_write_dicts(frame_trace_path, [asdict(record) for record in frame_records])
    point_output_path.write_text(json.dumps(point_records, indent=2), encoding="utf-8")
    summary = SequenceSummary(
        sequence=sequence.name,
        frames_total=len(frame_records),
        raw_pred_frames=raw_pred_frames,
        final_pred_frames=final_pred_frames,
        optical_flow_frames=optical_flow_frames,
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
    global args
    args = build_arg_parser().parse_args()

    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.ball_model_path = args.ball_model_path.expanduser().resolve()
    args.player_model_path = args.player_model_path.expanduser().resolve()
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
    print(f"[INFO] Ball model path: {args.ball_model_path}")
    print(f"[INFO] Player model path: {args.player_model_path}")
    print(f"[INFO] Ball confidence: {args.ball_confidence}")
    print(f"[INFO] Player confidence: {args.player_confidence}")
    print(
        f"[INFO] LB config: pos_thresh={args.position_threshold}, vel_thresh={args.velocity_threshold}, "
        f"history={args.history_frames}, of_gap={args.max_optical_flow_gap}"
    )
    if annotations_path is not None:
        print(f"[INFO] Evaluation annotations: {annotations_path}")

    ball_model_kwargs = {"pretrain_weights": str(args.ball_model_path)}
    if int(args.ball_model_resolution) > 0:
        ball_model_kwargs["resolution"] = int(args.ball_model_resolution)
    ball_model = RFDETRMedium(**ball_model_kwargs)

    player_model_kwargs = {"pretrain_weights": str(args.player_model_path)}
    if int(args.player_model_resolution) > 0:
        player_model_kwargs["resolution"] = int(args.player_model_resolution)
    player_model = RFDETRMedium(**player_model_kwargs)

    if args.enable_rfdetr_optimize:
        for name, model in (("ball", ball_model), ("player", player_model)):
            try:
                model.optimize_for_inference()
                print(f"[INFO] Optimized {name} RF-DETR model for inference")
            except Exception as exc:
                print(f"[WARN] Could not optimize {name} RF-DETR model: {exc}")

    summaries: List[SequenceSummary] = []
    all_detections: List[dict] = []
    all_point_outputs: List[dict] = []
    processed_image_ids: List[int] = []
    for sequence in sequences:
        print(f"[INFO] Processing {sequence.name}")
        summary, sequence_detections, sequence_points, sequence_image_ids = process_sequence(
            sequence=sequence,
            ball_model=ball_model,
            player_model=player_model,
            args=args,
            run_dir=run_dir,
        )
        summaries.append(summary)
        all_detections.extend(sequence_detections)
        all_point_outputs.extend(sequence_points)
        processed_image_ids.extend(sequence_image_ids)
        print(
            f"[INFO] {sequence.name}: raw_predictions={summary.raw_pred_frames}, "
            f"final_predictions={summary.final_pred_frames}, optical_flow_frames={summary.optical_flow_frames}, "
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
        "logic_reference": str((REPO_ROOT / "ball_tracking_lb.py").resolve()),
        "logic_notes": {
            "raw_stage_definition": "Top ball detection after player-exclusion filtering, before LB tracking/fallback logic.",
            "final_stage_definition": "Accepted RF-DETR detection or optical-flow fallback box when emitted by the pipeline.",
            "serialization_only_change": "Final benchmark output records one final box per frame even if the original preview script would draw/write redundantly.",
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
