#!/usr/bin/env python3
"""
RF-DETR ball_1120 primary + EdgeTAM short-horizon fallback benchmark runner.

This experiment keeps RF-DETR as the primary ball detector and only lets
EdgeTAM bridge short miss gaps after a real RF-DETR acceptance. EdgeTAM is
never allowed to recursively reseed from its own fallback outputs, which keeps
the benchmark focused on recall recovery instead of standalone propagation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from ball_detection_metrics import (
    BALL_CATEGORY_ID,
    build_detection_record,
    evaluate_detections,
    write_benchmark_metrics_csv,
    write_detection_export,
)
from benchmark_dataset import BenchmarkSequence, default_annotation_path
from benchmark_rfdetr_groundingdino_fallback_tracking import (
    DEFAULT_BALL_CLASS_ID,
    DEFAULT_BALL_CONFIDENCE,
    DEFAULT_DATA_ROOT,
    DEFAULT_ENABLE_RFDETR_OPTIMIZE,
    DEFAULT_EVAL_SCORE_THRESHOLD,
    DEFAULT_FPS,
    DEFAULT_MATCH_IOU,
    DEFAULT_MAX_ACTIVE_GAP,
    DEFAULT_MAX_FRAMES_PER_SEQ,
    DEFAULT_OUTLIER_HISTORY_FRAMES,
    DEFAULT_OUTLIER_RESET_FRAMES,
    DEFAULT_OUTLIER_WAIT_FRAMES,
    DEFAULT_PLAYER_CLASS_IDS,
    DEFAULT_PLAYER_CONFIDENCE,
    DEFAULT_POSITION_THRESHOLD,
    DEFAULT_SEQ_END,
    DEFAULT_SEQ_LIST,
    DEFAULT_SEQ_START,
    DEFAULT_STABLE_TRACK_THRESHOLD,
    DEFAULT_VALIDATION_GATE_THRESHOLD,
    DEFAULT_VELOCITY_THRESHOLD,
    DetectionCandidate,
    FrameRecord,
    RFDETRGroundingDinoTracker,
    RFDetrDetector,
    SequenceSummary,
    aggregate_sequence_summaries,
    area_of,
    clip_xyxy,
    csv_write_dicts,
    ensure_dir,
    exclude_candidates_in_boxes,
    load_sequence_fps,
    mean_or_zero,
    parse_int_list,
    resolve_sequences,
    safe_div,
    str2bool,
    write_track_predictions,
    xyxy_center,
    xyxy_wh,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EDGETAM_ROOT = REPO_ROOT / "EdgeTAM"
if str(EDGETAM_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGETAM_ROOT))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "rfdetr_edgetam_fallback_tracking"
DEFAULT_RUN_NAME = "rfdetr_edgetam_fallback_v1"

DEFAULT_BALL_MODEL_PATH = REPO_ROOT / "checkpoints" / "ball_1120.pth"
DEFAULT_PLAYER_MODEL_PATH = REPO_ROOT / "checkpoints" / "player.pth"
DEFAULT_BALL_RESOLUTION = 1120
DEFAULT_ENABLE_PLAYER_EXCLUSION = False

DEFAULT_EDGETAM_CHECKPOINT = EDGETAM_ROOT / "checkpoints" / "edgetam.pt"
DEFAULT_EDGETAM_CONFIG = "configs/edgetam.yaml"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_USE_AMP = True

DEFAULT_ENABLE_EDGETAM_FALLBACK = True
DEFAULT_EDGETAM_TRACK_HORIZON = DEFAULT_MAX_ACTIVE_GAP
DEFAULT_EDGETAM_MASK_THRESHOLD = 0.0
DEFAULT_EDGETAM_MIN_BOX_AREA = 4.0
DEFAULT_EDGETAM_MAX_BOX_AREA = 6000.0
DEFAULT_EDGETAM_MIN_BOX_SIDE = 2.0
DEFAULT_EDGETAM_OFFLOAD_VIDEO_TO_CPU = True
DEFAULT_EDGETAM_OFFLOAD_STATE_TO_CPU = True
DEFAULT_EDGETAM_ASYNC_FRAME_LOADING = True
DEFAULT_STAGE_FRAME_DIGITS = 6


class RFDetr1120Detector(RFDetrDetector):
    def __init__(
        self,
        ball_model_path: Path,
        player_model_path: Path,
        ball_confidence: float,
        player_confidence: float,
        ball_class_id: int,
        player_class_ids: Sequence[int],
        enable_player_exclusion: bool,
        optimize_for_inference: bool,
        ball_resolution: Optional[int],
    ) -> None:
        if not ball_model_path.exists():
            raise FileNotFoundError(f"RF-DETR ball checkpoint not found: {ball_model_path}")
        if enable_player_exclusion and not player_model_path.exists():
            raise FileNotFoundError(f"RF-DETR player checkpoint not found: {player_model_path}")

        self.ball_confidence = ball_confidence
        self.player_confidence = player_confidence
        self.ball_class_id = ball_class_id
        self.player_class_ids = np.array(list(player_class_ids), dtype=np.int32)
        self.enable_player_exclusion = enable_player_exclusion

        from rfdetr import RFDETRMedium

        ball_model_kwargs = {"pretrain_weights": str(ball_model_path)}
        if ball_resolution is not None and int(ball_resolution) > 0:
            ball_model_kwargs["resolution"] = int(ball_resolution)

        self.ball_model = RFDETRMedium(**ball_model_kwargs)
        self.player_model = RFDETRMedium(pretrain_weights=str(player_model_path)) if enable_player_exclusion else None

        if optimize_for_inference:
            for name, model in [("ball", self.ball_model), ("player", self.player_model)]:
                if model is None:
                    continue
                try:
                    model.optimize_for_inference()
                    print(f"[INFO] Optimized RF-DETR {name} model for inference")
                except Exception as exc:
                    print(f"[WARN] Could not optimize RF-DETR {name} model: {exc}")

    def predict_ball_candidates(self, image_bgr: np.ndarray) -> Tuple[List[DetectionCandidate], np.ndarray]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        player_boxes = np.empty((0, 4), dtype=np.float32)
        if self.enable_player_exclusion and self.player_model is not None:
            player_detections = self.player_model.predict(image_rgb, confidence=self.player_confidence)
            if len(player_detections) > 0:
                player_xyxy, _player_confidence, player_class_ids = self._extract_detection_arrays(player_detections)
                keep = np.isin(player_class_ids, self.player_class_ids)
                if np.any(keep):
                    player_boxes = np.asarray(player_xyxy[keep], dtype=np.float32)

        ball_detections = self.ball_model.predict(image_rgb, confidence=self.ball_confidence)
        if len(ball_detections) == 0:
            return [], player_boxes

        ball_xyxy, ball_confidence, ball_class_ids = self._extract_detection_arrays(ball_detections)
        keep_ball = ball_class_ids == self.ball_class_id
        if not np.any(keep_ball):
            return [], player_boxes
        ball_xyxy = np.asarray(ball_xyxy[keep_ball], dtype=np.float32)
        ball_confidence = np.asarray(ball_confidence[keep_ball], dtype=np.float32)

        if self.enable_player_exclusion and player_boxes.size > 0:
            centers = np.column_stack(
                (
                    (ball_xyxy[:, 0] + ball_xyxy[:, 2]) / 2.0,
                    (ball_xyxy[:, 1] + ball_xyxy[:, 3]) / 2.0,
                )
            )
            valid_indices = [
                index
                for index, center in enumerate(centers)
                if not np.any(
                    (center[0] >= player_boxes[:, 0])
                    & (center[0] <= player_boxes[:, 2])
                    & (center[1] >= player_boxes[:, 1])
                    & (center[1] <= player_boxes[:, 3])
                )
            ]
            ball_xyxy = ball_xyxy[valid_indices]
            ball_confidence = ball_confidence[valid_indices]

        return self._detections_to_candidates(ball_xyxy, ball_confidence, source="rfdetr"), player_boxes


def mask_logits_to_xyxy(mask_logits: torch.Tensor, threshold: float) -> Optional[np.ndarray]:
    mask = (mask_logits.squeeze() > threshold).detach()
    if mask.ndim != 2:
        return None

    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return None

    ys = coords[:, 0]
    xs = coords[:, 1]
    return np.array(
        [
            float(xs.min().item()),
            float(ys.min().item()),
            float(xs.max().item() + 1),
            float(ys.max().item() + 1),
        ],
        dtype=np.float32,
    )


def mask_logits_to_score(mask_logits: torch.Tensor, threshold: float) -> float:
    logits = mask_logits.squeeze().detach()
    if logits.numel() == 0:
        return 0.0

    positive_logits = logits[logits > threshold]
    if positive_logits.numel() == 0:
        return 0.0

    mean_logit = float(positive_logits.float().mean().item())
    return float(1.0 / (1.0 + np.exp(-mean_logit)))


def gate_candidates_by_prediction(
    candidates: Sequence[DetectionCandidate],
    predicted_position: Optional[np.ndarray],
    gate_distance: float,
) -> List[DetectionCandidate]:
    if predicted_position is None:
        return list(candidates)

    gated: List[DetectionCandidate] = []
    for candidate in candidates:
        distance = float(np.linalg.norm(xyxy_center(candidate.xyxy) - predicted_position))
        if distance <= gate_distance:
            gated.append(candidate)
    return gated


def stage_sequence_for_edgetam(
    sequence: BenchmarkSequence,
    staging_root: Path,
    frame_digits: int = DEFAULT_STAGE_FRAME_DIGITS,
) -> Path:
    seq_stage_dir = staging_root / sequence.name
    if seq_stage_dir.exists():
        shutil.rmtree(seq_stage_dir)
    ensure_dir(seq_stage_dir)

    for frame_idx, image_path in enumerate(sequence.image_paths):
        staged_path = seq_stage_dir / f"{frame_idx:0{frame_digits}d}.jpg"
        suffix = image_path.suffix.lower()

        if suffix in {".jpg", ".jpeg"}:
            try:
                staged_path.symlink_to(image_path.resolve())
            except OSError:
                shutil.copy2(image_path, staged_path)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read frame {image_path} while staging {sequence.name}")
        ok = cv2.imwrite(str(staged_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"Could not convert {image_path} to staged JPEG for {sequence.name}")

    return seq_stage_dir


class EdgeTAMVideoFallbackDetector:
    def __init__(
        self,
        checkpoint_path: Path,
        config_name: str,
        device: str,
        use_amp: bool,
        track_horizon: int,
        mask_threshold: float,
        min_box_area: float,
        max_box_area: float,
        min_box_side: float,
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
        async_loading_frames: bool,
    ) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"EdgeTAM checkpoint not found: {checkpoint_path}")

        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.track_horizon = max(1, int(track_horizon))
        self.mask_threshold = float(mask_threshold)
        self.min_box_area = float(min_box_area)
        self.max_box_area = float(max_box_area)
        self.min_box_side = float(min_box_side)
        self.offload_video_to_cpu = bool(offload_video_to_cpu)
        self.offload_state_to_cpu = bool(offload_state_to_cpu)
        self.async_loading_frames = bool(async_loading_frames)

        self.predictor = build_sam2_video_predictor(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=str(self.device),
        )

        self.sequence_name: Optional[str] = None
        self.sequence: Optional[BenchmarkSequence] = None
        self.sequence_staging_root: Optional[Path] = None
        self.sequence_state = None
        self.cached_candidates_by_frame: Dict[int, DetectionCandidate] = {}

    def prepare_sequence(self, sequence: BenchmarkSequence, staging_root: Path) -> None:
        self.sequence_name = sequence.name
        self.sequence = sequence
        self.sequence_staging_root = ensure_dir(staging_root / sequence.name)
        self.cached_candidates_by_frame.clear()
        self.sequence_state = None

    def reset_runtime(self) -> None:
        if self.sequence_state is not None:
            self.predictor.reset_state(self.sequence_state)
        self.sequence_state = None
        self.cached_candidates_by_frame.clear()

    def has_future_prediction(self, frame_id: int) -> bool:
        return frame_id in self.cached_candidates_by_frame

    def get_cached_candidate(
        self,
        frame_id: int,
        source: str = "edgetam_fallback",
    ) -> Optional[DetectionCandidate]:
        candidate = self.cached_candidates_by_frame.get(frame_id)
        if candidate is None:
            return None

        return DetectionCandidate(
            xyxy=candidate.xyxy.copy(),
            score=float(candidate.score),
            phrase=candidate.phrase,
            source=source,
        )

    def reseed_from_box(self, frame_id: int, box_xyxy: np.ndarray) -> None:
        if self.sequence is None or self.sequence_staging_root is None:
            raise RuntimeError("EdgeTAM sequence context is not initialized")

        if self.sequence_state is not None:
            self.predictor.reset_state(self.sequence_state)
        self.cached_candidates_by_frame.clear()

        start_index = int(frame_id) - 1
        end_index = min(start_index + self.track_horizon, len(self.sequence.image_paths) - 1)
        window_stage_dir = self._stage_window(start_index=start_index, end_index=end_index)
        self.sequence_state = self.predictor.init_state(
            video_path=str(window_stage_dir),
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=self.async_loading_frames,
        )

        frame_idx = 0
        box = np.asarray(box_xyxy, dtype=np.float32)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_amp
            else contextlib.nullcontext()
        )

        with torch.inference_mode():
            with autocast_context:
                self.predictor.add_new_points_or_box(
                    inference_state=self.sequence_state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    box=box,
                )
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                    self.sequence_state,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=self.track_horizon,
                    reverse=False,
                ):
                    for obj_offset, obj_id in enumerate(out_obj_ids):
                        if int(obj_id) != 1:
                            continue
                        candidate = self._mask_logits_to_candidate(
                            mask_logits=out_mask_logits[obj_offset],
                            source="edgetam_fallback",
                        )
                        output_frame_id = start_index + int(out_frame_idx) + 1
                        if candidate is None:
                            self.cached_candidates_by_frame.pop(output_frame_id, None)
                        else:
                            self.cached_candidates_by_frame[output_frame_id] = candidate

    def _stage_window(self, start_index: int, end_index: int) -> Path:
        if self.sequence is None or self.sequence_staging_root is None:
            raise RuntimeError("EdgeTAM sequence context is not initialized")

        window_name = f"{start_index:06d}_{end_index:06d}"
        window_dir = self.sequence_staging_root / window_name
        if window_dir.exists():
            shutil.rmtree(window_dir)
        ensure_dir(window_dir)

        for offset, image_path in enumerate(self.sequence.image_paths[start_index : end_index + 1]):
            staged_path = window_dir / f"{offset:0{DEFAULT_STAGE_FRAME_DIGITS}d}.jpg"
            suffix = image_path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                try:
                    staged_path.symlink_to(image_path.resolve())
                except OSError:
                    shutil.copy2(image_path, staged_path)
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read frame {image_path} while staging EdgeTAM window")
            ok = cv2.imwrite(str(staged_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                raise RuntimeError(f"Could not convert {image_path} to staged JPEG for EdgeTAM window")

        return window_dir

    def _mask_logits_to_candidate(
        self,
        mask_logits: torch.Tensor,
        source: str,
    ) -> Optional[DetectionCandidate]:
        box = mask_logits_to_xyxy(mask_logits, self.mask_threshold)
        if box is None:
            return None

        box = np.asarray(box, dtype=np.float32)
        wh = xyxy_wh(box)
        box_area = area_of(box)
        if wh[0] < self.min_box_side or wh[1] < self.min_box_side:
            return None
        if box_area < self.min_box_area or box_area > self.max_box_area:
            return None

        return DetectionCandidate(
            xyxy=box,
            score=mask_logits_to_score(mask_logits, self.mask_threshold),
            phrase="ball",
            source=source,
        )


def process_sequence(
    sequence: BenchmarkSequence,
    rfdetr_detector: RFDetr1120Detector,
    edgetam_detector: Optional[EdgeTAMVideoFallbackDetector],
    args: argparse.Namespace,
    run_dir: Path,
    staging_root: Optional[Path],
) -> Tuple[SequenceSummary, Path, List[dict]]:
    seq_name = sequence.name
    image_paths = sequence.image_paths
    if not image_paths:
        raise FileNotFoundError(f"No frames found for {seq_name}")

    tracker = RFDETRGroundingDinoTracker(fps=load_sequence_fps(sequence, args.fps), args=args)

    seq_output_dir = ensure_dir(run_dir / seq_name)
    prediction_path = seq_output_dir / "tracker_predictions.txt"
    frame_trace_path = seq_output_dir / "frame_trace.csv"

    if edgetam_detector is not None:
        if staging_root is None:
            raise RuntimeError("EdgeTAM staging root is required when the fallback is enabled")
        print(f"[INFO] Preparing EdgeTAM state for {seq_name}")
        edgetam_detector.prepare_sequence(sequence=sequence, staging_root=staging_root)

    frame_records: List[FrameRecord] = []
    predictions: List[Tuple[int, object, float]] = []
    detection_records: List[dict] = []
    raw_pred_frames = 0
    final_pred_frames = 0
    fallback_recoveries = 0

    for frame_id, image_path in enumerate(image_paths, start=1):
        total_start = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read frame {image_path}")

        primary_start = time.perf_counter()
        raw_candidates, player_boxes = rfdetr_detector.predict_ball_candidates(image)
        primary_ms = (time.perf_counter() - primary_start) * 1000.0
        raw_best = raw_candidates[0] if raw_candidates else None
        if raw_best is not None:
            raw_pred_frames += 1
            detection_records.append(
                build_detection_record(
                    sequence=seq_name,
                    frame_index=frame_id,
                    original_frame=sequence.original_frames_by_frame[frame_id],
                    file_name=sequence.file_names_by_frame[frame_id],
                    image_id=sequence.image_ids_by_frame[frame_id],
                    bbox_xyxy=raw_best.xyxy,
                    score=raw_best.score,
                    stage="raw",
                    source=raw_best.source,
                )
            )

        tracking_start = time.perf_counter()
        predicted_position = tracker.predict()

        accepted: Optional[DetectionCandidate] = None
        primary_selected = tracker.select_best_candidate(raw_candidates, predicted_position)
        if primary_selected is not None and tracker.try_accept_candidate(primary_selected):
            accepted = primary_selected

        fallback_candidates: List[DetectionCandidate] = []
        fallback_ms = 0.0
        if accepted is None and args.enable_edgetam_fallback and edgetam_detector is not None:
            fallback_start = time.perf_counter()
            if tracker.track_initialized and predicted_position is not None:
                cached_candidate = edgetam_detector.get_cached_candidate(frame_id)
                if cached_candidate is not None:
                    fallback_candidates = [cached_candidate]
                    fallback_candidates = exclude_candidates_in_boxes(fallback_candidates, player_boxes)
                    fallback_candidates = gate_candidates_by_prediction(
                        fallback_candidates,
                        predicted_position=predicted_position,
                        gate_distance=args.validation_gate_threshold,
                    )
            fallback_ms = (time.perf_counter() - fallback_start) * 1000.0

            fallback_selected = tracker.select_best_candidate(fallback_candidates, predicted_position)
            if fallback_selected is not None:
                fallback_selected = DetectionCandidate(
                    xyxy=fallback_selected.xyxy,
                    score=fallback_selected.score,
                    phrase=fallback_selected.phrase,
                    source="edgetam_fallback",
                )
                if tracker.try_accept_candidate(fallback_selected):
                    accepted = fallback_selected
                    fallback_recoveries += 1

        if accepted is None:
            tracker.handle_miss()
            output_box = None
            output_score = 0.0
            output_source = "none"
        else:
            output_box = accepted.xyxy.copy()
            output_score = accepted.score
            output_source = accepted.source

        if (
            output_box is not None
            and output_source == "rfdetr"
            and args.enable_edgetam_fallback
            and edgetam_detector is not None
            and not edgetam_detector.has_future_prediction(frame_id + 1)
        ):
            try:
                edgetam_detector.reseed_from_box(frame_id=frame_id, box_xyxy=output_box)
            except Exception as exc:
                print(f"[WARN] EdgeTAM reseed failed on {seq_name} frame {frame_id}: {exc}")
                edgetam_detector.reset_runtime()

        tracking_ms = (time.perf_counter() - tracking_start) * 1000.0

        if output_box is not None:
            output_box = clip_xyxy(output_box, width=image.shape[1], height=image.shape[0])
            predictions.append((frame_id, output_box, output_score))
            final_pred_frames += 1
            detection_records.append(
                build_detection_record(
                    sequence=seq_name,
                    frame_index=frame_id,
                    original_frame=sequence.original_frames_by_frame[frame_id],
                    file_name=sequence.file_names_by_frame[frame_id],
                    image_id=sequence.image_ids_by_frame[frame_id],
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
                gt_exists=False,
                gt_iou_raw=0.0,
                gt_iou_final=0.0,
                raw_candidate_count=len(raw_candidates),
                fallback_candidate_count=len(fallback_candidates),
                raw_best_score=raw_best.score if raw_best is not None else 0.0,
                fallback_best_score=fallback_candidates[0].score if fallback_candidates else 0.0,
                output_score=output_score,
                output_source=output_source,
                tracker_active=tracker.track_initialized,
                tracker_gap=tracker.track_lost_count,
                primary_ms=primary_ms,
                fallback_ms=fallback_ms,
                tracking_ms=tracking_ms,
                total_ms=total_ms,
            )
        )

    write_track_predictions(prediction_path, predictions)
    csv_write_dicts(frame_trace_path, [asdict(record) for record in frame_records])
    summary = SequenceSummary(
        sequence=seq_name,
        frames_total=len(frame_records),
        gt_frames=0,
        raw_pred_frames=raw_pred_frames,
        final_pred_frames=final_pred_frames,
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
        fallback_recoveries=fallback_recoveries,
        primary_ms_avg=mean_or_zero([item.primary_ms for item in frame_records]),
        fallback_ms_avg=mean_or_zero([item.fallback_ms for item in frame_records]),
        tracking_ms_avg=mean_or_zero([item.tracking_ms for item in frame_records]),
        total_ms_avg=mean_or_zero([item.total_ms for item in frame_records]),
        runtime_fps=safe_div(1000.0, mean_or_zero([item.total_ms for item in frame_records])),
    )
    return summary, prediction_path, detection_records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RF-DETR ball_1120 primary + EdgeTAM short-horizon fallback benchmark runner."
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
    parser.add_argument("--ball_resolution", type=int, default=DEFAULT_BALL_RESOLUTION)
    parser.add_argument("--ball_confidence", type=float, default=DEFAULT_BALL_CONFIDENCE)
    parser.add_argument("--player_confidence", type=float, default=DEFAULT_PLAYER_CONFIDENCE)
    parser.add_argument("--ball_class_id", type=int, default=DEFAULT_BALL_CLASS_ID)
    parser.add_argument("--player_class_ids", type=str, default=DEFAULT_PLAYER_CLASS_IDS)
    parser.add_argument("--enable_player_exclusion", type=str2bool, default=DEFAULT_ENABLE_PLAYER_EXCLUSION)
    parser.add_argument("--enable_rfdetr_optimize", type=str2bool, default=DEFAULT_ENABLE_RFDETR_OPTIMIZE)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--validation_gate_threshold", type=float, default=DEFAULT_VALIDATION_GATE_THRESHOLD)
    parser.add_argument("--stable_track_threshold", type=int, default=DEFAULT_STABLE_TRACK_THRESHOLD)
    parser.add_argument("--max_active_gap", type=int, default=DEFAULT_MAX_ACTIVE_GAP)

    parser.add_argument("--position_threshold", type=float, default=DEFAULT_POSITION_THRESHOLD)
    parser.add_argument("--velocity_threshold", type=float, default=DEFAULT_VELOCITY_THRESHOLD)
    parser.add_argument("--outlier_history_frames", type=int, default=DEFAULT_OUTLIER_HISTORY_FRAMES)
    parser.add_argument("--outlier_wait_frames", type=int, default=DEFAULT_OUTLIER_WAIT_FRAMES)
    parser.add_argument("--outlier_reset_frames", type=int, default=DEFAULT_OUTLIER_RESET_FRAMES)

    parser.add_argument("--edgetam_checkpoint", type=Path, default=DEFAULT_EDGETAM_CHECKPOINT)
    parser.add_argument("--edgetam_config", type=str, default=DEFAULT_EDGETAM_CONFIG)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--use_amp", type=str2bool, default=DEFAULT_USE_AMP)
    parser.add_argument("--enable_edgetam_fallback", type=str2bool, default=DEFAULT_ENABLE_EDGETAM_FALLBACK)
    parser.add_argument("--edgetam_track_horizon", type=int, default=DEFAULT_EDGETAM_TRACK_HORIZON)
    parser.add_argument("--edgetam_mask_threshold", type=float, default=DEFAULT_EDGETAM_MASK_THRESHOLD)
    parser.add_argument("--edgetam_min_box_area", type=float, default=DEFAULT_EDGETAM_MIN_BOX_AREA)
    parser.add_argument("--edgetam_max_box_area", type=float, default=DEFAULT_EDGETAM_MAX_BOX_AREA)
    parser.add_argument("--edgetam_min_box_side", type=float, default=DEFAULT_EDGETAM_MIN_BOX_SIDE)
    parser.add_argument("--edgetam_offload_video_to_cpu", type=str2bool, default=DEFAULT_EDGETAM_OFFLOAD_VIDEO_TO_CPU)
    parser.add_argument("--edgetam_offload_state_to_cpu", type=str2bool, default=DEFAULT_EDGETAM_OFFLOAD_STATE_TO_CPU)
    parser.add_argument("--edgetam_async_loading_frames", type=str2bool, default=DEFAULT_EDGETAM_ASYNC_FRAME_LOADING)

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
    args.player_model_path = args.player_model_path.expanduser().resolve()
    args.edgetam_checkpoint = args.edgetam_checkpoint.expanduser().resolve()
    args.player_class_ids = parse_int_list(args.player_class_ids)
    if args.annotations is not None:
        args.annotations = args.annotations.expanduser().resolve()

    run_dir = ensure_dir(args.output_root / args.run_name)
    staging_root = ensure_dir(run_dir / "_edgetam_staging") if args.enable_edgetam_fallback else None
    sequences = resolve_sequences(
        data_root=args.data_root,
        seq_start=args.seq_start,
        seq_end=args.seq_end,
        seq_list=args.seq_list,
        max_frames_per_seq=args.max_frames_per_seq,
    )
    annotations_path = args.annotations or default_annotation_path(args.data_root)

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Sequences: {', '.join(sequence.name for sequence in sequences)}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] RF-DETR ball model: {args.ball_model_path}")
    print(f"[INFO] RF-DETR ball resolution: {args.ball_resolution}")
    print(f"[INFO] RF-DETR player exclusion: {args.enable_player_exclusion}")
    print(f"[INFO] EdgeTAM fallback enabled: {args.enable_edgetam_fallback}")
    if args.enable_edgetam_fallback:
        print(f"[INFO] EdgeTAM checkpoint: {args.edgetam_checkpoint}")
        print(f"[INFO] EdgeTAM track horizon: {args.edgetam_track_horizon}")
    if annotations_path is not None:
        print(f"[INFO] Evaluation annotations: {annotations_path}")

    rfdetr_detector = RFDetr1120Detector(
        ball_model_path=args.ball_model_path,
        player_model_path=args.player_model_path,
        ball_confidence=args.ball_confidence,
        player_confidence=args.player_confidence,
        ball_class_id=args.ball_class_id,
        player_class_ids=args.player_class_ids,
        enable_player_exclusion=args.enable_player_exclusion,
        optimize_for_inference=args.enable_rfdetr_optimize,
        ball_resolution=int(args.ball_resolution) if int(args.ball_resolution) > 0 else None,
    )

    edgetam_detector: Optional[EdgeTAMVideoFallbackDetector] = None
    if args.enable_edgetam_fallback:
        edgetam_detector = EdgeTAMVideoFallbackDetector(
            checkpoint_path=args.edgetam_checkpoint,
            config_name=args.edgetam_config,
            device=args.device,
            use_amp=args.use_amp,
            track_horizon=args.edgetam_track_horizon,
            mask_threshold=args.edgetam_mask_threshold,
            min_box_area=args.edgetam_min_box_area,
            max_box_area=args.edgetam_max_box_area,
            min_box_side=args.edgetam_min_box_side,
            offload_video_to_cpu=args.edgetam_offload_video_to_cpu,
            offload_state_to_cpu=args.edgetam_offload_state_to_cpu,
            async_loading_frames=args.edgetam_async_loading_frames,
        )

    summaries: List[SequenceSummary] = []
    all_detections: List[dict] = []
    for sequence in sequences:
        print(f"[INFO] Processing {sequence.name}")
        summary, _prediction_path, sequence_detections = process_sequence(
            sequence=sequence,
            rfdetr_detector=rfdetr_detector,
            edgetam_detector=edgetam_detector,
            args=args,
            run_dir=run_dir,
            staging_root=staging_root,
        )
        summaries.append(summary)
        all_detections.extend(sequence_detections)
        print(
            f"[INFO] {sequence.name}: raw_predictions={summary.raw_pred_frames}, "
            f"final_predictions={summary.final_pred_frames}, fallback_recoveries={summary.fallback_recoveries}, "
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

    evaluation_summary = {"status": "skipped", "reason": "annotations not found"}
    if annotations_path is not None and annotations_path.exists():
        raw_metrics = evaluate_detections(
            detections=all_detections,
            annotations_path=annotations_path,
            stage="raw",
            iou_threshold=args.match_iou,
            score_threshold=args.eval_score_threshold,
            ball_category_id=args.ball_category_id,
        )
        final_metrics = evaluate_detections(
            detections=all_detections,
            annotations_path=annotations_path,
            stage="final",
            iou_threshold=args.match_iou,
            score_threshold=args.eval_score_threshold,
            ball_category_id=args.ball_category_id,
        )
        raw_by_sequence = {row["sequence"]: row for row in raw_metrics["per_sequence"]}
        final_by_sequence = {row["sequence"]: row for row in final_metrics["per_sequence"]}
        for summary in summaries:
            raw_row = raw_by_sequence.get(summary.sequence, {})
            final_row = final_by_sequence.get(summary.sequence, {})
            summary.gt_frames = int(raw_row.get("gt_count", 0.0))
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
    write_benchmark_metrics_csv(
        run_dir / "benchmark_metrics.csv",
        precision=aggregate.get("final_precision", 0.0),
        recall=aggregate.get("final_recall", 0.0),
        latency_ms=aggregate.get("total_ms_avg", 0.0),
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
    with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_summary, handle, indent=2)

    print(f"[INFO] Sequence summary written to {run_dir / 'sequence_summary.csv'}")
    print(f"[INFO] Benchmark metrics written to {run_dir / 'benchmark_metrics.csv'}")
    print(f"[INFO] Experiment summary written to {run_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
