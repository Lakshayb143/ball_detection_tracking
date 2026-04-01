#!/usr/bin/env python3
"""
RF-DETR ball-only + outlier rejection + SST fallback benchmark runner.

This mirrors the RF-DETR outlier-only + GroundingDINO fallback experiment, but
uses SST as the fallback detector.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import time

from benchmark_rfdetr_outlier_gdino_fallback_tracking import (
    DEFAULT_BALL_CLASS_ID,
    DEFAULT_BALL_CONFIDENCE,
    DEFAULT_DATA_ROOT,
    DEFAULT_ENABLE_RFDETR_OPTIMIZE,
    DEFAULT_MATCH_IOU,
    DEFAULT_MAX_ACTIVE_GAP,
    DEFAULT_MAX_FRAMES_PER_SEQ,
    DEFAULT_OUTLIER_HISTORY_FRAMES,
    DEFAULT_OUTLIER_RESET_FRAMES,
    DEFAULT_OUTLIER_WAIT_FRAMES,
    DEFAULT_POSITION_THRESHOLD,
    DEFAULT_SEQ_END,
    DEFAULT_SEQ_LIST,
    DEFAULT_SEQ_START,
    DEFAULT_TRACKEVAL_ROOT,
    DEFAULT_VELOCITY_THRESHOLD,
    DetectionCandidate,
    FrameRecord,
    OutlierOnlyTracker,
    RFDetrBallDetector,
    SequenceSummary,
    aggregate_sequence_summaries,
    best_iou_against_gt,
    clip_xyxy,
    csv_write_dicts,
    ensure_dir,
    evaluate_sequence,
    load_ball_gt,
    resolve_sequences,
    run_trackeval,
    str2bool,
    write_track_predictions,
)
from benchmark_sst_ball_tracking import (
    DEFAULT_SST_BALL_CLASS_ID,
    DEFAULT_SST_BOX_DETECTIONS_PER_IMG,
    DEFAULT_SST_CHECKPOINT,
    DEFAULT_SST_DEVICE,
    DEFAULT_SST_MAX_BOX_AREA,
    DEFAULT_SST_MIN_BOX_AREA,
    DEFAULT_SST_MIN_BOX_SIDE,
    DEFAULT_SST_NUM_CLASSES,
    DEFAULT_SST_PRETRAINED_BACKBONE,
    DEFAULT_SST_SCORE_THRESHOLD,
    DEFAULT_SST_USE_AMP,
    SSTBallDetector,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "rfdetr_outlier_sst_fallback_tracking"
DEFAULT_RUN_NAME = "rfdetr_outlier_sst_fallback_v1"
DEFAULT_BALL_MODEL_PATH = REPO_ROOT / "checkpoints" / "ball.pth"

DEFAULT_ENABLE_SST_FALLBACK = True


def process_sequence(
    seq_dir: Path,
    rfdetr_detector: RFDetrBallDetector,
    sst_detector: Optional[SSTBallDetector],
    args: argparse.Namespace,
    run_dir: Path,
) -> Tuple[SequenceSummary, Path]:
    seq_name = seq_dir.name
    image_dir = seq_dir / "img1"
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise FileNotFoundError(f"No frames found in {image_dir}")

    if args.max_frames_per_seq > 0:
        image_paths = image_paths[: args.max_frames_per_seq]

    gt_by_frame = load_ball_gt(seq_dir)
    tracker = OutlierOnlyTracker(args=args)

    seq_output_dir = ensure_dir(run_dir / seq_name)
    prediction_path = seq_output_dir / "tracker_predictions.txt"
    frame_trace_path = seq_output_dir / "frame_trace.csv"

    frame_records: List[FrameRecord] = []
    predictions: List[Tuple[int, object, float]] = []
    raw_pred_frames = 0
    final_pred_frames = 0
    fallback_recoveries = 0

    for frame_id, image_path in enumerate(image_paths, start=1):
        total_start = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read frame {image_path}")

        primary_start = time.perf_counter()
        raw_candidates = rfdetr_detector.predict_ball_candidates(image)
        primary_ms = (time.perf_counter() - primary_start) * 1000.0
        raw_best = raw_candidates[0] if raw_candidates else None
        raw_best_box = raw_best.xyxy if raw_best is not None else None
        if raw_best is not None:
            raw_pred_frames += 1

        tracking_start = time.perf_counter()
        accepted: Optional[DetectionCandidate] = None
        primary_selected = tracker.select_best_candidate(raw_candidates)
        if primary_selected is not None and tracker.try_accept_candidate(primary_selected):
            accepted = primary_selected

        fallback_candidates: List[DetectionCandidate] = []
        fallback_ms = 0.0
        if accepted is None and args.enable_sst_fallback and sst_detector is not None:
            fallback_start = time.perf_counter()
            fallback_candidates = sst_detector.predict_candidates(image, source="sst_fallback")
            fallback_ms = (time.perf_counter() - fallback_start) * 1000.0

            fallback_selected = tracker.select_best_candidate(fallback_candidates)
            if fallback_selected is not None:
                fallback_selected = DetectionCandidate(
                    xyxy=fallback_selected.xyxy,
                    score=fallback_selected.score,
                    phrase=fallback_selected.phrase,
                    source="sst_fallback",
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
        tracking_ms = (time.perf_counter() - tracking_start) * 1000.0

        if output_box is not None:
            output_box = clip_xyxy(output_box, width=image.shape[1], height=image.shape[0])
            predictions.append((frame_id, output_box, output_score))
            final_pred_frames += 1

        gt_boxes = gt_by_frame.get(frame_id, [])
        gt_iou_raw = best_iou_against_gt(raw_best_box, gt_boxes)
        gt_iou_final = best_iou_against_gt(output_box, gt_boxes)
        total_ms = (time.perf_counter() - total_start) * 1000.0

        frame_records.append(
            FrameRecord(
                frame=frame_id,
                gt_exists=bool(gt_boxes),
                gt_iou_raw=gt_iou_raw,
                gt_iou_final=gt_iou_final,
                raw_candidate_count=len(raw_candidates),
                fallback_candidate_count=len(fallback_candidates),
                raw_best_score=raw_best.score if raw_best is not None else 0.0,
                fallback_best_score=fallback_candidates[0].score if fallback_candidates else 0.0,
                output_score=output_score,
                output_source=output_source,
                tracker_active=tracker.track_active,
                tracker_gap=tracker.track_lost_count,
                primary_ms=primary_ms,
                fallback_ms=fallback_ms,
                tracking_ms=tracking_ms,
                total_ms=total_ms,
            )
        )

    write_track_predictions(prediction_path, predictions)
    csv_write_dicts(frame_trace_path, [asdict(record) for record in frame_records])
    summary = evaluate_sequence(
        sequence_name=seq_name,
        frame_records=frame_records,
        gt_by_frame=gt_by_frame,
        raw_pred_frames=raw_pred_frames,
        final_pred_frames=final_pred_frames,
        fallback_recoveries=fallback_recoveries,
        match_iou=args.match_iou,
    )
    return summary, prediction_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RF-DETR ball-only + outlier rejection + SST fallback benchmark runner."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--seq_start", type=int, default=DEFAULT_SEQ_START)
    parser.add_argument("--seq_end", type=int, default=DEFAULT_SEQ_END)
    parser.add_argument("--seq_list", type=str, default=DEFAULT_SEQ_LIST)
    parser.add_argument("--max_frames_per_seq", type=int, default=DEFAULT_MAX_FRAMES_PER_SEQ)

    parser.add_argument("--ball_model_path", type=Path, default=DEFAULT_BALL_MODEL_PATH)
    parser.add_argument("--ball_confidence", type=float, default=DEFAULT_BALL_CONFIDENCE)
    parser.add_argument("--ball_class_id", type=int, default=DEFAULT_BALL_CLASS_ID)
    parser.add_argument("--enable_rfdetr_optimize", type=str2bool, default=DEFAULT_ENABLE_RFDETR_OPTIMIZE)
    parser.add_argument("--max_active_gap", type=int, default=DEFAULT_MAX_ACTIVE_GAP)

    parser.add_argument("--position_threshold", type=float, default=DEFAULT_POSITION_THRESHOLD)
    parser.add_argument("--velocity_threshold", type=float, default=DEFAULT_VELOCITY_THRESHOLD)
    parser.add_argument("--outlier_history_frames", type=int, default=DEFAULT_OUTLIER_HISTORY_FRAMES)
    parser.add_argument("--outlier_wait_frames", type=int, default=DEFAULT_OUTLIER_WAIT_FRAMES)
    parser.add_argument("--outlier_reset_frames", type=int, default=DEFAULT_OUTLIER_RESET_FRAMES)

    parser.add_argument("--sst_checkpoint", type=Path, default=DEFAULT_SST_CHECKPOINT)
    parser.add_argument("--sst_num_classes", type=int, default=DEFAULT_SST_NUM_CLASSES)
    parser.add_argument("--sst_ball_class_id", type=int, default=DEFAULT_SST_BALL_CLASS_ID)
    parser.add_argument("--sst_score_threshold", type=float, default=DEFAULT_SST_SCORE_THRESHOLD)
    parser.add_argument("--sst_min_box_area", type=float, default=DEFAULT_SST_MIN_BOX_AREA)
    parser.add_argument("--sst_max_box_area", type=float, default=DEFAULT_SST_MAX_BOX_AREA)
    parser.add_argument("--sst_min_box_side", type=float, default=DEFAULT_SST_MIN_BOX_SIDE)
    parser.add_argument("--sst_detections_per_img", type=int, default=DEFAULT_SST_BOX_DETECTIONS_PER_IMG)
    parser.add_argument("--sst_pretrained_backbone", type=str2bool, default=DEFAULT_SST_PRETRAINED_BACKBONE)
    parser.add_argument("--device", type=str, default=DEFAULT_SST_DEVICE)
    parser.add_argument("--use_amp", type=str2bool, default=DEFAULT_SST_USE_AMP)
    parser.add_argument("--enable_sst_fallback", type=str2bool, default=DEFAULT_ENABLE_SST_FALLBACK)

    parser.add_argument("--match_iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--trackeval_root", type=Path, default=DEFAULT_TRACKEVAL_ROOT)
    parser.add_argument("--run_trackeval", type=str2bool, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.ball_model_path = args.ball_model_path.expanduser().resolve()
    args.sst_checkpoint = args.sst_checkpoint.expanduser().resolve()
    args.trackeval_root = args.trackeval_root.expanduser().resolve()

    run_dir = ensure_dir(args.output_root / args.run_name)
    sequence_dirs = resolve_sequences(
        data_root=args.data_root,
        seq_start=args.seq_start,
        seq_end=args.seq_end,
        seq_list=args.seq_list,
    )

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Sequences: {', '.join(seq_dir.name for seq_dir in sequence_dirs)}")
    print(f"[INFO] Device: {args.device}")
    print("[INFO] RF-DETR primary: ball model only")
    print(f"[INFO] SST fallback enabled: {args.enable_sst_fallback}")

    rfdetr_detector = RFDetrBallDetector(
        ball_model_path=args.ball_model_path,
        ball_confidence=args.ball_confidence,
        ball_class_id=args.ball_class_id,
        optimize_for_inference=args.enable_rfdetr_optimize,
    )
    sst_detector: Optional[SSTBallDetector] = None
    if args.enable_sst_fallback:
        sst_detector = SSTBallDetector(
            checkpoint_path=args.sst_checkpoint,
            device=args.device,
            num_classes=args.sst_num_classes,
            ball_class_id=args.sst_ball_class_id,
            score_threshold=args.sst_score_threshold,
            min_box_area=args.sst_min_box_area,
            max_box_area=args.sst_max_box_area,
            min_box_side=args.sst_min_box_side,
            detections_per_img=args.sst_detections_per_img,
            pretrained_backbone=args.sst_pretrained_backbone,
            use_amp=args.use_amp,
        )

    summaries: List[SequenceSummary] = []
    prediction_paths: Dict[str, Path] = {}
    for seq_dir in sequence_dirs:
        print(f"[INFO] Processing {seq_dir.name}")
        summary, prediction_path = process_sequence(seq_dir, rfdetr_detector, sst_detector, args, run_dir)
        summaries.append(summary)
        prediction_paths[seq_dir.name] = prediction_path
        print(
            f"[INFO] {seq_dir.name}: raw_recall={summary.raw_recall:.4f}, "
            f"final_recall={summary.final_recall:.4f}, gap_bridged={summary.gap_bridged_frames}, "
            f"fps={summary.runtime_fps:.2f}"
        )

    csv_write_dicts(run_dir / "sequence_summary.csv", [asdict(summary) for summary in summaries])
    aggregate = aggregate_sequence_summaries(summaries)

    trackeval_result: Dict[str, object] = {"status": "skipped", "reason": "disabled"}
    if args.run_trackeval:
        trackeval_result = run_trackeval(
            trackeval_root=args.trackeval_root,
            work_dir=ensure_dir(run_dir / "trackeval_work"),
            sequence_dirs=sequence_dirs,
            prediction_paths=prediction_paths,
        )

    experiment_summary = {
        "run_name": args.run_name,
        "data_root": str(args.data_root),
        "sequences": [seq_dir.name for seq_dir in sequence_dirs],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "aggregate": aggregate,
        "trackeval": trackeval_result,
    }
    with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_summary, handle, indent=2)

    print(f"[INFO] Sequence summary written to {run_dir / 'sequence_summary.csv'}")
    print(f"[INFO] Experiment summary written to {run_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
