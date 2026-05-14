#!/usr/bin/env python3
"""
SST-only benchmark runner for SoccerNet SNMOT ball tracking.

This mirrors the GroundingDINO experiment structure so the outputs are directly
comparable across experiments, while using the SST detector as the per-frame
ball detector.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TVF

from ball_detection_metrics import (
    BALL_CATEGORY_ID,
    evaluate_detections,
    write_benchmark_metrics_csv,
    write_detection_export,
)
from benchmark_dataset import default_annotation_path

REPO_ROOT = Path(__file__).resolve().parents[1]
SST_SRC_ROOT = REPO_ROOT / "SST" / "src"
if str(SST_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SST_SRC_ROOT))

from models import AnchorGenerator, fasterrcnn_resnet50_fpn  # type: ignore  # noqa: E402

from benchmark_groundingdino_ball_tracking import (  # noqa: E402
    DEFAULT_BASE_GATE_DISTANCE,
    DEFAULT_BOX_EMA,
    DEFAULT_BRIDGE_SCORE_DECAY,
    DEFAULT_CROP_EXPANSION,
    DEFAULT_ENABLE_CROP_REDETECT,
    DEFAULT_EVAL_SCORE_THRESHOLD,
    DEFAULT_GATE_GROWTH_PER_MISS,
    DEFAULT_INIT_CONFIDENCE,
    DEFAULT_MATCH_IOU,
    DEFAULT_MAX_ACTIVE_GAP,
    DEFAULT_MAX_CROP_SIZE,
    DEFAULT_MAX_FRAMES_PER_SEQ,
    DEFAULT_MAX_PREDICT_GAP,
    DEFAULT_MIN_BRIDGE_SCORE,
    DEFAULT_MIN_CROP_SIZE,
    DEFAULT_SEQ_END,
    DEFAULT_SEQ_LIST,
    DEFAULT_SEQ_START,
    DEFAULT_DATA_ROOT,
    DetectionCandidate,
    SequenceSummary,
    aggregate_sequence_summaries,
    area_of,
    clip_xyxy,
    csv_write_dicts,
    ensure_dir,
    process_sequence,
    resolve_sequences,
    str2bool,
    xyxy_wh,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "sst_ball_tracking"
DEFAULT_RUN_NAME = "sst_gap_bridge_v1"

DEFAULT_SST_CHECKPOINT = REPO_ROOT / "checkpoints" / "sst_checkpoint.pth"
DEFAULT_SST_NUM_CLASSES = 7
DEFAULT_SST_BALL_CLASS_ID = 1
DEFAULT_SST_SCORE_THRESHOLD = 0.20
DEFAULT_SST_MIN_BOX_AREA = 4.0
DEFAULT_SST_MAX_BOX_AREA = 6000.0
DEFAULT_SST_MIN_BOX_SIDE = 2.0
DEFAULT_SST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SST_USE_AMP = True
DEFAULT_SST_BOX_DETECTIONS_PER_IMG = 100
DEFAULT_SST_PRETRAINED_BACKBONE = False


def build_default_anchor_generator() -> AnchorGenerator:
    scales = tuple((value * 0.337, value * 0.517, value * 1.939) for value in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.289, 1.0, 3.458),) * len(scales)
    return AnchorGenerator(scales, aspect_ratios)


class SSTBallDetector:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str,
        num_classes: int,
        ball_class_id: int,
        score_threshold: float,
        min_box_area: float,
        max_box_area: float,
        min_box_side: float,
        detections_per_img: int,
        pretrained_backbone: bool,
        use_amp: bool,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.ball_class_id = ball_class_id
        self.score_threshold = score_threshold
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.min_box_side = min_box_side
        self.use_amp = use_amp and self.device.type == "cuda"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SST checkpoint not found: {checkpoint_path}")

        checkpoint = self._load_checkpoint(checkpoint_path)
        anchor_generator = checkpoint.get("anchor_generator", build_default_anchor_generator())

        model = fasterrcnn_resnet50_fpn(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            rpn_anchor_generator=anchor_generator,
            box_detections_per_img=detections_per_img,
            box_score_thresh=max(1e-6, min(score_threshold, 0.99)),
        )

        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(self.device).eval()

    @staticmethod
    def _load_checkpoint(checkpoint_path: Path) -> dict:
        try:
            return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(checkpoint_path), map_location="cpu")

    def _prepare_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return TVF.to_tensor(image_rgb)

    def predict_candidates(self, image_bgr: np.ndarray, source: str = "detector") -> List[DetectionCandidate]:
        height, width = image_bgr.shape[:2]
        image_tensor = self._prepare_tensor(image_bgr).to(self.device)

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model([image_tensor])[0]
            else:
                outputs = self.model([image_tensor])[0]

        boxes = outputs["boxes"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()

        candidates: List[DetectionCandidate] = []
        for box, score, label in zip(boxes, scores, labels):
            if int(label) != self.ball_class_id:
                continue
            if float(score) < self.score_threshold:
                continue

            xyxy = clip_xyxy(np.asarray(box, dtype=np.float32), width=width, height=height)
            wh = xyxy_wh(xyxy)
            area = area_of(xyxy)
            if wh[0] < self.min_box_side or wh[1] < self.min_box_side:
                continue
            if area < self.min_box_area or area > self.max_box_area:
                continue

            candidates.append(
                DetectionCandidate(
                    xyxy=xyxy,
                    score=float(score),
                    phrase="ball",
                    source=source,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def predict_candidates_in_crop(self, image_bgr: np.ndarray, crop_xyxy: np.ndarray) -> List[DetectionCandidate]:
        x1, y1, x2, y2 = [int(round(value)) for value in crop_xyxy]
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        candidates = self.predict_candidates(crop, source="crop")
        translated: List[DetectionCandidate] = []
        for item in candidates:
            translated_box = item.xyxy.copy()
            translated_box[[0, 2]] += x1
            translated_box[[1, 3]] += y1
            translated.append(
                DetectionCandidate(
                    xyxy=translated_box,
                    score=item.score,
                    phrase=item.phrase,
                    source=item.source,
                )
            )
        translated.sort(key=lambda item: item.score, reverse=True)
        return translated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SST-only benchmark runner for SoccerNet SNMOT ball tracking."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--seq_start", type=int, default=DEFAULT_SEQ_START)
    parser.add_argument("--seq_end", type=int, default=DEFAULT_SEQ_END)
    parser.add_argument("--seq_list", type=str, default=DEFAULT_SEQ_LIST)
    parser.add_argument("--max_frames_per_seq", type=int, default=DEFAULT_MAX_FRAMES_PER_SEQ)

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

    parser.add_argument("--init_confidence", type=float, default=DEFAULT_INIT_CONFIDENCE)
    parser.add_argument("--base_gate_distance", type=float, default=DEFAULT_BASE_GATE_DISTANCE)
    parser.add_argument("--gate_growth_per_miss", type=float, default=DEFAULT_GATE_GROWTH_PER_MISS)
    parser.add_argument("--enable_crop_redetect", type=str2bool, default=DEFAULT_ENABLE_CROP_REDETECT)
    parser.add_argument("--crop_expansion", type=float, default=DEFAULT_CROP_EXPANSION)
    parser.add_argument("--min_crop_size", type=int, default=DEFAULT_MIN_CROP_SIZE)
    parser.add_argument("--max_crop_size", type=int, default=DEFAULT_MAX_CROP_SIZE)
    parser.add_argument("--max_predict_gap", type=int, default=DEFAULT_MAX_PREDICT_GAP)
    parser.add_argument("--max_active_gap", type=int, default=DEFAULT_MAX_ACTIVE_GAP)
    parser.add_argument("--box_ema", type=float, default=DEFAULT_BOX_EMA)
    parser.add_argument("--bridge_score_decay", type=float, default=DEFAULT_BRIDGE_SCORE_DECAY)
    parser.add_argument("--min_bridge_score", type=float, default=DEFAULT_MIN_BRIDGE_SCORE)

    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--match_iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--eval_score_threshold", type=float, default=DEFAULT_EVAL_SCORE_THRESHOLD)
    parser.add_argument("--ball_category_id", type=int, default=BALL_CATEGORY_ID)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.sst_checkpoint = args.sst_checkpoint.expanduser().resolve()
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
    print(f"[INFO] Sequences: {', '.join(sequence.name for sequence in sequences)}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Crop re-detect enabled: {args.enable_crop_redetect}")
    if annotations_path is not None:
        print(f"[INFO] Evaluation annotations: {annotations_path}")

    detector = SSTBallDetector(
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
    all_detections: List[dict] = []
    for sequence in sequences:
        print(f"[INFO] Processing {sequence.name}")
        summary, _prediction_path, sequence_detections = process_sequence(sequence, detector, args, run_dir)
        summaries.append(summary)
        all_detections.extend(sequence_detections)
        print(
            f"[INFO] {sequence.name}: raw_predictions={summary.raw_pred_frames}, "
            f"final_predictions={summary.final_pred_frames}, crop_recoveries={summary.crop_recoveries}, "
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
