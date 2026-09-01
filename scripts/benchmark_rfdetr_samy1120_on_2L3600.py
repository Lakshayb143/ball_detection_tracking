#!/usr/bin/env python3
"""
Benchmark ball_samy_1120 (ball-only model) on the 2-label dataset (ball + ball_out).

Standard 4-metric eval is run against ball GT (category_id=0).
An extra column fp_ball_out counts how many frames had the top prediction matching
a ball_out GT box (category_id=1) at IoU >= eval_iou, showing what share of FPs
are caused by the model firing on out-of-field balls rather than random noise.
fp_ball_out is a subset of fp_total and is included in it.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _eval_rfdetr_ball_only_common import RFDetrBallOnlyDetector  # noqa: E402
from ball_detection_metrics import (  # noqa: E402
    build_detection_record,
    count_predictions_matched_by_category,
    evaluate_detections,
    infer_category_id_by_name,
    safe_div,
    write_detection_export,
)
from benchmark_dataset import default_annotation_path, resolve_sequences  # noqa: E402

DEFAULT_DATA_ROOT = REPO_ROOT / "ball_dataset_2L_3600" / "all"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "samy1120_on_2L3600"
DEFAULT_MODEL_PATH = REPO_ROOT / "checkpoints" / "ball_samy_1120.pth"
DEFAULT_RESOLUTION = 1120


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval ball_samy_1120 on 2-label dataset with ball_out FP breakdown."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default="samy1120_on_2L3600")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--confidence_threshold", type=float, default=0.01)
    parser.add_argument("--eval_iou", type=float, default=0.01)
    parser.add_argument("--score_threshold", type=float, default=0.01)
    parser.add_argument("--max_frames", type=int, default=0, help="Cap frames per sequence (0 = no limit)")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = default_annotation_path(data_root)
    if annotations_path is None:
        raise FileNotFoundError(f"No _annotations.coco.json found under {data_root}")

    ball_cat_id = infer_category_id_by_name(annotations_path, "ball", fallback_category_id=0)
    ball_out_cat_id = infer_category_id_by_name(annotations_path, "ball_out", fallback_category_id=1)

    sequences = resolve_sequences(
        data_root=data_root,
        seq_start=0,
        seq_end=999,
        seq_list="",
        max_frames_per_seq=args.max_frames,
        annotations_path=annotations_path,
    )

    print(f"[INFO] Data root:         {data_root}")
    print(f"[INFO] Annotations:       {annotations_path}")
    print(f"[INFO] ball cat_id:       {ball_cat_id}  |  ball_out cat_id: {ball_out_cat_id}")
    print(f"[INFO] Model:             {model_path}")
    print(f"[INFO] Sequences:         {[s.name for s in sequences]}")
    print(f"[INFO] Output dir:        {run_dir}")

    detector = RFDetrBallOnlyDetector(
        model_path=model_path,
        ball_class_id=0,
        confidence_threshold=args.confidence_threshold,
        resolution=args.resolution,
        optimize_for_inference=True,
    )

    all_detections: List[dict] = []
    processed_image_ids: List[int] = []
    total_frames = 0
    total_inference_ms = 0.0

    try:
        import torch
        inference_ctx = torch.inference_mode()
    except Exception:
        inference_ctx = contextlib.nullcontext()

    with inference_ctx:
        for sequence in sequences:
            seq_dets = 0
            for frame_index, image_path in enumerate(sequence.image_paths, start=1):
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Could not read {image_path}")

                image_id = sequence.image_ids_by_frame.get(frame_index)
                if image_id is not None:
                    processed_image_ids.append(int(image_id))

                t0 = time.perf_counter()
                detections = detector.predict_ball_detections(image)
                total_inference_ms += (time.perf_counter() - t0) * 1000.0
                total_frames += 1

                for det in detections:
                    seq_dets += 1
                    all_detections.append(
                        build_detection_record(
                            sequence=sequence.name,
                            frame_index=frame_index,
                            original_frame=sequence.original_frames_by_frame[frame_index],
                            file_name=sequence.file_names_by_frame[frame_index],
                            image_id=image_id,
                            bbox_xyxy=det["xyxy"],
                            score=det["score"],
                            stage="final",
                            source="rfdetr",
                        )
                    )

            print(f"[INFO] {sequence.name}: frames={len(sequence.image_paths)}, detections={seq_dets}")

    detections_path = run_dir / "detections.json"
    write_detection_export(
        path=detections_path,
        run_name=args.run_name,
        data_root=data_root,
        detections=all_detections,
        annotations_path=annotations_path,
    )

    # The event-style metric requires at most one ball GT per frame.
    # Find and exclude the rare frames that have multiple ball GT boxes.
    _raw_ann = json.loads(annotations_path.read_text(encoding="utf-8"))
    from collections import defaultdict as _dd
    _ball_counts: dict = _dd(int)
    for _a in _raw_ann["annotations"]:
        if int(_a["category_id"]) == ball_cat_id:
            _ball_counts[int(_a["image_id"])] += 1
    multi_gt_ids = {img_id for img_id, cnt in _ball_counts.items() if cnt > 1}
    eval_image_ids = [i for i in processed_image_ids if i not in multi_gt_ids]
    if multi_gt_ids:
        print(f"[INFO] Skipping {len(multi_gt_ids)} frame(s) with >1 ball GT (image_ids: {sorted(multi_gt_ids)})")

    # Standard eval against ball GT (category_id=ball_cat_id)
    evaluation = evaluate_detections(
        detections=all_detections,
        annotations_path=annotations_path,
        stage="final",
        iou_threshold=args.eval_iou,
        score_threshold=args.score_threshold,
        ball_category_id=ball_cat_id,
        allowed_image_ids=eval_image_ids,
    )
    agg = evaluation["aggregate"]

    # Extra: count predictions that matched ball_out GT
    fp_ball_out = count_predictions_matched_by_category(
        detections=all_detections,
        annotations_path=annotations_path,
        alt_category_id=ball_out_cat_id,
        iou_threshold=args.eval_iou,
        score_threshold=args.score_threshold,
        stage="final",
        allowed_image_ids=eval_image_ids,
    )

    tp = int(agg.get("event_tp", 0))
    missed = int(agg.get("missed_detection_count", 0))
    fp_wrong_pos = int(agg.get("false_positive_count", 0))
    no_gt_predicted = int(agg.get("no_gt_predicted", 0))
    fp_total = fp_wrong_pos + no_gt_predicted
    tn = int(agg.get("tn", 0))
    gt_frames = int(agg.get("gt_frames", 0))
    inference_ms_avg = safe_div(total_inference_ms, total_frames)

    summary_row = {
        "run_name": args.run_name,
        "frames_total": total_frames,
        "gt_ball_frames": gt_frames,
        "tp": tp,
        "missed": missed,
        "fp_total": fp_total,
        "fp_ball_out": fp_ball_out,
        "fp_noise": fp_total - fp_ball_out,
        "tn": tn,
        "precision": round(float(agg.get("precision", 0.0)), 4),
        "recall": round(float(agg.get("recall", 0.0)), 4),
        "iou_threshold": args.eval_iou,
        "score_threshold": args.score_threshold,
        "inference_ms_avg": round(inference_ms_avg, 2),
    }

    summary_path = run_dir / "summary.csv"
    _write_csv(summary_path, [summary_row])

    experiment = {
        "run_name": args.run_name,
        "data_root": str(data_root),
        "annotations_path": str(annotations_path),
        "model_path": str(model_path),
        "ball_cat_id": ball_cat_id,
        "ball_out_cat_id": ball_out_cat_id,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "summary": summary_row,
        "per_sequence": evaluation.get("per_sequence", []),
    }
    with (run_dir / "experiment.json").open("w", encoding="utf-8") as fh:
        json.dump(experiment, fh, indent=2)

    print(f"\n{'='*60}")
    print(f"  frames_total  : {total_frames}")
    print(f"  gt_ball_frames: {gt_frames}")
    print(f"  tp            : {tp}")
    print(f"  missed        : {missed}")
    print(f"  fp_total      : {fp_total}  (false_positive_count + no_gt_predicted)")
    print(f"  fp_ball_out   : {fp_ball_out}  <- predictions that matched ball_out GT (IoU>={args.eval_iou})")
    print(f"  fp_noise      : {fp_total - fp_ball_out}  <- predictions with no GT match at all")
    print(f"  tn            : {tn}")
    print(f"  precision     : {float(agg.get('precision', 0.0)):.4f}")
    print(f"  recall        : {float(agg.get('recall', 0.0)):.4f}")
    print(f"  inference_ms  : {inference_ms_avg:.2f}")
    print(f"{'='*60}")
    print(f"[INFO] Summary CSV  : {summary_path}")
    print(f"[INFO] Detections   : {detections_path}")
    print(f"[INFO] Experiment   : {run_dir / 'experiment.json'}")


if __name__ == "__main__":
    main()
