#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import ransacp  # noqa: E402
from ball_detection_metrics import (  # noqa: E402
    build_detection_record,
    evaluate_detections,
    infer_category_id_by_name,
    write_benchmark_metrics_csv,
    write_detection_export,
)
from distance_rule_adjustment import (  # noqa: E402
    DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
    attach_distance_rule_summary,
    resolve_center_distance_bucket_threshold_px,
    write_dual_stage_distance_rule_metrics_csv,
)


FRAME_PATTERN = re.compile(r"frame_(\d+)", re.IGNORECASE)
DEFAULT_RUN_NAME = "physics_ransac__clip1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "clip1_fresh_runs"
DEFAULT_DATA_ROOT = REPO_ROOT / "train"
DEFAULT_ANNOTATIONS = DEFAULT_DATA_ROOT / "_annotations.coco.json"
DEFAULT_BOX_SIZE = 20.0


@dataclass
class FrameRecord:
    frame_idx: int
    coco_frame: int
    image_id: Optional[int]
    raw_predicted: bool
    raw_score: float
    in_ransac_fit: bool
    segment: str
    segment_physics_ok: Optional[bool]
    ransac_inlier: Optional[bool]
    ransac_residual_px: Optional[float]
    final_kept: bool
    final_source: str


@dataclass
class SegmentRecord:
    segment: str
    detections: int
    a: Optional[float]
    inlier_ratio: Optional[float]
    inliers: int
    outliers: int
    physics_ok: Optional[bool]
    verdict: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ransacp.py physics-RANSAC filtering on clip1 tracker outputs "
            "against the train COCO ground truth."
        )
    )
    parser.add_argument("--tracker_json", type=Path, default=Path(ransacp.TRACKER_OUTPUT_PATH))
    parser.add_argument("--airborne_json", type=Path, default=Path(ransacp.AIRBORNE_SEGS_PATH))
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--frame_index_offset",
        type=int,
        default=1,
        help="COCO frame number = tracker frame_idx + offset. clip1 tracker JSON is zero-based, so default is 1.",
    )
    parser.add_argument(
        "--box_size",
        type=float,
        default=DEFAULT_BOX_SIZE,
        help="Fixed box size used when the tracker JSON only contains center points.",
    )
    parser.add_argument("--match_iou", type=float, default=0.01)
    parser.add_argument("--eval_score_threshold", type=float, default=0.0)
    parser.add_argument("--ball_category_id", type=int, default=None)
    parser.add_argument(
        "--online_lookahead_frames",
        type=int,
        default=0,
        help="If >0, evaluate delayed online RANSAC using this many future frames.",
    )
    parser.add_argument(
        "--online_history_frames",
        type=int,
        default=ransacp.ONLINE_HISTORY_FRAMES,
        help="Past frames used by delayed online RANSAC; 0 means all segment history so far.",
    )
    parser.add_argument(
        "--bad_segment_policy",
        choices=["keep_inliers", "drop_all", "keep_raw"],
        default="keep_inliers",
        help=(
            "How to handle PHYSICS_BAD segments: keep per-frame inliers, drop the whole "
            "segment, or leave raw detections unchanged in that segment."
        ),
    )
    parser.add_argument(
        "--drop_outside_segments",
        action="store_true",
        help="Only keep RANSAC-kept detections; by default frames outside fitted airborne segments keep raw output.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_rows_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def frame_num_from_file_name(file_name: str) -> Optional[int]:
    match = FRAME_PATTERN.search(str(file_name))
    return int(match.group(1)) if match else None


def load_coco_frame_map(annotations_path: Path) -> Dict[int, dict]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    frame_map: Dict[int, dict] = {}
    for image in payload.get("images", []):
        frame_num = frame_num_from_file_name(str(image.get("file_name", "")))
        if frame_num is None:
            continue
        frame_map[frame_num] = {
            "image_id": int(image["id"]),
            "file_name": str(image["file_name"]),
            "width": int(image.get("width", 0) or 0),
            "height": int(image.get("height", 0) or 0),
        }
    if not frame_map:
        raise RuntimeError(f"Could not map any COCO images by frame number in {annotations_path}")
    return frame_map


def load_tracker_scores(path: Path) -> Dict[int, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scores: Dict[int, float] = {}

    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = enumerate(raw)
    else:
        return scores

    for key, value in items:
        if value is None:
            continue
        if isinstance(value, dict):
            frame_idx = int(value.get("frame_idx", value.get("frame", key)))
            score = value.get("confidence", value.get("score", 1.0))
        else:
            frame_idx = int(key)
            score = value[4] if len(value) > 4 else 1.0
        if score is not None:
            scores[frame_idx] = float(score)
    return scores


def clipped_center_box(
    cx: float,
    cy: float,
    width: float,
    height: float,
    fallback_box_size: float,
    image_width: int,
    image_height: int,
) -> List[float]:
    box_w = float(width) if float(width) > 0.0 else float(fallback_box_size)
    box_h = float(height) if float(height) > 0.0 else float(fallback_box_size)
    x1 = float(cx) - box_w / 2.0
    y1 = float(cy) - box_h / 2.0
    x2 = float(cx) + box_w / 2.0
    y2 = float(cy) + box_h / 2.0
    if image_width > 0:
        x1 = max(0.0, min(float(image_width), x1))
        x2 = max(0.0, min(float(image_width), x2))
    if image_height > 0:
        y1 = max(0.0, min(float(image_height), y1))
        y2 = max(0.0, min(float(image_height), y2))
    return [x1, y1, x2, y2]


def build_fit_decisions(
    segments: Sequence[Tuple[int, int]],
    tracker_data: Dict[int, Tuple[float, float, float, float]],
    bad_segment_policy: str,
) -> Tuple[Dict[int, dict], List[SegmentRecord]]:
    decisions: Dict[int, dict] = {}
    segment_records: List[SegmentRecord] = []

    for seg_start, seg_end in segments:
        fit = ransacp.process_segment(seg_start, seg_end, tracker_data)
        segment_name = f"{seg_start}-{seg_end}"
        if fit is None:
            segment_records.append(
                SegmentRecord(
                    segment=segment_name,
                    detections=0,
                    a=None,
                    inlier_ratio=None,
                    inliers=0,
                    outliers=0,
                    physics_ok=None,
                    verdict="SKIPPED",
                )
            )
            continue

        inlier_mask = fit["inlier_mask"]
        residuals = fit["residuals"]
        n_inliers = int(inlier_mask.sum())
        n_total = len(fit["frames"])
        segment_records.append(
            SegmentRecord(
                segment=segment_name,
                detections=n_total,
                a=float(fit["a"]),
                inlier_ratio=float(fit["inlier_ratio"]),
                inliers=n_inliers,
                outliers=n_total - n_inliers,
                physics_ok=bool(fit["physics_ok"]),
                verdict="PHYSICS_OK" if fit["physics_ok"] else "PHYSICS_BAD",
            )
        )

        for index, frame_idx in enumerate(fit["frames"]):
            is_inlier = bool(inlier_mask[index])
            if bad_segment_policy == "drop_all" and not fit["physics_ok"]:
                keep = False
            elif bad_segment_policy == "keep_raw" and not fit["physics_ok"]:
                keep = True
            else:
                keep = is_inlier

            decisions[int(frame_idx)] = {
                "keep": keep,
                "inlier": is_inlier,
                "residual": float(residuals[index]),
                "segment": segment_name,
                "physics_ok": bool(fit["physics_ok"]),
            }

    return decisions, segment_records


def build_online_fit_decisions(
    segments: Sequence[Tuple[int, int]],
    tracker_data: Dict[int, Tuple[float, float, float, float]],
    lookahead_frames: int,
    history_frames: int,
    bad_segment_policy: str,
) -> Tuple[Dict[int, dict], List[SegmentRecord]]:
    raw_decisions, _pending_frames = ransacp.build_online_delayed_decisions(
        segments=segments,
        tracker_data=tracker_data,
        lookahead_frames=int(lookahead_frames),
        history_frames=int(history_frames),
    )

    decisions: Dict[int, dict] = {}
    segment_records: List[SegmentRecord] = []
    for seg_start, seg_end in segments:
        segment_name = f"{seg_start}-{seg_end}"
        seg_decisions = [
            (frame_idx, decision)
            for frame_idx, decision in raw_decisions.items()
            if seg_start <= frame_idx <= seg_end
        ]
        n_inliers = sum(1 for _, decision in seg_decisions if decision["is_inlier"])
        n_outliers = sum(1 for _, decision in seg_decisions if not decision["is_inlier"])
        segment_records.append(
            SegmentRecord(
                segment=segment_name,
                detections=len(seg_decisions),
                a=None,
                inlier_ratio=(float(n_inliers) / len(seg_decisions)) if seg_decisions else None,
                inliers=n_inliers,
                outliers=n_outliers,
                physics_ok=None,
                verdict=f"ONLINE_DELAYED_+{int(lookahead_frames)}",
            )
        )

        for frame_idx, decision in seg_decisions:
            fit = decision["fit"]
            is_inlier = bool(decision["is_inlier"])
            if bad_segment_policy == "drop_all" and not fit["physics_ok"]:
                keep = False
            elif bad_segment_policy == "keep_raw" and not fit["physics_ok"]:
                keep = True
            else:
                keep = is_inlier

            decisions[int(frame_idx)] = {
                "keep": keep,
                "inlier": is_inlier,
                "residual": float(decision["residual"]),
                "segment": segment_name,
                "physics_ok": bool(fit["physics_ok"]),
                "decision_ready_frame": int(decision["decision_ready_frame"]),
                "fit_start": int(fit["fit_start"]),
                "fit_end": int(fit["fit_end"]),
            }

    return decisions, segment_records


def add_detection(
    detections: List[dict],
    *,
    stage: str,
    source: str,
    frame_idx: int,
    coco_frame: int,
    image: dict,
    xywh_record: Tuple[float, float, float, float],
    score: float,
    box_size: float,
) -> None:
    x, y, w, h = xywh_record
    cx = float(x) + float(w) / 2.0
    cy = float(y) + float(h) / 2.0
    bbox_xyxy = clipped_center_box(
        cx=cx,
        cy=cy,
        width=float(w),
        height=float(h),
        fallback_box_size=float(box_size),
        image_width=int(image.get("width", 0) or 0),
        image_height=int(image.get("height", 0) or 0),
    )
    detections.append(
        build_detection_record(
            sequence="FLAT-COCO",
            frame_index=int(coco_frame),
            original_frame=int(frame_idx),
            file_name=str(image["file_name"]),
            image_id=int(image["image_id"]),
            bbox_xyxy=bbox_xyxy,
            score=float(score),
            stage=stage,
            source=source,
        )
    )


def original_metric_csv(path: Path, latency_ms: float, match_iou: float, score_threshold: float, raw: dict, final: dict) -> None:
    write_benchmark_metrics_csv(
        path,
        latency_ms=float(latency_ms),
        iou_threshold=float(match_iou),
        score_threshold=float(score_threshold),
        raw_tp=int(raw.get("event_tp", 0.0)),
        raw_missed_detection_count=int(raw.get("missed_detection_count", 0.0)),
        raw_false_positive_count=int(raw.get("false_positive_count", 0.0)),
        raw_tn=int(raw.get("tn", 0.0)),
        raw_no_gt_predicted=int(raw.get("no_gt_predicted", 0.0)),
        raw_avg_false_positive_center_distance_px=float(raw.get("avg_false_positive_center_distance_px", 0.0)),
        final_tp=int(final.get("event_tp", 0.0)),
        final_missed_detection_count=int(final.get("missed_detection_count", 0.0)),
        final_false_positive_count=int(final.get("false_positive_count", 0.0)),
        final_tn=int(final.get("tn", 0.0)),
        final_no_gt_predicted=int(final.get("no_gt_predicted", 0.0)),
        final_avg_false_positive_center_distance_px=float(final.get("avg_false_positive_center_distance_px", 0.0)),
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    args.tracker_json = args.tracker_json.expanduser().resolve()
    args.airborne_json = args.airborne_json.expanduser().resolve()
    args.annotations = args.annotations.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.ball_category_id = (
        int(args.ball_category_id)
        if args.ball_category_id is not None
        else infer_category_id_by_name(args.annotations, "ball", fallback_category_id=1)
    )
    if args.ball_category_id is None:
        raise RuntimeError(f"Could not infer ball category id from {args.annotations}")

    run_dir = ensure_dir(args.output_root / args.run_name)
    frame_map = load_coco_frame_map(args.annotations)
    tracker_data = ransacp.load_tracker_output(args.tracker_json)
    tracker_scores = load_tracker_scores(args.tracker_json)
    segments = ransacp.load_airborne_segments(args.airborne_json)
    if int(args.online_lookahead_frames) > 0:
        decisions, segment_records = build_online_fit_decisions(
            segments=segments,
            tracker_data=tracker_data,
            lookahead_frames=int(args.online_lookahead_frames),
            history_frames=int(args.online_history_frames),
            bad_segment_policy=args.bad_segment_policy,
        )
    else:
        decisions, segment_records = build_fit_decisions(
            segments=segments,
            tracker_data=tracker_data,
            bad_segment_policy=args.bad_segment_policy,
        )

    detections: List[dict] = []
    point_outputs: List[dict] = []
    frame_records: List[FrameRecord] = []

    for frame_idx in sorted(tracker_data):
        coco_frame = int(frame_idx) + int(args.frame_index_offset)
        image = frame_map.get(coco_frame)
        if image is None:
            continue

        score = tracker_scores.get(frame_idx, 1.0)
        raw_record = tracker_data[frame_idx]
        add_detection(
            detections,
            stage="raw",
            source="tracker",
            frame_idx=frame_idx,
            coco_frame=coco_frame,
            image=image,
            xywh_record=raw_record,
            score=score,
            box_size=args.box_size,
        )

        decision = decisions.get(frame_idx)
        if decision is None:
            final_kept = not bool(args.drop_outside_segments)
            final_source = "tracker_outside_ransac_segment" if final_kept else "dropped_outside_ransac_segment"
        else:
            final_kept = bool(decision["keep"])
            final_source = "physics_ransac_inlier" if final_kept else "physics_ransac_outlier"

        if final_kept:
            add_detection(
                detections,
                stage="final",
                source=final_source,
                frame_idx=frame_idx,
                coco_frame=coco_frame,
                image=image,
                xywh_record=raw_record,
                score=score,
                box_size=args.box_size,
            )

        x, y, w, h = raw_record
        point_outputs.append(
            {
                "frame_idx": int(frame_idx),
                "coco_frame": int(coco_frame),
                "x": float(x) + float(w) / 2.0,
                "y": float(y) + float(h) / 2.0,
                "confidence": float(score),
                "source": final_source,
                "kept_after_ransac": bool(final_kept),
            }
        )
        frame_records.append(
            FrameRecord(
                frame_idx=int(frame_idx),
                coco_frame=int(coco_frame),
                image_id=int(image["image_id"]),
                raw_predicted=True,
                raw_score=float(score),
                in_ransac_fit=decision is not None,
                segment=str(decision["segment"]) if decision is not None else "",
                segment_physics_ok=bool(decision["physics_ok"]) if decision is not None else None,
                ransac_inlier=bool(decision["inlier"]) if decision is not None else None,
                ransac_residual_px=float(decision["residual"]) if decision is not None else None,
                final_kept=bool(final_kept),
                final_source=final_source,
            )
        )

    raw_metrics = evaluate_detections(
        detections=detections,
        annotations_path=args.annotations,
        stage="raw",
        iou_threshold=args.match_iou,
        score_threshold=args.eval_score_threshold,
        ball_category_id=int(args.ball_category_id),
    )
    final_metrics = evaluate_detections(
        detections=detections,
        annotations_path=args.annotations,
        stage="final",
        iou_threshold=args.match_iou,
        score_threshold=args.eval_score_threshold,
        ball_category_id=int(args.ball_category_id),
    )
    latency_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(frame_map))

    detections_path = run_dir / "detections.json"
    write_detection_export(
        path=detections_path,
        run_name=args.run_name,
        data_root=args.data_root,
        detections=detections,
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        annotations_path=args.annotations,
    )
    (run_dir / "point_outputs.json").write_text(json.dumps(point_outputs, indent=2), encoding="utf-8")
    write_rows_csv(run_dir / "frame_trace.csv", [asdict(record) for record in frame_records])
    write_rows_csv(run_dir / "segment_summary.csv", [asdict(record) for record in segment_records])

    raw_aggregate = raw_metrics["aggregate"]
    final_aggregate = final_metrics["aggregate"]
    experiment_summary = {
        "run_name": args.run_name,
        "data_root": str(args.data_root),
        "tracker_json": str(args.tracker_json),
        "airborne_json": str(args.airborne_json),
        "detections_path": str(detections_path),
        "point_outputs_path": str(run_dir / "point_outputs.json"),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "segments": [asdict(record) for record in segment_records],
        "counts": {
            "annotated_frames": len(frame_map),
            "raw_predictions": int(sum(1 for item in detections if item["stage"] == "raw")),
            "final_predictions": int(sum(1 for item in detections if item["stage"] == "final")),
            "ransac_inliers": int(sum(record.inliers for record in segment_records)),
            "ransac_outliers": int(sum(record.outliers for record in segment_records)),
        },
        "evaluation": {
            "status": "ok",
            "annotations_path": str(args.annotations),
            "ball_category_id": int(args.ball_category_id),
            "iou_threshold": float(args.match_iou),
            "score_threshold": float(args.eval_score_threshold),
            "raw": raw_metrics,
            "final": final_metrics,
        },
    }
    center_distance_bucket_threshold_px = resolve_center_distance_bucket_threshold_px(
        annotations_path=args.annotations,
        ball_category_id=int(args.ball_category_id),
        fallback_threshold_px=DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
    )
    distance_summary = attach_distance_rule_summary(
        experiment_summary,
        center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
        include_raw_stage=True,
    )

    write_dual_stage_distance_rule_metrics_csv(
        run_dir / "benchmark_metrics.csv",
        latency_ms=latency_ms,
        iou_threshold=float(args.match_iou),
        score_threshold=float(args.eval_score_threshold),
        raw_aggregate=raw_aggregate,
        final_aggregate=final_aggregate,
        center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
    )
    write_dual_stage_distance_rule_metrics_csv(
        args.output_root / f"{args.run_name}_benchmark_metrics.csv",
        latency_ms=latency_ms,
        iou_threshold=float(args.match_iou),
        score_threshold=float(args.eval_score_threshold),
        raw_aggregate=raw_aggregate,
        final_aggregate=final_aggregate,
        center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
    )
    original_metric_csv(
        run_dir / "benchmark_metrics_original.csv",
        latency_ms,
        args.match_iou,
        args.eval_score_threshold,
        raw_aggregate,
        final_aggregate,
    )
    original_metric_csv(
        args.output_root / f"{args.run_name}_benchmark_metrics_original.csv",
        latency_ms,
        args.match_iou,
        args.eval_score_threshold,
        raw_aggregate,
        final_aggregate,
    )

    (run_dir / "experiment_summary.json").write_text(
        json.dumps(experiment_summary, indent=2),
        encoding="utf-8",
    )

    final_adjusted = distance_summary.get("final", {})
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Detections written to {detections_path}")
    print(
        "[INFO] Raw original metric: "
        f"tp={int(raw_aggregate.get('event_tp', 0))}, "
        f"missed={int(raw_aggregate.get('missed_detection_count', 0))}, "
        f"fp={int(raw_aggregate.get('false_positive_count', 0))}, "
        f"tn={int(raw_aggregate.get('tn', 0))}"
    )
    print(
        "[INFO] Final original metric: "
        f"tp={int(final_aggregate.get('event_tp', 0))}, "
        f"missed={int(final_aggregate.get('missed_detection_count', 0))}, "
        f"fp={int(final_aggregate.get('false_positive_count', 0))}, "
        f"tn={int(final_aggregate.get('tn', 0))}"
    )
    if isinstance(final_adjusted, dict):
        print(
            "[INFO] Final adjusted metric: "
            f"tp={int(final_adjusted.get('tp', 0))}, "
            f"missed={int(final_adjusted.get('missed_detection_count', 0))}, "
            f"fp={int(final_adjusted.get('false_positive_count', 0))}, "
            f"distance_le_threshold_px_count={int(final_adjusted.get('distance_le_threshold_px_count', 0))}"
        )
    print(f"[INFO] Benchmark metrics written to {run_dir / 'benchmark_metrics.csv'}")


if __name__ == "__main__":
    main()
