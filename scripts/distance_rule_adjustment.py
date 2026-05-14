from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

from ball_detection_metrics import write_benchmark_metrics_csv


DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX = 50.0


def _as_int(value: Any) -> int:
    if value in ("", None):
        return 0
    return int(round(float(value)))


def _as_float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def _float_map_from_aggregate(aggregate: Mapping[str, Any], field_name: str) -> Dict[str, float]:
    raw = aggregate.get(field_name, {})
    if not isinstance(raw, Mapping):
        return {}
    output: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return output


def _distance_map_from_aggregate(aggregate: Mapping[str, Any]) -> Dict[str, float]:
    return _float_map_from_aggregate(aggregate, "false_positive_center_distance_by_frame_px")


def _threshold_map_from_aggregate(aggregate: Mapping[str, Any]) -> Dict[str, float]:
    return _float_map_from_aggregate(
        aggregate,
        "false_positive_center_distance_threshold_by_frame_px",
    )


def compute_average_ball_box_perimeter_px(
    annotations_path: Optional[Path],
    *,
    ball_category_id: int,
) -> Optional[float]:
    if annotations_path is None:
        return None
    annotations_path = annotations_path.expanduser().resolve()
    if not annotations_path.exists():
        return None

    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    perimeters = []
    for annotation in payload.get("annotations", []):
        if int(annotation.get("category_id", -1)) != int(ball_category_id):
            continue
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            continue
        _, _, width, height = [float(value) for value in bbox]
        if width <= 0.0 or height <= 0.0:
            continue
        perimeters.append(2.0 * (width + height))

    if not perimeters:
        return None
    return float(sum(perimeters) / len(perimeters))


def resolve_center_distance_bucket_threshold_px(
    *,
    annotations_path: Optional[Path],
    ball_category_id: int,
    fallback_threshold_px: float = DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
) -> float:
    # The current distance bucket rule uses the predicted bbox perimeter per FP
    # frame. This value is only a compatibility fallback for older aggregate
    # payloads that do not include per-frame predicted thresholds.
    _ = (annotations_path, ball_category_id)
    return float(fallback_threshold_px)


def build_distance_rule_adjusted_metrics(
    aggregate: Mapping[str, Any],
    *,
    center_distance_bucket_threshold_px: float = DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
) -> Dict[str, Any]:
    fallback_threshold_px = float(center_distance_bucket_threshold_px)
    original_tp = _as_int(aggregate.get("event_tp", aggregate.get("tp", 0.0)))
    original_missed = _as_int(aggregate.get("missed_detection_count", 0.0))
    original_fp = _as_int(aggregate.get("false_positive_count", aggregate.get("fp", 0.0)))
    original_tn = _as_int(aggregate.get("tn", 0.0))
    original_no_gt_count = _as_int(aggregate.get("no_gt_predicted", aggregate.get("no_gt_count", 0.0)))
    original_avg_distance = _as_float(aggregate.get("avg_false_positive_center_distance_px", 0.0))
    gt_frames = _as_int(aggregate.get("gt_frames", 0.0))
    no_gt_frames = _as_int(aggregate.get("no_gt_frames", 0.0))
    predicted_frames = _as_int(aggregate.get("predicted_frames", 0.0))

    distance_map = _distance_map_from_aggregate(aggregate)
    threshold_map = _threshold_map_from_aggregate(aggregate)
    use_per_detection_threshold = bool(threshold_map)

    distance_le_threshold: Dict[str, float] = {}
    distance_gt_threshold: Dict[str, float] = {}
    threshold_by_frame: Dict[str, float] = {}
    for frame_key, distance_px in distance_map.items():
        threshold_px = threshold_map.get(frame_key, fallback_threshold_px)
        threshold_by_frame[frame_key] = threshold_px
        if distance_px <= threshold_px:
            distance_le_threshold[frame_key] = distance_px
        else:
            distance_gt_threshold[frame_key] = distance_px

    threshold_le_by_frame = {
        frame_key: threshold_by_frame[frame_key] for frame_key in distance_le_threshold
    }
    threshold_gt_by_frame = {
        frame_key: threshold_by_frame[frame_key] for frame_key in distance_gt_threshold
    }
    avg_threshold_px = _as_float(
        aggregate.get("avg_false_positive_center_distance_threshold_px", 0.0)
    )
    if avg_threshold_px <= 0.0 and threshold_by_frame:
        avg_threshold_px = float(sum(threshold_by_frame.values()) / len(threshold_by_frame))
    reported_threshold_px = avg_threshold_px if use_per_detection_threshold else fallback_threshold_px

    return {
        "center_distance_threshold_rule": (
            "per_detection_predicted_bbox_perimeter"
            if use_per_detection_threshold
            else "global_fallback_threshold"
        ),
        "center_distance_global_fallback_threshold_px": fallback_threshold_px,
        "center_distance_bucket_threshold_px": reported_threshold_px,
        "tp": original_tp,
        "missed_detection_count": original_missed,
        "false_positive_count": original_fp,
        "tn": original_tn,
        "no_gt_count": original_no_gt_count,
        "avg_false_positive_center_distance_px": float(original_avg_distance),
        "gt_frames": gt_frames,
        "no_gt_frames": no_gt_frames,
        "predicted_frames": predicted_frames,
        "avg_false_positive_center_distance_threshold_px": avg_threshold_px,
        "false_positive_center_distance_threshold_by_frame_px": threshold_by_frame,
        "distance_le_threshold_px_count": len(distance_le_threshold),
        "original_tp": original_tp,
        "original_missed_detection_count": original_missed,
        "original_false_positive_count": original_fp,
        "original_tn": original_tn,
        "original_no_gt_count": original_no_gt_count,
        "original_avg_false_positive_center_distance_px": original_avg_distance,
        "original_avg_false_positive_center_distance_threshold_px": avg_threshold_px,
        "distance_le_threshold_px_by_frame": distance_le_threshold,
        "distance_gt_threshold_px_by_frame": distance_gt_threshold,
        "distance_le_threshold_px_threshold_by_frame": threshold_le_by_frame,
        "distance_gt_threshold_px_threshold_by_frame": threshold_gt_by_frame,
        # Backward-compatible aliases for older readers. These names no longer
        # imply a fixed 50px cutoff when per-detection thresholds are present.
        "center_distance_tp_threshold_px": reported_threshold_px,
        "distance_le_50_px_count": len(distance_le_threshold),
        "distance_le_50_px_by_frame": distance_le_threshold,
        "remaining_false_positive_center_distance_by_frame_px": distance_gt_threshold,
        "reclassified_fp_to_tp_count": len(distance_le_threshold),
        "reclassified_false_positive_center_distance_by_frame_px": distance_le_threshold,
        "no_gt_predicted": original_no_gt_count,
        "original_no_gt_predicted": original_no_gt_count,
    }


def attach_distance_rule_summary(
    experiment_summary: MutableMapping[str, Any],
    *,
    center_distance_bucket_threshold_px: float = DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
    include_raw_stage: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "center_distance_bucket_threshold_px": float(center_distance_bucket_threshold_px),
        "center_distance_threshold_rule": "per_detection_predicted_bbox_perimeter",
        "center_distance_global_fallback_threshold_px": float(center_distance_bucket_threshold_px),
    }
    evaluation = experiment_summary.get("evaluation", {})
    if isinstance(evaluation, Mapping) and str(evaluation.get("status", "")) == "ok":
        final_stage = evaluation.get("final")
        if isinstance(final_stage, Mapping):
            final_aggregate = final_stage.get("aggregate")
            if isinstance(final_aggregate, Mapping):
                payload["final"] = build_distance_rule_adjusted_metrics(
                    final_aggregate,
                    center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
                )
        if include_raw_stage:
            raw_stage = evaluation.get("raw")
            if isinstance(raw_stage, Mapping):
                raw_aggregate = raw_stage.get("aggregate")
                if isinstance(raw_aggregate, Mapping):
                    payload["raw"] = build_distance_rule_adjusted_metrics(
                        raw_aggregate,
                        center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
                    )

    final_payload = payload.get("final")
    if isinstance(final_payload, Mapping):
        payload["center_distance_bucket_threshold_px"] = float(
            final_payload.get(
                "center_distance_bucket_threshold_px",
                payload["center_distance_bucket_threshold_px"],
            )
        )
    payload["center_distance_tp_threshold_px"] = payload["center_distance_bucket_threshold_px"]
    experiment_summary["distance_bucket_metrics"] = payload
    experiment_summary["distance_rule_adjusted_tp50"] = payload
    experiment_summary["distance_rule_adjusted_bucketed"] = payload
    return payload


def write_single_stage_distance_rule_metrics_csv(
    path: Path,
    *,
    latency_ms: float,
    iou_threshold: float,
    score_threshold: float,
    aggregate: Mapping[str, Any],
    center_distance_bucket_threshold_px: float = DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
) -> Dict[str, Any]:
    adjusted = build_distance_rule_adjusted_metrics(
        aggregate,
        center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
    )
    write_benchmark_metrics_csv(
        path,
        latency_ms=float(latency_ms),
        tp=int(adjusted["tp"]),
        missed_detection_count=int(adjusted["missed_detection_count"]),
        false_positive_count=int(adjusted["false_positive_count"]),
        no_gt_count=int(adjusted["no_gt_count"]),
        distance_le_threshold_px_count=int(adjusted["distance_le_threshold_px_count"]),
        distance_le_50_px_count=int(adjusted["distance_le_50_px_count"]),
        tn=int(adjusted["tn"]),
        avg_false_positive_center_distance_px=float(adjusted["avg_false_positive_center_distance_px"]),
        iou_threshold=float(iou_threshold),
        score_threshold=float(score_threshold),
        gt_frames=int(adjusted["gt_frames"]),
        no_gt_frames=int(adjusted["no_gt_frames"]),
        predicted_frames=int(adjusted["predicted_frames"]),
        center_distance_threshold_rule=str(adjusted["center_distance_threshold_rule"]),
        center_distance_global_fallback_threshold_px=float(
            adjusted["center_distance_global_fallback_threshold_px"]
        ),
        center_distance_bucket_threshold_px=float(adjusted["center_distance_bucket_threshold_px"]),
        avg_false_positive_center_distance_threshold_px=float(
            adjusted["avg_false_positive_center_distance_threshold_px"]
        ),
        original_tp=int(adjusted["original_tp"]),
        original_missed_detection_count=int(adjusted["original_missed_detection_count"]),
        original_false_positive_count=int(adjusted["original_false_positive_count"]),
        original_no_gt_count=int(adjusted["original_no_gt_count"]),
        original_tn=int(adjusted["original_tn"]),
        original_avg_false_positive_center_distance_px=float(
            adjusted["original_avg_false_positive_center_distance_px"]
        ),
        original_avg_false_positive_center_distance_threshold_px=float(
            adjusted["original_avg_false_positive_center_distance_threshold_px"]
        ),
    )
    return adjusted


def write_dual_stage_distance_rule_metrics_csv(
    path: Path,
    *,
    latency_ms: float,
    iou_threshold: float,
    score_threshold: float,
    raw_aggregate: Optional[Mapping[str, Any]],
    final_aggregate: Optional[Mapping[str, Any]],
    center_distance_bucket_threshold_px: float = DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "latency_ms": float(latency_ms),
        "iou_threshold": float(iou_threshold),
        "score_threshold": float(score_threshold),
        "center_distance_threshold_rule": "per_detection_predicted_bbox_perimeter",
        "center_distance_global_fallback_threshold_px": float(center_distance_bucket_threshold_px),
        "center_distance_bucket_threshold_px": float(center_distance_bucket_threshold_px),
    }
    adjusted: Dict[str, Any] = {}

    if raw_aggregate:
        raw_adjusted = build_distance_rule_adjusted_metrics(
            raw_aggregate,
            center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
        )
        adjusted["raw"] = raw_adjusted
        payload.update(
            {
                "raw_tp": int(raw_adjusted["tp"]),
                "raw_missed_detection_count": int(raw_adjusted["missed_detection_count"]),
                "raw_false_positive_count": int(raw_adjusted["false_positive_count"]),
                "raw_no_gt_count": int(raw_adjusted["no_gt_count"]),
                "raw_distance_le_threshold_px_count": int(raw_adjusted["distance_le_threshold_px_count"]),
                "raw_distance_le_50_px_count": int(raw_adjusted["distance_le_50_px_count"]),
                "raw_tn": int(raw_adjusted["tn"]),
                "raw_avg_false_positive_center_distance_px": float(
                    raw_adjusted["avg_false_positive_center_distance_px"]
                ),
                "raw_center_distance_threshold_rule": str(raw_adjusted["center_distance_threshold_rule"]),
                "raw_center_distance_global_fallback_threshold_px": float(
                    raw_adjusted["center_distance_global_fallback_threshold_px"]
                ),
                "raw_center_distance_bucket_threshold_px": float(
                    raw_adjusted["center_distance_bucket_threshold_px"]
                ),
                "raw_avg_false_positive_center_distance_threshold_px": float(
                    raw_adjusted["avg_false_positive_center_distance_threshold_px"]
                ),
                "raw_original_tp": int(raw_adjusted["original_tp"]),
                "raw_original_missed_detection_count": int(
                    raw_adjusted["original_missed_detection_count"]
                ),
                "raw_original_false_positive_count": int(
                    raw_adjusted["original_false_positive_count"]
                ),
                "raw_original_no_gt_count": int(raw_adjusted["original_no_gt_count"]),
                "raw_original_tn": int(raw_adjusted["original_tn"]),
                "raw_original_avg_false_positive_center_distance_px": float(
                    raw_adjusted["original_avg_false_positive_center_distance_px"]
                ),
                "raw_original_avg_false_positive_center_distance_threshold_px": float(
                    raw_adjusted["original_avg_false_positive_center_distance_threshold_px"]
                ),
            }
        )

    if final_aggregate:
        final_adjusted = build_distance_rule_adjusted_metrics(
            final_aggregate,
            center_distance_bucket_threshold_px=center_distance_bucket_threshold_px,
        )
        adjusted["final"] = final_adjusted
        payload.update(
            {
                "final_tp": int(final_adjusted["tp"]),
                "final_missed_detection_count": int(final_adjusted["missed_detection_count"]),
                "final_false_positive_count": int(final_adjusted["false_positive_count"]),
                "final_no_gt_count": int(final_adjusted["no_gt_count"]),
                "final_distance_le_threshold_px_count": int(final_adjusted["distance_le_threshold_px_count"]),
                "final_distance_le_50_px_count": int(final_adjusted["distance_le_50_px_count"]),
                "final_tn": int(final_adjusted["tn"]),
                "final_avg_false_positive_center_distance_px": float(
                    final_adjusted["avg_false_positive_center_distance_px"]
                ),
                "final_center_distance_threshold_rule": str(
                    final_adjusted["center_distance_threshold_rule"]
                ),
                "final_center_distance_global_fallback_threshold_px": float(
                    final_adjusted["center_distance_global_fallback_threshold_px"]
                ),
                "final_center_distance_bucket_threshold_px": float(
                    final_adjusted["center_distance_bucket_threshold_px"]
                ),
                "final_avg_false_positive_center_distance_threshold_px": float(
                    final_adjusted["avg_false_positive_center_distance_threshold_px"]
                ),
                "final_original_tp": int(final_adjusted["original_tp"]),
                "final_original_missed_detection_count": int(
                    final_adjusted["original_missed_detection_count"]
                ),
                "final_original_false_positive_count": int(
                    final_adjusted["original_false_positive_count"]
                ),
                "final_original_no_gt_count": int(final_adjusted["original_no_gt_count"]),
                "final_original_tn": int(final_adjusted["original_tn"]),
                "final_original_avg_false_positive_center_distance_px": float(
                    final_adjusted["original_avg_false_positive_center_distance_px"]
                ),
                "final_original_avg_false_positive_center_distance_threshold_px": float(
                    final_adjusted["original_avg_false_positive_center_distance_threshold_px"]
                ),
            }
        )

    primary_adjusted = adjusted.get("final") or adjusted.get("raw")
    if primary_adjusted:
        payload["center_distance_bucket_threshold_px"] = float(
            primary_adjusted["center_distance_bucket_threshold_px"]
        )
    write_benchmark_metrics_csv(path, **payload)
    return adjusted
