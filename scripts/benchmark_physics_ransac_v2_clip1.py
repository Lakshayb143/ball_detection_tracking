#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import ransacp  # noqa: E402
from ball_detection_metrics import (  # noqa: E402
    evaluate_detections,
    infer_category_id_by_name,
    write_detection_export,
)
from benchmark_physics_ransac_clip1 import (  # noqa: E402
    DEFAULT_ANNOTATIONS,
    DEFAULT_BOX_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    add_detection,
    ensure_dir,
    load_coco_frame_map,
    original_metric_csv,
    write_rows_csv,
)
from distance_rule_adjustment import (  # noqa: E402
    DEFAULT_CENTER_DISTANCE_BUCKET_THRESHOLD_PX,
    attach_distance_rule_summary,
    resolve_center_distance_bucket_threshold_px,
    write_dual_stage_distance_rule_metrics_csv,
)


DEFAULT_TRACKER_JSON = (
    REPO_ROOT / "clip1_fresh_runs" / "ball_outlier_interpolator_v4__clip1" / "point_outputs.json"
)
DEFAULT_RUN_NAME = "v4_then_ransac_v2__clip1"
DEFAULT_MAX_SEGMENT_FRAMES = 60
DEFAULT_LONG_RAW_SPLIT_FRAMES = 20
DEFAULT_MIN_FIT_POINTS = ransacp.MIN_SEGMENT_FRAMES
DEFAULT_RECURSIVE_SPLIT_GAIN = 500.0


@dataclass
class PointRecord:
    frame_idx: int
    x: Optional[float]
    y: Optional[float]
    confidence: Optional[float]
    source: str
    interpolated: bool


@dataclass
class FitRecord:
    segment: str
    parent_segment: str
    split_reason: str
    fit_source: str
    detections: int
    a: Optional[float]
    inlier_ratio: Optional[float]
    inliers: int
    outliers: int
    physics_ok: Optional[bool]
    verdict: str


@dataclass
class FrameRecord:
    frame_idx: int
    coco_frame: int
    image_id: Optional[int]
    raw_predicted: bool
    raw_interpolated: bool
    raw_source: str
    raw_score: float
    in_ransac_fit: bool
    segment: str
    segment_physics_ok: Optional[bool]
    fit_source: str
    ransac_inlier: Optional[bool]
    ransac_residual_px: Optional[float]
    final_kept: bool
    final_source: str
    final_x: Optional[float]
    final_y: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate v2 physics-RANSAC repair on clip1 v4 tracker outputs. "
            "Unlike v1, v2 keeps v4 by default and uses clean subsegment fits "
            "to repair interpolated outliers."
        )
    )
    parser.add_argument("--tracker_json", type=Path, default=DEFAULT_TRACKER_JSON)
    parser.add_argument("--airborne_json", type=Path, default=Path(ransacp.AIRBORNE_SEGS_PATH))
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--frame_index_offset",
        type=int,
        default=0,
        help="COCO frame number = tracker frame_idx + offset. v4 point_outputs are already 1-based.",
    )
    parser.add_argument("--box_size", type=float, default=DEFAULT_BOX_SIZE)
    parser.add_argument("--match_iou", type=float, default=0.01)
    parser.add_argument("--eval_score_threshold", type=float, default=0.0)
    parser.add_argument("--ball_category_id", type=int, default=None)
    parser.add_argument("--max_segment_frames", type=int, default=DEFAULT_MAX_SEGMENT_FRAMES)
    parser.add_argument("--long_raw_split_frames", type=int, default=DEFAULT_LONG_RAW_SPLIT_FRAMES)
    parser.add_argument("--min_fit_points", type=int, default=DEFAULT_MIN_FIT_POINTS)
    parser.add_argument("--recursive_split_gain", type=float, default=DEFAULT_RECURSIVE_SPLIT_GAIN)
    parser.add_argument("--max_recursive_split_depth", type=int, default=4)
    parser.add_argument(
        "--velocity_split_accel_px",
        type=float,
        default=0.0,
        help=(
            "If >0, recursively split source pieces at raw-anchor velocity discontinuities "
            "whose frame-to-frame velocity change exceeds this pixel threshold."
        ),
    )
    parser.add_argument("--max_velocity_split_depth", type=int, default=4)
    parser.add_argument(
        "--residual_split_px",
        type=float,
        default=0.0,
        help=(
            "If >0, recursively split pieces around the largest fit residual when "
            "that residual exceeds this pixel threshold."
        ),
    )
    parser.add_argument("--residual_split_min_gain", type=float, default=0.0)
    parser.add_argument("--max_residual_split_depth", type=int, default=4)
    parser.add_argument(
        "--online_lookahead_frames",
        type=int,
        default=None,
        help=(
            "If set, use delayed-online fitting for each frame using detections only "
            "through frame + this buffer. Omit for offline/full-subsegment fitting."
        ),
    )
    parser.add_argument(
        "--online_history_frames",
        type=int,
        default=ransacp.ONLINE_HISTORY_FRAMES,
        help="Past frames used in delayed-online fitting; 0 means all subsegment history so far.",
    )
    parser.add_argument("--repair_residual_px", type=float, default=ransacp.INLIER_PIXEL_TOL)
    parser.add_argument("--inlier_pixel_tol", type=float, default=ransacp.INLIER_PIXEL_TOL)
    parser.add_argument("--gravity_a_min", type=float, default=ransacp.GRAVITY_A_MIN)
    parser.add_argument("--gravity_a_max", type=float, default=ransacp.GRAVITY_A_MAX)
    parser.add_argument("--inlier_ratio_ok", type=float, default=ransacp.INLIER_RATIO_OK)
    parser.add_argument(
        "--repair_fit_source_policy",
        choices=["raw", "raw_or_clean_all", "any_physics_ok"],
        default="raw",
        help=(
            "Which fitted trajectories may repair interpolated outputs. raw is safest; "
            "raw_or_clean_all allows all-point fits only when they pass clean-fit guards."
        ),
    )
    parser.add_argument("--clean_all_repair_min_inlier_ratio", type=float, default=0.95)
    parser.add_argument("--clean_all_repair_max_frames", type=int, default=20)
    parser.add_argument(
        "--fill_gap_max_frames",
        type=int,
        default=4,
        help="Fill missing frames only for no-output gaps at most this long inside a clean raw-anchor fit.",
    )
    parser.add_argument(
        "--no_repair_interpolated_outliers",
        action="store_true",
        help="Disable replacing interpolated outputs that disagree with a clean raw-anchor fit.",
    )
    parser.add_argument(
        "--no_fill_short_gaps",
        action="store_true",
        help="Disable adding fitted outputs for short no-output gaps inside clean raw-anchor fits.",
    )
    return parser.parse_args()


def load_point_records(path: Path) -> Dict[int, PointRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = enumerate(raw)
    else:
        raise ValueError(f"Unsupported tracker JSON format in {path}: {type(raw).__name__}")

    records: Dict[int, PointRecord] = {}
    for fallback_idx, value in items:
        if value is None:
            continue
        if isinstance(value, dict):
            frame_idx = int(value.get("frame_idx", value.get("frame", fallback_idx)))
            x = value.get("x")
            y = value.get("y")
            confidence = value.get("confidence", value.get("score"))
            source = str(value.get("source", "unknown"))
            interpolated = bool(value.get("interpolated", value.get("interpolated_output", False)))
        else:
            frame_idx = int(fallback_idx)
            x = value[0] if len(value) > 0 else None
            y = value[1] if len(value) > 1 else None
            confidence = value[4] if len(value) > 4 else 1.0
            source = "tracker"
            interpolated = False

        records[frame_idx] = PointRecord(
            frame_idx=frame_idx,
            x=float(x) if x is not None else None,
            y=float(y) if y is not None else None,
            confidence=float(confidence) if confidence is not None else None,
            source=source,
            interpolated=interpolated,
        )
    return records


def center_data_for_segment(
    records: Dict[int, PointRecord],
    seg_start: int,
    seg_end: int,
    *,
    raw_only: bool,
) -> Dict[int, Tuple[float, float, float, float]]:
    data: Dict[int, Tuple[float, float, float, float]] = {}
    for frame_idx in range(int(seg_start), int(seg_end) + 1):
        record = records.get(frame_idx)
        if record is None or record.x is None or record.y is None:
            continue
        if raw_only and record.interpolated:
            continue
        data[frame_idx] = (float(record.x), float(record.y), 0.0, 0.0)
    return data


def fit_segment(
    records: Dict[int, PointRecord],
    seg_start: int,
    seg_end: int,
    min_fit_points: int,
) -> Optional[dict]:
    raw_data = center_data_for_segment(records, seg_start, seg_end, raw_only=True)
    fit_source = "raw"
    if len(raw_data) >= int(min_fit_points):
        fit = ransacp.process_segment(seg_start, seg_end, raw_data)
    else:
        fit = None

    if fit is None:
        all_data = center_data_for_segment(records, seg_start, seg_end, raw_only=False)
        fit_source = "all"
        if len(all_data) < int(min_fit_points):
            return None
        fit = ransacp.process_segment(seg_start, seg_end, all_data)

    if fit is None:
        return None
    fit["fit_source"] = fit_source
    return fit


def fit_segment_window(
    records: Dict[int, PointRecord],
    seg_start: int,
    seg_end: int,
    fit_start: int,
    fit_end: int,
    min_fit_points: int,
) -> Optional[dict]:
    fit_start = max(int(seg_start), int(fit_start))
    fit_end = min(int(seg_end), int(fit_end))
    if fit_end < fit_start:
        return None

    raw_data = center_data_for_segment(records, fit_start, fit_end, raw_only=True)
    fit_source = "raw"
    if len(raw_data) >= int(min_fit_points):
        fit = ransacp.process_segment(seg_start, seg_end, raw_data)
    else:
        fit = None

    if fit is None:
        all_data = center_data_for_segment(records, fit_start, fit_end, raw_only=False)
        fit_source = "all"
        if len(all_data) < int(min_fit_points):
            return None
        fit = ransacp.process_segment(seg_start, seg_end, all_data)

    if fit is None:
        return None
    fit["fit_source"] = fit_source
    fit["fit_start"] = fit_start
    fit["fit_end"] = fit_end
    fit["mode"] = "online_delayed"
    return fit


def frame_kind(record: Optional[PointRecord]) -> str:
    if record is None or record.x is None or record.y is None:
        return "none"
    if record.interpolated:
        return "interp"
    return "raw"


def source_runs(records: Dict[int, PointRecord], seg_start: int, seg_end: int) -> List[Tuple[int, int, str]]:
    runs: List[Tuple[int, int, str]] = []
    for frame_idx in range(int(seg_start), int(seg_end) + 1):
        kind = frame_kind(records.get(frame_idx))
        if not runs or runs[-1][2] != kind:
            runs.append((frame_idx, frame_idx, kind))
        else:
            start, _end, prev_kind = runs[-1]
            runs[-1] = (start, frame_idx, prev_kind)
    return runs


def split_long_segment_by_sources(
    records: Dict[int, PointRecord],
    seg_start: int,
    seg_end: int,
    max_segment_frames: int,
    long_raw_split_frames: int,
    min_fit_points: int,
) -> List[Tuple[int, int, str]]:
    if int(seg_end) - int(seg_start) + 1 <= int(max_segment_frames):
        return [(int(seg_start), int(seg_end), "airborne_segment")]

    islands: List[Tuple[int, int]] = []
    island_start: Optional[int] = None
    for frame_idx in range(int(seg_start), int(seg_end) + 1):
        if frame_kind(records.get(frame_idx)) == "none":
            if island_start is not None:
                islands.append((island_start, frame_idx - 1))
                island_start = None
            continue
        if island_start is None:
            island_start = frame_idx
    if island_start is not None:
        islands.append((island_start, int(seg_end)))

    pieces: List[Tuple[int, int, str]] = []
    for island_start, island_end in islands:
        current_start = island_start
        for run_start, run_end, kind in source_runs(records, island_start, island_end):
            if kind != "raw":
                continue
            if run_end - run_start + 1 < int(long_raw_split_frames):
                continue

            if run_start > current_start and run_start - current_start >= int(min_fit_points):
                pieces.append((current_start, run_start - 1, "before_long_raw_run"))
                current_start = run_start

            remaining = island_end - run_end
            current_len = run_end - current_start + 1
            if remaining >= int(min_fit_points) and current_len >= int(min_fit_points):
                pieces.append((current_start, run_end, "long_raw_run"))
                current_start = run_end + 1

        if island_end - current_start + 1 >= int(min_fit_points):
            pieces.append((current_start, island_end, "source_island_tail"))

    return pieces


def fit_score(fit: Optional[dict]) -> float:
    if fit is None:
        return 0.0
    ok_bonus = 1000.0 if bool(fit.get("physics_ok", False)) else 0.0
    return ok_bonus + float(fit.get("inlier_ratio", 0.0)) * float(len(fit.get("frames", [])))


def count_nonmissing_points(records: Dict[int, PointRecord], seg_start: int, seg_end: int) -> int:
    count = 0
    for frame_idx in range(int(seg_start), int(seg_end) + 1):
        record = records.get(frame_idx)
        if record is not None and record.x is not None and record.y is not None:
            count += 1
    return count


def raw_anchor_points(records: Dict[int, PointRecord], seg_start: int, seg_end: int) -> List[Tuple[int, float, float]]:
    points: List[Tuple[int, float, float]] = []
    for frame_idx in range(int(seg_start), int(seg_end) + 1):
        record = records.get(frame_idx)
        if record is None or record.x is None or record.y is None or record.interpolated:
            continue
        points.append((frame_idx, float(record.x), float(record.y)))
    return points


def split_by_velocity_discontinuity(
    records: Dict[int, PointRecord],
    seg_start: int,
    seg_end: int,
    min_fit_points: int,
    velocity_split_accel_px: float,
    max_depth: int,
    *,
    depth: int = 0,
) -> List[Tuple[int, int, str]]:
    if float(velocity_split_accel_px) <= 0.0 or depth >= int(max_depth):
        return [(int(seg_start), int(seg_end), "velocity_split_disabled" if depth == 0 else "velocity_split_done")]

    points = raw_anchor_points(records, seg_start, seg_end)
    if len(points) < max(3, int(min_fit_points)):
        return [(int(seg_start), int(seg_end), "velocity_split_too_few_raw")]

    best: Optional[Tuple[float, int]] = None
    for left, mid, right in zip(points, points[1:], points[2:]):
        f0, x0, y0 = left
        f1, x1, y1 = mid
        f2, x2, y2 = right
        dt0 = f1 - f0
        dt1 = f2 - f1
        if dt0 <= 0 or dt1 <= 0:
            continue
        vx0 = (x1 - x0) / dt0
        vy0 = (y1 - y0) / dt0
        vx1 = (x2 - x1) / dt1
        vy1 = (y2 - y1) / dt1
        accel = float(np.hypot(vx1 - vx0, vy1 - vy0))
        split_frame = int(f1)
        if count_nonmissing_points(records, seg_start, split_frame) < int(min_fit_points):
            continue
        if count_nonmissing_points(records, split_frame + 1, seg_end) < int(min_fit_points):
            continue
        if best is None or accel > best[0]:
            best = (accel, split_frame)

    if best is None or best[0] < float(velocity_split_accel_px):
        return [(int(seg_start), int(seg_end), "velocity_split_no_jump")]

    split_frame = best[1]
    left_pieces = split_by_velocity_discontinuity(
        records,
        seg_start,
        split_frame,
        min_fit_points,
        velocity_split_accel_px,
        max_depth,
        depth=depth + 1,
    )
    right_pieces = split_by_velocity_discontinuity(
        records,
        split_frame + 1,
        seg_end,
        min_fit_points,
        velocity_split_accel_px,
        max_depth,
        depth=depth + 1,
    )
    pieces: List[Tuple[int, int, str]] = []
    for start, end, reason in left_pieces + right_pieces:
        if reason.startswith("velocity_split"):
            pieces.append((start, end, f"velocity_jump_{best[0]:.1f}px"))
        else:
            pieces.append((start, end, reason))
    return pieces


def split_by_largest_residual(
    records: Dict[int, PointRecord],
    seg_start: int,
    seg_end: int,
    min_fit_points: int,
    residual_split_px: float,
    residual_split_min_gain: float,
    max_depth: int,
    *,
    depth: int = 0,
) -> List[Tuple[int, int, str]]:
    if float(residual_split_px) <= 0.0 or depth >= int(max_depth):
        return [(int(seg_start), int(seg_end), "residual_split_disabled" if depth == 0 else "residual_split_done")]

    fit = fit_segment(records, seg_start, seg_end, min_fit_points)
    if fit is None or len(fit.get("frames", [])) < max(3, int(min_fit_points)):
        return [(int(seg_start), int(seg_end), "residual_split_too_few_points")]

    residuals = np.asarray(fit["residuals"], dtype=np.float64)
    if residuals.size == 0:
        return [(int(seg_start), int(seg_end), "residual_split_no_residuals")]

    max_index = int(np.argmax(residuals))
    max_residual = float(residuals[max_index])
    if max_residual < float(residual_split_px):
        return [(int(seg_start), int(seg_end), "residual_split_below_threshold")]

    frames = [int(frame) for frame in fit["frames"]]
    residual_frame = frames[max_index]
    candidates = [residual_frame - 1, residual_frame]
    parent_score = fit_score(fit)
    best: Optional[Tuple[float, int]] = None
    for split_frame in candidates:
        if split_frame < int(seg_start) or split_frame >= int(seg_end):
            continue
        if count_nonmissing_points(records, seg_start, split_frame) < int(min_fit_points):
            continue
        if count_nonmissing_points(records, split_frame + 1, seg_end) < int(min_fit_points):
            continue
        left_fit = fit_segment(records, seg_start, split_frame, min_fit_points)
        right_fit = fit_segment(records, split_frame + 1, seg_end, min_fit_points)
        if left_fit is None or right_fit is None:
            continue
        score = fit_score(left_fit) + fit_score(right_fit)
        if best is None or score > best[0]:
            best = (score, split_frame)

    if best is None or best[0] < parent_score + float(residual_split_min_gain):
        return [(int(seg_start), int(seg_end), "residual_split_no_gain")]

    split_frame = best[1]
    left_pieces = split_by_largest_residual(
        records,
        seg_start,
        split_frame,
        min_fit_points,
        residual_split_px,
        residual_split_min_gain,
        max_depth,
        depth=depth + 1,
    )
    right_pieces = split_by_largest_residual(
        records,
        split_frame + 1,
        seg_end,
        min_fit_points,
        residual_split_px,
        residual_split_min_gain,
        max_depth,
        depth=depth + 1,
    )
    pieces: List[Tuple[int, int, str]] = []
    for start, end, reason in left_pieces + right_pieces:
        if reason.startswith("residual_split"):
            pieces.append((start, end, f"largest_residual_{max_residual:.1f}px"))
        else:
            pieces.append((start, end, reason))
    return pieces


def recursive_split_bad_segment(
    records: Dict[int, PointRecord],
    seg_start: int,
    seg_end: int,
    min_fit_points: int,
    recursive_split_gain: float,
    max_depth: int,
    *,
    depth: int = 0,
) -> List[Tuple[int, int, str]]:
    fit = fit_segment(records, seg_start, seg_end, min_fit_points)
    if fit is None or bool(fit["physics_ok"]):
        return [(int(seg_start), int(seg_end), "physics_ok" if fit is not None else "too_few_points")]
    if depth >= int(max_depth):
        return [(int(seg_start), int(seg_end), "max_split_depth")]

    parent_score = fit_score(fit)
    best: Optional[Tuple[float, int, dict, dict]] = None
    min_span = int(min_fit_points)
    for split_frame in range(int(seg_start) + min_span - 1, int(seg_end) - min_span + 1):
        left_fit = fit_segment(records, seg_start, split_frame, min_fit_points)
        right_fit = fit_segment(records, split_frame + 1, seg_end, min_fit_points)
        if left_fit is None or right_fit is None:
            continue
        score = fit_score(left_fit) + fit_score(right_fit)
        if not bool(left_fit["physics_ok"]) and not bool(right_fit["physics_ok"]):
            continue
        if best is None or score > best[0] or (score == best[0] and split_frame > best[1]):
            best = (score, split_frame, left_fit, right_fit)

    if best is None or best[0] < parent_score + float(recursive_split_gain):
        return [(int(seg_start), int(seg_end), "physics_bad_unsplit")]

    split_frame = best[1]
    left = recursive_split_bad_segment(
        records,
        seg_start,
        split_frame,
        min_fit_points,
        recursive_split_gain,
        max_depth,
        depth=depth + 1,
    )
    right = recursive_split_bad_segment(
        records,
        split_frame + 1,
        seg_end,
        min_fit_points,
        recursive_split_gain,
        max_depth,
        depth=depth + 1,
    )
    return left + right


def build_subsegments(
    segments: Sequence[Tuple[int, int]],
    records: Dict[int, PointRecord],
    args: argparse.Namespace,
) -> List[Tuple[int, int, str, str]]:
    subsegments: List[Tuple[int, int, str, str]] = []
    for seg_start, seg_end in segments:
        parent_name = f"{int(seg_start)}-{int(seg_end)}"
        source_pieces = split_long_segment_by_sources(
            records=records,
            seg_start=int(seg_start),
            seg_end=int(seg_end),
            max_segment_frames=int(args.max_segment_frames),
            long_raw_split_frames=int(args.long_raw_split_frames),
            min_fit_points=int(args.min_fit_points),
        )
        for piece_start, piece_end, source_reason in source_pieces:
            velocity_pieces = split_by_velocity_discontinuity(
                records=records,
                seg_start=piece_start,
                seg_end=piece_end,
                min_fit_points=int(args.min_fit_points),
                velocity_split_accel_px=float(args.velocity_split_accel_px),
                max_depth=int(args.max_velocity_split_depth),
            )
            for velocity_start, velocity_end, velocity_reason in velocity_pieces:
                residual_pieces = split_by_largest_residual(
                    records=records,
                    seg_start=velocity_start,
                    seg_end=velocity_end,
                    min_fit_points=int(args.min_fit_points),
                    residual_split_px=float(args.residual_split_px),
                    residual_split_min_gain=float(args.residual_split_min_gain),
                    max_depth=int(args.max_residual_split_depth),
                )
                for residual_start, residual_end, residual_reason in residual_pieces:
                    for sub_start, sub_end, recursive_reason in recursive_split_bad_segment(
                        records=records,
                        seg_start=residual_start,
                        seg_end=residual_end,
                        min_fit_points=int(args.min_fit_points),
                        recursive_split_gain=float(args.recursive_split_gain),
                        max_depth=int(args.max_recursive_split_depth),
                    ):
                        if recursive_reason in {"physics_ok", "too_few_points"}:
                            reason = source_reason
                            if float(args.velocity_split_accel_px) > 0.0:
                                reason = f"{reason};{velocity_reason}"
                            if float(args.residual_split_px) > 0.0:
                                reason = f"{reason};{residual_reason}"
                        else:
                            reason = recursive_reason
                        subsegments.append((sub_start, sub_end, parent_name, reason))
    return subsegments


def predicted_center(fit: dict, frame_idx: int) -> Tuple[float, float]:
    t = float(int(frame_idx) - int(fit["seg_start"]))
    x = float(fit["x_slope"]) * t + float(fit["x_intercept"])
    y = float(np.polyval([float(fit["a"]), float(fit["b"]), float(fit["c"])], t))
    return float(x), float(y)


def fit_decision_for_frame(fit: dict, record: Optional[PointRecord], frame_idx: int) -> dict:
    pred_x, pred_y = predicted_center(fit, frame_idx)
    if record is None or record.x is None or record.y is None:
        residual = None
        is_inlier = None
    else:
        residual = abs(float(record.y) - pred_y)
        is_inlier = bool(residual <= ransacp.INLIER_PIXEL_TOL)
    return {
        "fit": fit,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "residual": residual,
        "inlier": is_inlier,
        "segment": f"{int(fit['seg_start'])}-{int(fit['seg_end'])}",
        "physics_ok": bool(fit["physics_ok"]),
        "fit_source": str(fit.get("fit_source", "unknown")),
        "fit_inlier_ratio": float(fit.get("inlier_ratio", 0.0)),
        "fit_detection_count": int(len(fit.get("frames", []))),
    }


def build_fit_decisions(
    subsegments: Sequence[Tuple[int, int, str, str]],
    records: Dict[int, PointRecord],
    args: argparse.Namespace,
) -> Tuple[Dict[int, dict], List[FitRecord]]:
    frame_decisions: Dict[int, dict] = {}
    fit_records: List[FitRecord] = []

    for seg_start, seg_end, parent_segment, split_reason in subsegments:
        fit = fit_segment(records, seg_start, seg_end, int(args.min_fit_points))
        segment_name = f"{int(seg_start)}-{int(seg_end)}"
        if fit is None:
            fit_records.append(
                FitRecord(
                    segment=segment_name,
                    parent_segment=parent_segment,
                    split_reason=split_reason,
                    fit_source="none",
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
        n_inliers = int(inlier_mask.sum())
        n_total = len(fit["frames"])
        fit_records.append(
            FitRecord(
                segment=segment_name,
                parent_segment=parent_segment,
                split_reason=split_reason,
                fit_source=str(fit.get("fit_source", "unknown")),
                detections=n_total,
                a=float(fit["a"]),
                inlier_ratio=float(fit["inlier_ratio"]),
                inliers=n_inliers,
                outliers=n_total - n_inliers,
                physics_ok=bool(fit["physics_ok"]),
                verdict="PHYSICS_OK" if fit["physics_ok"] else "PHYSICS_BAD",
            )
        )

        for frame_idx in range(int(seg_start), int(seg_end) + 1):
            decision = fit_decision_for_frame(fit, records.get(frame_idx), frame_idx)
            previous = frame_decisions.get(frame_idx)
            if previous is None:
                frame_decisions[frame_idx] = decision
                continue

            previous_fit = previous["fit"]
            decision_rank = (
                int(bool(decision["physics_ok"])),
                int(decision["fit_source"] == "raw"),
                -float(decision["residual"] if decision["residual"] is not None else 1e9),
            )
            previous_rank = (
                int(bool(previous["physics_ok"])),
                int(str(previous_fit.get("fit_source", "unknown")) == "raw"),
                -float(previous["residual"] if previous["residual"] is not None else 1e9),
            )
            if decision_rank > previous_rank:
                frame_decisions[frame_idx] = decision

    return frame_decisions, fit_records


def build_online_fit_decisions(
    subsegments: Sequence[Tuple[int, int, str, str]],
    records: Dict[int, PointRecord],
    args: argparse.Namespace,
) -> Tuple[Dict[int, dict], List[FitRecord]]:
    frame_decisions: Dict[int, dict] = {}
    fit_records: List[FitRecord] = []
    lookahead_frames = int(args.online_lookahead_frames)
    history_frames = int(args.online_history_frames)

    for seg_start, seg_end, parent_segment, split_reason in subsegments:
        segment_name = f"{int(seg_start)}-{int(seg_end)}"
        seg_decisions: List[dict] = []
        raw_fit_count = 0
        all_fit_count = 0
        a_values: List[float] = []
        ratio_values: List[float] = []

        for frame_idx in range(int(seg_start), int(seg_end) + 1):
            if history_frames > 0:
                fit_start = max(int(seg_start), frame_idx - history_frames)
            else:
                fit_start = int(seg_start)
            fit_end = min(int(seg_end), frame_idx + lookahead_frames)
            fit = fit_segment_window(
                records=records,
                seg_start=int(seg_start),
                seg_end=int(seg_end),
                fit_start=fit_start,
                fit_end=fit_end,
                min_fit_points=int(args.min_fit_points),
            )
            if fit is None:
                continue

            decision = fit_decision_for_frame(fit, records.get(frame_idx), frame_idx)
            decision["decision_ready_frame"] = int(frame_idx + lookahead_frames)
            decision["fit_start"] = int(fit["fit_start"])
            decision["fit_end"] = int(fit["fit_end"])
            frame_decisions[frame_idx] = decision
            seg_decisions.append(decision)
            if str(fit.get("fit_source", "unknown")) == "raw":
                raw_fit_count += 1
            elif str(fit.get("fit_source", "unknown")) == "all":
                all_fit_count += 1
            a_values.append(float(fit["a"]))
            ratio_values.append(float(fit["inlier_ratio"]))

        inlier_count = sum(1 for decision in seg_decisions if decision["inlier"] is True)
        outlier_count = sum(1 for decision in seg_decisions if decision["inlier"] is False)
        ok_count = sum(1 for decision in seg_decisions if bool(decision["physics_ok"]))
        if raw_fit_count and all_fit_count:
            fit_source = "mixed"
        elif raw_fit_count:
            fit_source = "raw"
        elif all_fit_count:
            fit_source = "all"
        else:
            fit_source = "none"

        fit_records.append(
            FitRecord(
                segment=segment_name,
                parent_segment=parent_segment,
                split_reason=f"{split_reason};online_lookahead_{lookahead_frames}",
                fit_source=fit_source,
                detections=len(seg_decisions),
                a=float(np.mean(a_values)) if a_values else None,
                inlier_ratio=float(inlier_count / (inlier_count + outlier_count))
                if (inlier_count + outlier_count) > 0
                else None,
                inliers=int(inlier_count),
                outliers=int(outlier_count),
                physics_ok=bool(ok_count == len(seg_decisions)) if seg_decisions else None,
                verdict=f"ONLINE_DELAYED_+{lookahead_frames}_OK={ok_count}/{len(seg_decisions)}",
            )
        )

    return frame_decisions, fit_records


def short_missing_gap_frames(records: Dict[int, PointRecord], max_gap_frames: int) -> set[int]:
    fillable: set[int] = set()
    if max_gap_frames <= 0:
        return fillable

    frames = sorted(records)
    run_start: Optional[int] = None
    run_end: Optional[int] = None
    has_left_prediction = False
    for frame_idx in frames:
        record = records[frame_idx]
        missing = record.x is None or record.y is None
        if missing:
            if run_start is None:
                run_start = frame_idx
            run_end = frame_idx
            continue

        if run_start is not None and run_end is not None:
            if has_left_prediction and run_end - run_start + 1 <= int(max_gap_frames):
                fillable.update(range(run_start, run_end + 1))
            run_start = None
            run_end = None
        has_left_prediction = True

    return fillable


def fit_source_can_repair(decision: dict, args: argparse.Namespace) -> bool:
    fit_source = str(decision["fit_source"])
    if fit_source == "raw":
        return True
    if args.repair_fit_source_policy == "any_physics_ok":
        return True
    if args.repair_fit_source_policy != "raw_or_clean_all":
        return False
    return (
        fit_source == "all"
        and float(decision.get("fit_inlier_ratio", 0.0)) >= float(args.clean_all_repair_min_inlier_ratio)
        and int(decision.get("fit_detection_count", 0)) <= int(args.clean_all_repair_max_frames)
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    ransacp.INLIER_PIXEL_TOL = float(args.inlier_pixel_tol)
    ransacp.GRAVITY_A_MIN = float(args.gravity_a_min)
    ransacp.GRAVITY_A_MAX = float(args.gravity_a_max)
    ransacp.INLIER_RATIO_OK = float(args.inlier_ratio_ok)

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
    records = load_point_records(args.tracker_json)
    airborne_segments = ransacp.load_airborne_segments(args.airborne_json)
    subsegments = build_subsegments(airborne_segments, records, args)
    if args.online_lookahead_frames is None:
        decisions, fit_records = build_fit_decisions(subsegments, records, args)
    else:
        decisions, fit_records = build_online_fit_decisions(subsegments, records, args)
    fillable_missing = short_missing_gap_frames(
        records,
        0 if bool(args.no_fill_short_gaps) else int(args.fill_gap_max_frames),
    )

    detections: List[dict] = []
    point_outputs: List[dict] = []
    frame_records: List[FrameRecord] = []

    for frame_idx in sorted(records):
        record = records[frame_idx]
        coco_frame = int(frame_idx) + int(args.frame_index_offset)
        image = frame_map.get(coco_frame)
        if image is None:
            continue

        raw_predicted = record.x is not None and record.y is not None
        score = float(record.confidence) if record.confidence is not None else 1.0
        decision = decisions.get(frame_idx)

        if raw_predicted:
            add_detection(
                detections,
                stage="raw",
                source="tracker",
                frame_idx=frame_idx,
                coco_frame=coco_frame,
                image=image,
                xywh_record=(float(record.x), float(record.y), 0.0, 0.0),
                score=score,
                box_size=float(args.box_size),
            )

        final_kept = raw_predicted
        final_x = float(record.x) if record.x is not None else None
        final_y = float(record.y) if record.y is not None else None
        final_source = "tracker_raw" if raw_predicted else "no_output"

        can_repair = (
            raw_predicted
            and not bool(args.no_repair_interpolated_outliers)
            and bool(record.interpolated)
            and decision is not None
            and bool(decision["physics_ok"])
            and fit_source_can_repair(decision, args)
            and decision["residual"] is not None
            and float(decision["residual"]) > float(args.repair_residual_px)
        )
        if can_repair:
            final_x = float(decision["pred_x"])
            final_y = float(decision["pred_y"])
            final_source = "ransac_repaired_interpolation"

        can_fill = (
            not raw_predicted
            and frame_idx in fillable_missing
            and decision is not None
            and bool(decision["physics_ok"])
            and fit_source_can_repair(decision, args)
        )
        if can_fill:
            final_kept = True
            final_x = float(decision["pred_x"])
            final_y = float(decision["pred_y"])
            final_source = "ransac_filled_gap"

        if final_kept and final_x is not None and final_y is not None:
            add_detection(
                detections,
                stage="final",
                source=final_source,
                frame_idx=frame_idx,
                coco_frame=coco_frame,
                image=image,
                xywh_record=(final_x, final_y, 0.0, 0.0),
                score=score,
                box_size=float(args.box_size),
            )

        point_outputs.append(
            {
                "frame_idx": int(frame_idx),
                "coco_frame": int(coco_frame),
                "x": final_x,
                "y": final_y,
                "confidence": score if final_kept else None,
                "source": final_source,
                "kept_after_ransac": bool(final_kept),
                "raw_interpolated": bool(record.interpolated),
            }
        )
        frame_records.append(
            FrameRecord(
                frame_idx=int(frame_idx),
                coco_frame=int(coco_frame),
                image_id=int(image["image_id"]),
                raw_predicted=bool(raw_predicted),
                raw_interpolated=bool(record.interpolated),
                raw_source=str(record.source),
                raw_score=float(score) if raw_predicted else 0.0,
                in_ransac_fit=decision is not None,
                segment=str(decision["segment"]) if decision is not None else "",
                segment_physics_ok=bool(decision["physics_ok"]) if decision is not None else None,
                fit_source=str(decision["fit_source"]) if decision is not None else "",
                ransac_inlier=bool(decision["inlier"]) if decision is not None and decision["inlier"] is not None else None,
                ransac_residual_px=(
                    float(decision["residual"]) if decision is not None and decision["residual"] is not None else None
                ),
                final_kept=bool(final_kept),
                final_source=str(final_source),
                final_x=float(final_x) if final_x is not None else None,
                final_y=float(final_y) if final_y is not None else None,
            )
        )

    raw_metrics = evaluate_detections(
        detections=detections,
        annotations_path=args.annotations,
        stage="raw",
        iou_threshold=float(args.match_iou),
        score_threshold=float(args.eval_score_threshold),
        ball_category_id=int(args.ball_category_id),
    )
    final_metrics = evaluate_detections(
        detections=detections,
        annotations_path=args.annotations,
        stage="final",
        iou_threshold=float(args.match_iou),
        score_threshold=float(args.eval_score_threshold),
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
    write_rows_csv(run_dir / "segment_summary.csv", [asdict(record) for record in fit_records])

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
        "source_airborne_segments": [{"start": int(s), "end": int(e)} for s, e in airborne_segments],
        "subsegments": [
            {"start": int(s), "end": int(e), "parent_segment": parent, "split_reason": reason}
            for s, e, parent, reason in subsegments
        ],
        "segments": [asdict(record) for record in fit_records],
        "counts": {
            "annotated_frames": len(frame_map),
            "raw_predictions": int(sum(1 for item in detections if item["stage"] == "raw")),
            "final_predictions": int(sum(1 for item in detections if item["stage"] == "final")),
            "ransac_inliers": int(sum(record.inliers for record in fit_records)),
            "ransac_outliers": int(sum(record.outliers for record in fit_records)),
            "repaired_interpolated_outputs": int(
                sum(1 for record in frame_records if record.final_source == "ransac_repaired_interpolation")
            ),
            "filled_gap_outputs": int(sum(1 for record in frame_records if record.final_source == "ransac_filled_gap")),
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
        float(args.match_iou),
        float(args.eval_score_threshold),
        raw_aggregate,
        final_aggregate,
    )
    original_metric_csv(
        args.output_root / f"{args.run_name}_benchmark_metrics_original.csv",
        latency_ms,
        float(args.match_iou),
        float(args.eval_score_threshold),
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
    print(
        "[INFO] RANSAC v2 changes: "
        f"repaired={experiment_summary['counts']['repaired_interpolated_outputs']}, "
        f"filled={experiment_summary['counts']['filled_gap_outputs']}, "
        f"subsegments={len(subsegments)}"
    )
    print(f"[INFO] Benchmark metrics written to {run_dir / 'benchmark_metrics.csv'}")


if __name__ == "__main__":
    main()
