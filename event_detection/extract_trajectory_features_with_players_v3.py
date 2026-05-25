"""
Trajectory feature extractor v3 for airborne event detection.

Extends v2 with per-frame ground baseline derived from player bboxes.
This solves the stale-baseline problem in clips with camera pans/zooms.

Per-frame ground baseline = median(player_bbox.y2) across all detected players.
"""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_BUFFER_L = 30
SHORT_WINDOWS = (5, 10, 15)

ARC_WINDOW_FRAMES = 15
ARC_MIN_POINTS = 6
ARC_INLIER_PIXEL_TOL = 30.0
ARC_GRAVITY_A_MIN = 0.05
ARC_GRAVITY_A_MAX = 5.0
ARC_INLIER_RATIO_OK = 0.65


def load_detections(path: str) -> dict[int, tuple[float, float, float] | None]:
    """Returns dict: frame_idx -> (x, y, conf) or None if no detection."""
    with open(path) as f:
        data = json.load(f)

    det: dict[int, tuple[float, float, float] | None] = {}
    for entry in data:
        frame_idx = int(entry["frame_idx"])
        x, y = entry.get("x"), entry.get("y")
        conf = entry.get("confidence")
        if x is not None and y is not None:
            det[frame_idx] = (
                float(x),
                float(y),
                float(conf) if conf is not None else 0.0,
            )
        else:
            det[frame_idx] = None
    return det


def load_player_detections(path: str) -> dict[int, list[dict]]:
    """Load player detections: frame_idx -> list of {x1, y1, x2, y2, class_id, confidence}"""
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def compute_player_ground_baseline_per_frame(
    player_dets: dict[int, list[dict]],
    max_frame: int,
) -> dict[int, float | None]:
    """
    For each frame, compute ground baseline as max y2 (bottom-most) of player bboxes.

    Max y2 represents the closest point to ground line where any player was detected.
    Falls back to median if max is unreliable (outliers).

    Returns: {frame_idx -> ground_y2} or None if no players detected.
    """
    baseline_by_frame: dict[int, float | None] = {}

    for t in range(max_frame + 1):
        players = player_dets.get(t, [])
        if not players:
            baseline_by_frame[t] = None
            continue

        y2_coords = [p["y2"] for p in players if "y2" in p]
        if y2_coords:
            # Use max y2 (closest to bottom/ground line in image)
            # but clamp to 90th percentile to avoid outliers
            y2_array = np.array(y2_coords)
            p90 = float(np.percentile(y2_array, 90))
            baseline_by_frame[t] = p90
        else:
            baseline_by_frame[t] = None

    return baseline_by_frame


def _nan_arc_features(suffix: str = "15") -> dict[str, float | int | bool]:
    return {
        f"parabola_n_{suffix}": 0,
        f"parabola_a_{suffix}": np.nan,
        f"parabola_b_{suffix}": np.nan,
        f"parabola_c_{suffix}": np.nan,
        f"parabola_rmse_{suffix}": np.nan,
        f"parabola_inlier_ratio_{suffix}": np.nan,
        f"parabola_ok_{suffix}": False,
    }


def _fit_parabola_realtime(
    frame_y_points: Iterable[tuple[int, float]],
    *,
    min_points: int = ARC_MIN_POINTS,
    inlier_pixel_tol: float = ARC_INLIER_PIXEL_TOL,
    gravity_a_min: float = ARC_GRAVITY_A_MIN,
    gravity_a_max: float = ARC_GRAVITY_A_MAX,
    inlier_ratio_ok: float = ARC_INLIER_RATIO_OK,
    suffix: str = "15",
) -> dict[str, float | int | bool]:
    """Fast causal parabola fit for y = a*t^2 + b*t + c."""
    points = list(frame_y_points)
    if len(points) < min_points:
        return _nan_arc_features(suffix)

    frames = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    ts = frames - frames[0]
    if len(np.unique(ts)) < 3:
        return _nan_arc_features(suffix)

    try:
        coeffs = np.polyfit(ts, ys, 2)
    except Exception:
        return _nan_arc_features(suffix)

    pred = np.polyval(coeffs, ts)
    residuals = np.abs(ys - pred)
    inliers = residuals <= inlier_pixel_tol

    if 3 <= int(inliers.sum()) < len(points):
        try:
            coeffs = np.polyfit(ts[inliers], ys[inliers], 2)
        except Exception:
            pass

    pred = np.polyval(coeffs, ts)
    residuals = np.abs(ys - pred)
    inliers = residuals <= inlier_pixel_tol
    inlier_ratio = float(inliers.mean())
    rmse = float(np.sqrt(np.mean((ys - pred) ** 2)))
    a, b, c = [float(v) for v in coeffs]
    parabola_ok = (
        gravity_a_min <= a <= gravity_a_max
        and inlier_ratio >= inlier_ratio_ok
        and rmse <= inlier_pixel_tol * 1.5
    )

    return {
        f"parabola_n_{suffix}": int(len(points)),
        f"parabola_a_{suffix}": a,
        f"parabola_b_{suffix}": b,
        f"parabola_c_{suffix}": c,
        f"parabola_rmse_{suffix}": rmse,
        f"parabola_inlier_ratio_{suffix}": inlier_ratio,
        f"parabola_ok_{suffix}": bool(parabola_ok),
    }


def _window_motion_features(
    buffer_dets: list[tuple[int, float, float, float]],
    t: int,
    window: int,
) -> dict[str, float | int]:
    suffix = str(window)
    start = max(0, t - window + 1)
    dets = [d for d in buffer_dets if d[0] >= start]
    density = len(dets) / window

    out: dict[str, float | int] = {
        f"density_{suffix}": float(density),
        f"n_detections_{suffix}": int(len(dets)),
        f"y_delta_{suffix}": np.nan,
        f"y_range_{suffix}": np.nan,
        f"min_dy_per_frame_{suffix}": np.nan,
        f"upward_pairs_{suffix}": 0,
    }

    if len(dets) >= 1:
        ys = np.array([d[2] for d in dets], dtype=np.float64)
        out[f"y_range_{suffix}"] = float(ys.max() - ys.min())
    if len(dets) >= 2:
        out[f"y_delta_{suffix}"] = float(dets[-1][2] - dets[0][2])
        dys = []
        upward_pairs = 0
        for d1, d2 in zip(dets[:-1], dets[1:]):
            frame_gap = max(1, d2[0] - d1[0])
            dy_per_frame = float((d2[2] - d1[2]) / frame_gap)
            dys.append(dy_per_frame)
            if dy_per_frame <= -3.0:
                upward_pairs += 1
        out[f"min_dy_per_frame_{suffix}"] = float(min(dys))
        out[f"upward_pairs_{suffix}"] = int(upward_pairs)

    return out


def extract_features(
    detections: dict[int, tuple[float, float, float] | None],
    player_detections: dict[int, list[dict]],
    buffer_l: int = DEFAULT_BUFFER_L,
    *,
    arc_window_frames: int = ARC_WINDOW_FRAMES,
    arc_min_points: int = ARC_MIN_POINTS,
    arc_inlier_pixel_tol: float = ARC_INLIER_PIXEL_TOL,
) -> pd.DataFrame:
    """
    For each frame t, compute causal trajectory features including player-based ground baseline.

    New column in v3: `player_ground_baseline_y` = median y2 of player bboxes
    """
    if not detections:
        return pd.DataFrame()

    max_frame = max(detections.keys())

    # Pre-compute player ground baseline for all frames
    player_baseline_by_frame = compute_player_ground_baseline_per_frame(player_detections, max_frame)

    rows = []

    for t in range(max_frame + 1):
        buffer_start = max(0, t - buffer_l + 1)
        buffer_frames = list(range(buffer_start, t + 1))
        buffer_dets = [
            (f, *detections[f])
            for f in buffer_frames
            if f in detections and detections[f] is not None
        ]

        current_det = detections.get(t)
        current_has_detection = current_det is not None
        n_dets = len(buffer_dets)
        ys = np.array([d[2] for d in buffer_dets], dtype=np.float64) if buffer_dets else np.array([])

        if n_dets > 0:
            median_y = float(np.median(ys))
            y_range = float(ys.max() - ys.min())
            last_frame, last_x, last_y, last_conf = buffer_dets[-1]
            frames_since_last = t - last_frame
            last_x_val = float(last_x)
            last_y_val = float(last_y)
            last_conf_val = float(last_conf)
        else:
            median_y = np.nan
            y_range = np.nan
            last_frame = -1
            last_x_val = np.nan
            last_y_val = np.nan
            last_conf_val = np.nan
            frames_since_last = buffer_l

        if n_dets >= 2:
            f1, x1, y1, c1 = buffer_dets[-2]
            f2, x2, y2, c2 = buffer_dets[-1]
            frame_gap_last_pair = int(max(1, f2 - f1))
            missing_frames_before_current = int(max(0, frame_gap_last_pair - 1)) if f2 == t else 0
            dy_last_pair = float(y2 - y1)
            dy_per_frame_last_pair = float(dy_last_pair / frame_gap_last_pair)
            dy_buffer_first_to_last = float(buffer_dets[-1][2] - buffer_dets[0][2])
            prev_detection_frame = int(f1)
            prev_y = float(y1)
        else:
            frame_gap_last_pair = 0
            missing_frames_before_current = 0
            dy_last_pair = np.nan
            dy_per_frame_last_pair = np.nan
            dy_buffer_first_to_last = np.nan
            prev_detection_frame = -1
            prev_y = np.nan

        reappeared_after_gap = bool(current_has_detection and missing_frames_before_current > 0)
        y_jump_after_gap = dy_last_pair if reappeared_after_gap else np.nan
        y_drop_after_gap = (
            float(max(0.0, -dy_last_pair))
            if reappeared_after_gap and not np.isnan(dy_last_pair)
            else 0.0
        )

        # V3: Add player-based ground baseline
        player_ground_baseline = player_baseline_by_frame.get(t)

        row: dict[str, float | int | bool] = {
            "frame": t,
            "n_detections_in_buffer": n_dets,
            "detection_density": n_dets / buffer_l,
            "median_y_in_buffer": median_y,
            "y_range_in_buffer": y_range,
            "frames_since_last_det": int(frames_since_last),
            "last_x": last_x_val,
            "last_y": last_y_val,
            "last_confidence": last_conf_val,
            "last_detection_frame": int(last_frame),
            "prev_detection_frame": int(prev_detection_frame),
            "prev_y": prev_y,
            "dy_last_pair": dy_last_pair,
            "dy_per_frame_last_pair": dy_per_frame_last_pair,
            "dy_buffer_first_to_last": dy_buffer_first_to_last,
            "frame_gap_last_pair": int(frame_gap_last_pair),
            "missing_frames_before_current": int(missing_frames_before_current),
            "current_has_detection": bool(current_has_detection),
            "reappeared_after_gap": bool(reappeared_after_gap),
            "y_jump_after_gap": y_jump_after_gap,
            "y_drop_after_gap": y_drop_after_gap,
            "player_ground_baseline_y": player_ground_baseline,
        }

        for window in SHORT_WINDOWS:
            row.update(_window_motion_features(buffer_dets, t, window))

        arc_start = max(0, t - arc_window_frames + 1)
        arc_points = [
            (f, y)
            for f, _x, y, _conf in buffer_dets
            if f >= arc_start
        ]
        row.update(
            _fit_parabola_realtime(
                arc_points,
                min_points=arc_min_points,
                inlier_pixel_tol=arc_inlier_pixel_tol,
                suffix=str(arc_window_frames),
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("detections_json")
    parser.add_argument("player_detections_json")
    parser.add_argument("--out", default="trajectory_features_v3.csv")
    parser.add_argument("--buffer-l", type=int, default=DEFAULT_BUFFER_L)
    parser.add_argument("--arc-window", type=int, default=ARC_WINDOW_FRAMES)
    args = parser.parse_args()

    dets = load_detections(args.detections_json)
    player_dets = load_player_detections(args.player_detections_json)
    df = extract_features(dets, player_dets, buffer_l=args.buffer_l, arc_window_frames=args.arc_window)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows)")
