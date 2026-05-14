"""
Rule scoring for airborne event detector v2.

The v2 rule keeps the v1 direct-launch behavior, then adds a separate
reappearance launch rule used by the v2 state machine. The reappearance rule
is for a common detector failure mode: recent grounded ball, short detection
gap, then the ball reappears much higher in the image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AirborneRuleV2Config:
    ground_window_frames: int = 10
    min_density_for_ground: float = 0.3

    upward_dy_per_frame_threshold: float = -8.0
    sudden_gap_frames: int = 3
    min_last_y_for_arc: Optional[float] = None

    weight_upward_velocity: float = 2.0
    weight_sudden_gap: float = 0.5
    weight_parabolic_arc: float = 0.4

    upward_velocity_consistency_frames: int = 3
    upward_velocity_consistency_threshold: float = -3.0

    fire_threshold: float = 1.2

    # Reappearance launch rule. These are generic image-space conditions:
    # after a short missing span, the ball should reappear meaningfully above
    # its last grounded y-position.
    candidate_min_pre_gap_density: float = 0.5
    candidate_min_missing_frames: int = 2
    candidate_max_missing_frames: int = 30
    reappearance_min_y_drop_px: float = 45.0
    reappearance_min_y_drop_per_frame: float = 4.0
    reappearance_fire_threshold: float = 1.0
    weight_reappearance_y_drop: float = 0.8
    weight_reappearance_upward_pair: float = 0.5
    weight_reappearance_arc: float = 0.3

    event_durations: dict[str, int] = field(default_factory=lambda: {
        "airborne": 25,
        "high_pass": 25,
        "cross": 25,
        "shot": 20,
        "header": 15,
        "free_kick": 25,
    })
    event_duration_default: int = 25
    default_event_type: str = "airborne"


@dataclass
class AirborneEventV2:
    start_frame: int
    duration_frames: int
    event_type: str
    fired_at_frame: int
    confidence_score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class LaunchCandidateV2:
    missing_start_frame: int
    last_ground_frame: int
    last_ground_y: float
    pre_gap_density: float
    baseline_y: Optional[float]


def _is_number(value) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def _arc_contribution(features_row, weight: float) -> float:
    if bool(features_row.get("parabola_ok_15", False)):
        ratio = features_row.get("parabola_inlier_ratio_15", 0.0)
        if _is_number(ratio):
            return float(weight * min(max(float(ratio), 0.0), 1.0))
    return 0.0


def evaluate_direct_launch_rule(features_row, recent_density: float, config: AirborneRuleV2Config):
    """
    Score a direct launch on the current frame.

    Returns (score, breakdown) if the rule fires, else None.
    """
    if recent_density < config.min_density_for_ground:
        return None

    score = 0.0
    breakdown = {
        "upward_velocity": 0.0,
        "sudden_gap": 0.0,
        "parabolic_arc": 0.0,
    }

    dy = features_row.get("dy_per_frame_last_pair")
    frames_since = int(features_row.get("frames_since_last_det", 0))
    if _is_number(dy) and float(dy) <= config.upward_dy_per_frame_threshold and frames_since == 0:
        magnitude = min(abs(float(dy)) / 20.0, 1.0)
        contribution = config.weight_upward_velocity * magnitude
        score += contribution
        breakdown["upward_velocity"] = contribution

    if frames_since >= config.sudden_gap_frames:
        gap_strength = min(frames_since / 10.0, 1.0)
        contribution = config.weight_sudden_gap * gap_strength
        score += contribution
        breakdown["sudden_gap"] = contribution

    contribution = _arc_contribution(features_row, config.weight_parabolic_arc)
    score += contribution
    breakdown["parabolic_arc"] = contribution

    if config.min_last_y_for_arc is not None:
        last_y = features_row.get("last_y")
        if not _is_number(last_y) or float(last_y) > config.min_last_y_for_arc:
            return None

    if score < config.fire_threshold:
        return None
    return score, breakdown


def evaluate_reappearance_launch_rule(
    features_row,
    candidate: LaunchCandidateV2,
    config: AirborneRuleV2Config,
):
    """
    Score a launch when the ball reappears after a short detector gap.

    This intentionally does not require camera motion, pose, or future frames.
    It uses the last grounded y, current y, gap length, and optional causal
    parabola quality from the last 15 frames.
    """
    if not bool(features_row.get("current_has_detection", False)):
        return None

    t = int(features_row["frame"])
    missing_frames = t - candidate.last_ground_frame - 1
    if missing_frames < config.candidate_min_missing_frames:
        return None
    if missing_frames > config.candidate_max_missing_frames:
        return None
    if candidate.pre_gap_density < config.candidate_min_pre_gap_density:
        return None

    last_y = features_row.get("last_y")
    if not _is_number(last_y):
        return None

    y_drop = float(candidate.last_ground_y - float(last_y))
    y_drop_per_frame = y_drop / max(1, t - candidate.last_ground_frame)
    if y_drop < config.reappearance_min_y_drop_px:
        return None
    if y_drop_per_frame < config.reappearance_min_y_drop_per_frame:
        return None

    score = 0.0
    breakdown = {
        "reappearance_y_drop": 0.0,
        "upward_pair_after_gap": 0.0,
        "parabolic_arc": 0.0,
    }

    drop_strength = min(y_drop / 120.0, 1.0)
    contribution = config.weight_reappearance_y_drop * drop_strength
    score += contribution
    breakdown["reappearance_y_drop"] = contribution

    dy = features_row.get("dy_per_frame_last_pair")
    if _is_number(dy) and float(dy) <= config.upward_velocity_consistency_threshold:
        pair_strength = min(abs(float(dy)) / 20.0, 1.0)
        contribution = config.weight_reappearance_upward_pair * pair_strength
        score += contribution
        breakdown["upward_pair_after_gap"] = contribution

    contribution = _arc_contribution(features_row, config.weight_reappearance_arc)
    score += contribution
    breakdown["parabolic_arc"] = contribution

    if score < config.reappearance_fire_threshold:
        return None
    return score, breakdown
