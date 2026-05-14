"""
Rule-based airborne event detector — weighted-score version.

Each signal contributes a weighted score. The rule fires when the total score
crosses a threshold. Pose contributes but cannot fire alone — it only confirms
or strengthens trajectory signals.

Score components (defaults):
  - Trajectory: upward y-velocity     -> up to 1.5
  - Trajectory: sudden detection gap  -> up to 1.0
  - Pose: any kicking player          -> up to 0.5
  - Pose: kicker near last ball       -> up to 0.5

Fire threshold: 1.2

This means trajectory alone CAN fire if either trajectory signal is strong.
Pose alone CANNOT fire (max ~1.0 < 1.2). But trajectory's medium-strength
signals + pose confirmation crosses the threshold.

Two interfaces, same logic:
  - detect_airborne_events_offline(features_df, config) for evaluation
  - StreamingAirborneDetector(config) for real-time
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ============================================================
# Config — to be tuned after Step 1's plots
# ============================================================
@dataclass
class AirborneRuleConfig:
    ground_window_frames: int = 10
    min_density_for_ground: float = 0.3

    upward_dy_per_frame_threshold: float = -8.0
    sudden_gap_frames: int = 3
    min_last_y_for_arc: Optional[float] = None

    weight_upward_velocity: float = 2.0
    weight_sudden_gap: float = 0.5
    weight_pose_any_kicking: float = 0.5
    weight_pose_kicker_near_ball: float = 0.5

    kicker_near_ball_distance_px: float = 100.0

    upward_velocity_consistency_frames: int = 3
    upward_velocity_consistency_threshold: float = -3.0 

    fire_threshold: float = 1.2

    event_durations: Dict[str, int] = field(default_factory=lambda: {
        "airborne": 25, "high_pass": 25, "cross": 25,
        "shot": 20, "header": 15, "free_kick": 25,
    })
    event_duration_default: int = 25

    default_event_type: str = "airborne"

    # Cooldown after an event's active window ends, before another can fire normally
    post_event_cooldown_frames: int = 5   # ONE definition only


@dataclass
class AirborneEvent:
    start_frame: int
    duration_frames: int
    event_type: str
    fired_at_frame: int
    confidence_score: float
    score_breakdown: Dict[str, float] = field(default_factory=dict)


# ============================================================
# Score function
# ============================================================
def evaluate_rule_at_frame(features_row, recent_density, config):
    """
    Returns (score, breakdown_dict) if rule fires at this frame, else None.
    Breakdown is for diagnostics — shows which signals contributed.
    """
    # Gate: must have ground-like recent state. Pure airborne tracks (no
    # detections at all in the recent past) shouldn't fire because they aren't
    # the start of an airborne event — they're already mid-flight.
    if recent_density < config.min_density_for_ground:
        return None

    score = 0.0
    breakdown = {
        "upward_velocity": 0.0,
        "sudden_gap": 0.0,
        "pose_any_kicking": 0.0,
        "pose_kicker_near_ball": 0.0,
    }

    # Trajectory signal 1: upward velocity in image (negative dy)
    dy = features_row.get("dy_per_frame_last_pair")
    if dy is not None and not (isinstance(dy, float) and np.isnan(dy)):
        frames_since = features_row.get("frames_since_last_det", 0)
        if dy <= config.upward_dy_per_frame_threshold and frames_since == 0:
            magnitude = min(abs(dy) / 20.0, 1.0)
            contribution = config.weight_upward_velocity * magnitude
            score += contribution
            breakdown["upward_velocity"] = contribution

    # Trajectory signal 2: sudden detection gap
    frames_since = features_row.get("frames_since_last_det", 0)
    if frames_since >= config.sudden_gap_frames:
        gap_strength = min(frames_since / 10.0, 1.0)
        contribution = config.weight_sudden_gap * gap_strength
        score += contribution
        breakdown["sudden_gap"] = contribution

    # Pose signal 1: any kicking player in the scene
    if bool(features_row.get("pose_any_kicking", False)):
        kick_conf = float(features_row.get("pose_max_kicking_conf", 0.0))
        contribution = config.weight_pose_any_kicking * kick_conf
        score += contribution
        breakdown["pose_any_kicking"] = contribution

        # Pose signal 2: that kicking player is near the last ball position
        dist = features_row.get("pose_dist_ball_to_kicker")
        if dist is not None and not (isinstance(dist, float) and np.isnan(dist)):
            if dist <= config.kicker_near_ball_distance_px:
                # Closer = stronger
                proximity = 1.0 - (dist / config.kicker_near_ball_distance_px)
                contribution = config.weight_pose_kicker_near_ball * proximity
                score += contribution
                breakdown["pose_kicker_near_ball"] = contribution

    # Optional position gate
    if config.min_last_y_for_arc is not None:
        last_y = features_row.get("last_y")
        if last_y is None or (isinstance(last_y, float) and np.isnan(last_y)) \
                or last_y > config.min_last_y_for_arc:
            return None

    if score < config.fire_threshold:
        return None

    return score, breakdown


# ============================================================
# Offline detector (full DataFrame)
# ============================================================
def detect_airborne_events_offline(features_df, config):
    events = []
    active_event_until = -1
    cooldown_until = -1

    feature_dicts = features_df.to_dict("records")

    for i, row in enumerate(feature_dicts):
        t = int(row["frame"])

        in_active_or_cooldown = t <= active_event_until or t <= cooldown_until

        # Always evaluate the rule — we'll decide whether to suppress after
        lookback_start = max(0, i - config.ground_window_frames + 1)
        recent_rows = feature_dicts[lookback_start:i + 1]
        recent_n_dets = sum(
            1 for r in recent_rows if r.get("frames_since_last_det", 999) == 0
        )
        recent_density = recent_n_dets / max(1, len(recent_rows))
        result = evaluate_rule_at_frame(row, recent_density, config)
        if result is None:
            continue
        score, breakdown = result

        # CONSISTENCY CHECK: require upward motion to be sustained, not a one-frame bounce.
        # Look at the last N frames; require at least 2 of them to show upward velocity.
        consistency_frames = config.upward_velocity_consistency_frames
        consistency_threshold = config.upward_velocity_consistency_threshold
        recent_dys = [
            r.get("dy_per_frame_last_pair") for r in feature_dicts[max(0, i - consistency_frames + 1):i + 1]
        ]
        upward_count = sum(
            1 for dy in recent_dys
            if dy is not None and not (isinstance(dy, float) and np.isnan(dy))
            and dy <= consistency_threshold
        )
        if upward_count < 2:
            continue

        

        # Suppression rule: skip ONLY if inside active window AND score isn't strongly above threshold
        # This lets a clearly-new event break through an existing window
        strong_signal_threshold = config.fire_threshold * 1.5
        in_active_window = t <= active_event_until
        in_cooldown_only = active_event_until < t <= cooldown_until

        if in_active_window:
            continue   # hard suppress — same event still in flight
        
        if in_cooldown_only and score < strong_signal_threshold:
            continue   # weak signal during cooldown — probably tail of prior event

        # Fire the event (and replace the active window if we're breaking through)
        event_type = config.default_event_type
        duration = config.event_durations.get(event_type, config.event_duration_default)
        events.append(AirborneEvent(
            start_frame=t, duration_frames=duration, event_type=event_type,
            fired_at_frame=t, confidence_score=score, score_breakdown=breakdown,
        ))
        active_event_until = t + duration
        cooldown_until = active_event_until + config.post_event_cooldown_frames

    return events


# ============================================================
# Streaming detector (one frame at a time)
# ============================================================
class StreamingAirborneDetector:
    def __init__(self, config):
        self.config = config
        self._recent_rows = []
        self._active_event_until = -1
        self._cooldown_until = -1

    def step(self, features_row):
        t = int(features_row.get("frame"))

        self._recent_rows.append(features_row)
        if len(self._recent_rows) > self.config.ground_window_frames:
            self._recent_rows.pop(0)

        recent_n_dets = sum(
            1 for r in self._recent_rows
            if r.get("frames_since_last_det", 999) == 0
        )
        recent_density = recent_n_dets / max(1, len(self._recent_rows))

        result = evaluate_rule_at_frame(features_row, recent_density, self.config)
        if result is None:
            return None

        score, breakdown = result

        # Same suppression logic as offline
        strong_signal_threshold = self.config.fire_threshold * 1.5
        in_active_window = t <= self._active_event_until
        in_cooldown_only = self._active_event_until < t <= self._cooldown_until

        if in_active_window:
            return None
        if in_cooldown_only and score < strong_signal_threshold:
            return None

        event_type = self.config.default_event_type
        duration = self.config.event_durations.get(event_type, self.config.event_duration_default)
        event = AirborneEvent(
            start_frame=t, duration_frames=duration, event_type=event_type,
            fired_at_frame=t, confidence_score=score, score_breakdown=breakdown,
        )
        self._active_event_until = t + duration
        self._cooldown_until = self._active_event_until + self.config.post_event_cooldown_frames
        return event


# ============================================================
# Quick test harness
# ============================================================
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("trajectory_features.csv")
    config = AirborneRuleConfig()
    events = detect_airborne_events_offline(df, config)
    print(f"Detected {len(events)} airborne events")
    for e in events:
        print(f"  frame {e.start_frame:>4}  duration={e.duration_frames:>3}  "
              f"score={e.confidence_score:.2f}  breakdown={e.score_breakdown}")