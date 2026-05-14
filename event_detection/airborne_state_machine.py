"""
Airborne state machine: wraps the rule-based airborne detector with proper
landing detection so events have measured (not fixed) end frames.

States:
  GROUND     - default; ball is on the ground or in low-height play
  AIRBORNE   - ball is in flight; tracker should suspend bbox output
  UNCERTAIN  - safety state; emergency exit if airborne lasts too long

Landing detection requires the ball to have actually ASCENDED a meaningful
amount above the ground baseline before a "near baseline" signal is accepted
as a landing. This prevents false landings right after launch.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque

from airborne_rule import (
    AirborneRuleConfig,
    AirborneEvent,
    evaluate_rule_at_frame,
)


# ============================================================
# Config
# ============================================================
@dataclass
class StateMachineConfig:
    # Ground baseline: rolling median of ball y over last N GROUND frames
    ground_baseline_window_frames: int = 120

    # Landing detection
    landing_distance_to_baseline_px: float = 50.0
    landing_consecutive_detections: int = 2

    # Ball must climb at least this many pixels above baseline before
    # we'll accept a landing signal. Kills the "landed in 3 frames" bug.
    min_ascent_before_landing_px: float = 60.0

    # Safety net: maximum airborne duration before forcing UNCERTAIN
    max_airborne_frames: int = 180

    # Cooldown after an event ends before a new one can fire.
    # Prevents immediate re-fires when the ball bounces right after landing.
    post_event_cooldown_frames: int = 15

    # Recovery from UNCERTAIN: how many ground-level detections to confirm we're back to GROUND
    uncertain_recovery_frames: int = 5


@dataclass
class CompletedAirborneEvent:
    """An airborne event with measured (not fixed) end frame."""
    start_frame: int
    end_frame: int
    duration_frames: int
    fire_score: float
    fire_breakdown: dict
    end_reason: str   # "landed", "max_duration", "still_active_at_end_of_video"

    def to_airborne_event(self) -> AirborneEvent:
        return AirborneEvent(
            start_frame=self.start_frame,
            duration_frames=self.duration_frames,
            event_type="airborne",
            fired_at_frame=self.start_frame,
            confidence_score=self.fire_score,
            score_breakdown=self.fire_breakdown,
        )


# ============================================================
# State machine
# ============================================================
class AirborneStateMachine:
    """
    Streaming-style state machine. Call .step(features_row) per frame.
    Returns a CompletedAirborneEvent when an airborne event ends, else None.
    """

    GROUND = "GROUND"
    AIRBORNE = "AIRBORNE"
    UNCERTAIN = "UNCERTAIN"

    def __init__(self, rule_config: AirborneRuleConfig, sm_config: StateMachineConfig):
        self.rule_config = rule_config
        self.sm_config = sm_config

        self.state = self.GROUND
        self._recent_rows = []

        # Ground baseline tracking
        self._ground_y_history = deque(maxlen=sm_config.ground_baseline_window_frames)
        self.ground_baseline_y: Optional[float] = None

        # Active event state
        self._active_event_start: Optional[int] = None
        self._active_event_score: Optional[float] = None
        self._active_event_breakdown: Optional[dict] = None
        self._active_event_baseline: Optional[float] = None
        self._min_y_during_airborne: Optional[float] = None

        # Counters for landing / uncertain recovery
        self._near_ground_streak: int = 0
        self._uncertain_recovery_streak: int = 0

        # Post-event cooldown: frame index until which new events are suppressed
        self._cooldown_until: int = -1

    def _update_recent_rows(self, features_row):
        self._recent_rows.append(features_row)
        if len(self._recent_rows) > self.rule_config.ground_window_frames:
            self._recent_rows.pop(0)

    def _compute_recent_density(self):
        recent_n_dets = sum(
            1 for r in self._recent_rows
            if r.get("frames_since_last_det", 999) == 0
        )
        return recent_n_dets / max(1, len(self._recent_rows))

    def _update_ground_baseline(self, features_row):
        last_y = features_row.get("last_y")
        frames_since = features_row.get("frames_since_last_det", 999)
        if (last_y is not None
                and not (isinstance(last_y, float) and np.isnan(last_y))
                and frames_since == 0):
            self._ground_y_history.append(float(last_y))
            if len(self._ground_y_history) >= 5:
                self.ground_baseline_y = float(np.median(self._ground_y_history))

    def _is_ball_near_ground(self, features_row) -> bool:
        if self.ground_baseline_y is None:
            return False
        last_y = features_row.get("last_y")
        frames_since = features_row.get("frames_since_last_det", 999)
        if last_y is None or (isinstance(last_y, float) and np.isnan(last_y)) or frames_since != 0:
            return False
        return abs(float(last_y) - self.ground_baseline_y) <= self.sm_config.landing_distance_to_baseline_px

    def step(self, features_row) -> Optional[CompletedAirborneEvent]:
        t = int(features_row["frame"])
        self._update_recent_rows(features_row)
        completed_event = None

        if self.state == self.GROUND:
            self._update_ground_baseline(features_row)
            if t <= self._cooldown_until:
                return completed_event  # suppress new events during post-landing cooldown
            recent_density = self._compute_recent_density()
            result = evaluate_rule_at_frame(features_row, recent_density, self.rule_config)
            if result is not None and self._passes_consistency_check():
                score, breakdown = result
                self.state = self.AIRBORNE
                self._active_event_start = t
                self._active_event_score = score
                self._active_event_breakdown = breakdown
                self._active_event_baseline = self.ground_baseline_y
                last_y = features_row.get("last_y")
                self._min_y_during_airborne = (
                    float(last_y)
                    if last_y is not None and not (isinstance(last_y, float) and np.isnan(last_y))
                    else None
                )
                self._near_ground_streak = 0

        elif self.state == self.AIRBORNE:
            airborne_duration = t - self._active_event_start

            # Track minimum y (= maximum altitude) during this event
            last_y = features_row.get("last_y")
            frames_since = features_row.get("frames_since_last_det", 999)
            if (last_y is not None
                    and not (isinstance(last_y, float) and np.isnan(last_y))
                    and frames_since == 0):
                if self._min_y_during_airborne is None:
                    self._min_y_during_airborne = float(last_y)
                else:
                    self._min_y_during_airborne = min(self._min_y_during_airborne, float(last_y))

            # Did the ball ascend enough?
            if self._active_event_baseline is not None and self._min_y_during_airborne is not None:
                ascent = self._active_event_baseline - self._min_y_during_airborne
            else:
                ascent = 0.0
            has_ascended = ascent >= self.sm_config.min_ascent_before_landing_px

            # Update landing streak only after sufficient ascent
            if has_ascended and self._is_ball_near_ground(features_row):
                self._near_ground_streak += 1
            else:
                self._near_ground_streak = 0

            if self._near_ground_streak >= self.sm_config.landing_consecutive_detections:
                completed_event = CompletedAirborneEvent(
                    start_frame=self._active_event_start,
                    end_frame=t,
                    duration_frames=t - self._active_event_start,
                    fire_score=self._active_event_score,
                    fire_breakdown=self._active_event_breakdown,
                    end_reason="landed",
                )
                self._reset_event_state()
                self._cooldown_until = t + self.sm_config.post_event_cooldown_frames
                self.state = self.GROUND

            elif airborne_duration > self.sm_config.max_airborne_frames:
                completed_event = CompletedAirborneEvent(
                    start_frame=self._active_event_start,
                    end_frame=t,
                    duration_frames=airborne_duration,
                    fire_score=self._active_event_score,
                    fire_breakdown=self._active_event_breakdown,
                    end_reason="max_duration",
                )
                self._reset_event_state()
                self._cooldown_until = t + self.sm_config.post_event_cooldown_frames
                self.state = self.GROUND

        elif self.state == self.UNCERTAIN:
            # Kept for backward compatibility but should rarely be reached now.
            # Treat like GROUND: try the rule, accept new airborne events.
            self._update_ground_baseline(features_row)
            recent_density = self._compute_recent_density()
            result = evaluate_rule_at_frame(features_row, recent_density, self.rule_config)
            if result is not None and self._passes_consistency_check():
                score, breakdown = result
                self.state = self.AIRBORNE
                self._active_event_start = t
                self._active_event_score = score
                self._active_event_breakdown = breakdown
                self._active_event_baseline = self.ground_baseline_y
                last_y = features_row.get("last_y")
                self._min_y_during_airborne = (
                    float(last_y)
                    if last_y is not None and not (isinstance(last_y, float) and np.isnan(last_y))
                    else None
                )
                self._near_ground_streak = 0

        return completed_event

    def _passes_consistency_check(self) -> bool:
        """Require >=2 of the last N frames to show sustained upward motion.
        Kills single-frame bounce spikes that happen to cross the score threshold."""
        cf = self.rule_config.upward_velocity_consistency_frames
        ct = self.rule_config.upward_velocity_consistency_threshold
        recent_dys = [r.get("dy_per_frame_last_pair") for r in self._recent_rows[-cf:]]
        upward_count = sum(
            1 for dy in recent_dys
            if dy is not None
            and not (isinstance(dy, float) and np.isnan(dy))
            and dy <= ct
        )
        return upward_count >= 2

    def _reset_event_state(self):
        self._active_event_start = None
        self._active_event_score = None
        self._active_event_breakdown = None
        self._active_event_baseline = None
        self._min_y_during_airborne = None
        self._near_ground_streak = 0


# ============================================================
# Offline batch detection using the state machine
# ============================================================
def detect_airborne_events_with_state_machine(
    features_df,
    rule_config: AirborneRuleConfig,
    sm_config: StateMachineConfig,
) -> List[CompletedAirborneEvent]:
    sm = AirborneStateMachine(rule_config, sm_config)
    completed_events = []

    feature_dicts = features_df.to_dict("records")
    for row in feature_dicts:
        completed = sm.step(row)
        if completed is not None:
            completed_events.append(completed)

    if sm.state == sm.AIRBORNE and sm._active_event_start is not None:
        last_frame = int(feature_dicts[-1]["frame"])
        completed_events.append(CompletedAirborneEvent(
            start_frame=sm._active_event_start,
            end_frame=last_frame,
            duration_frames=last_frame - sm._active_event_start,
            fire_score=sm._active_event_score,
            fire_breakdown=sm._active_event_breakdown,
            end_reason="still_active_at_end_of_video",
        ))

    return completed_events


# ============================================================
# Test harness
# ============================================================
if __name__ == "__main__":
    df = pd.read_csv("trajectory_features_c2.csv")
    rule_config = AirborneRuleConfig()
    sm_config = StateMachineConfig()

    events = detect_airborne_events_with_state_machine(df, rule_config, sm_config)
    print(f"Detected {len(events)} airborne events:")
    print(f"  {'start':>6} {'end':>6} {'duration':>8} {'reason':>22} {'score':>6}")
    for e in events:
        print(f"  {e.start_frame:>6} {e.end_frame:>6} {e.duration_frames:>8} "
              f"{e.end_reason:>22} {e.fire_score:>6.2f}")
