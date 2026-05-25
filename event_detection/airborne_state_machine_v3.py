"""
Airborne state machine v3.

Extends v2 with player-feet-derived per-frame ground baseline.
Instead of maintaining a historical deque median, uses the current frame's
median player bbox bottom (y2) as ground baseline. This solves the stale-baseline
problem in clips with camera pans/zooms (e.g., testing_clip_1080).

Falls back to deque median if no players detected in current frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from airborne_rule_v2 import (
    AirborneEventV2,
    AirborneRuleV2Config,
    LaunchCandidateV2,
    evaluate_direct_launch_rule,
    evaluate_reappearance_launch_rule,
)


@dataclass
class StateMachineV3Config:
    ground_baseline_window_frames: int = 120

    landing_distance_to_baseline_px: float = 50.0
    landing_consecutive_detections: int = 2
    min_ascent_before_landing_px: float = 60.0

    max_airborne_frames: int = 180
    post_event_cooldown_frames: int = 15


@dataclass
class CompletedAirborneEventV3:
    start_frame: int
    end_frame: int
    duration_frames: int
    fired_at_frame: int
    fire_score: float
    fire_breakdown: dict
    end_reason: str

    def to_airborne_event(self) -> AirborneEventV2:
        return AirborneEventV2(
            start_frame=self.start_frame,
            duration_frames=self.duration_frames,
            event_type="airborne",
            fired_at_frame=self.fired_at_frame,
            confidence_score=self.fire_score,
            score_breakdown=self.fire_breakdown,
        )


class AirborneStateMachineV3:
    GROUND = "GROUND"
    LAUNCH_CANDIDATE = "LAUNCH_CANDIDATE"
    AIRBORNE = "AIRBORNE"

    def __init__(self, rule_config: AirborneRuleV2Config, sm_config: StateMachineV3Config):
        self.rule_config = rule_config
        self.sm_config = sm_config

        self.state = self.GROUND
        self._recent_rows = []
        self._ground_y_history = deque(maxlen=sm_config.ground_baseline_window_frames)
        self.ground_baseline_y: Optional[float] = None

        self._candidate: Optional[LaunchCandidateV2] = None

        self._active_event_start: Optional[int] = None
        self._active_event_fired_at: Optional[int] = None
        self._active_event_score: Optional[float] = None
        self._active_event_breakdown: Optional[dict] = None
        self._active_event_baseline: Optional[float] = None
        self._min_y_during_airborne: Optional[float] = None
        self._near_ground_streak: int = 0
        self._cooldown_until: int = -1

    def _update_recent_rows(self, features_row):
        self._recent_rows.append(features_row)
        if len(self._recent_rows) > self.rule_config.ground_window_frames:
            self._recent_rows.pop(0)

    def _compute_recent_density(self, *, exclude_current: bool = False) -> float:
        rows = self._recent_rows[:-1] if exclude_current else self._recent_rows
        if not rows:
            return 0.0
        recent_n_dets = sum(1 for r in rows if int(r.get("frames_since_last_det", 999)) == 0)
        return recent_n_dets / max(1, len(rows))

    def _passes_consistency_check(self) -> bool:
        cf = self.rule_config.upward_velocity_consistency_frames
        ct = self.rule_config.upward_velocity_consistency_threshold
        recent_dys = [r.get("dy_per_frame_last_pair") for r in self._recent_rows[-cf:]]
        upward_count = sum(
            1 for dy in recent_dys
            if dy is not None
            and not (isinstance(dy, float) and np.isnan(dy))
            and float(dy) <= ct
        )
        return upward_count >= 2

    def _update_ground_baseline(self, features_row):
        """V3: Use player-detected baseline if available, else fall back to deque median."""
        # Primary: player-feet derived baseline from current frame
        player_baseline = features_row.get("player_ground_baseline_y")
        if player_baseline is not None and not (isinstance(player_baseline, float) and np.isnan(player_baseline)):
            self.ground_baseline_y = float(player_baseline)
            return

        # Secondary fallback: keep building deque from ball ground detections
        last_y = features_row.get("last_y")
        frames_since = int(features_row.get("frames_since_last_det", 999))
        if (
            last_y is not None
            and not (isinstance(last_y, float) and np.isnan(last_y))
            and frames_since == 0
        ):
            self._ground_y_history.append(float(last_y))
            if len(self._ground_y_history) >= 5:
                self.ground_baseline_y = float(np.median(self._ground_y_history))

    def _is_ball_near_ground(self, features_row) -> bool:
        if self.ground_baseline_y is None:
            return False
        last_y = features_row.get("last_y")
        frames_since = int(features_row.get("frames_since_last_det", 999))
        if last_y is None or (isinstance(last_y, float) and np.isnan(last_y)) or frames_since != 0:
            return False
        return abs(float(last_y) - self.ground_baseline_y) <= self.sm_config.landing_distance_to_baseline_px

    def _start_airborne(
        self,
        *,
        start_frame: int,
        fired_at_frame: int,
        score: float,
        breakdown: dict,
        baseline_y: Optional[float],
        current_last_y,
    ):
        self.state = self.AIRBORNE
        self._active_event_start = int(start_frame)
        self._active_event_fired_at = int(fired_at_frame)
        self._active_event_score = float(score)
        self._active_event_breakdown = dict(breakdown)
        self._active_event_baseline = baseline_y
        self._min_y_during_airborne = (
            float(current_last_y)
            if current_last_y is not None and not (isinstance(current_last_y, float) and np.isnan(current_last_y))
            else None
        )
        self._near_ground_streak = 0
        self._candidate = None

    def _maybe_start_launch_candidate(self, features_row):
        t = int(features_row["frame"])
        frames_since = int(features_row.get("frames_since_last_det", 999))
        if frames_since != 1:
            return
        last_y = features_row.get("last_y")
        last_frame = int(features_row.get("last_detection_frame", -1))
        if last_frame < 0 or last_y is None or (isinstance(last_y, float) and np.isnan(last_y)):
            return

        pre_gap_density = self._compute_recent_density(exclude_current=True)
        if pre_gap_density < self.rule_config.candidate_min_pre_gap_density:
            return

        self._candidate = LaunchCandidateV2(
            missing_start_frame=t,
            last_ground_frame=last_frame,
            last_ground_y=float(last_y),
            pre_gap_density=float(pre_gap_density),
            baseline_y=self.ground_baseline_y,
        )
        self.state = self.LAUNCH_CANDIDATE

    def step(self, features_row) -> Optional[CompletedAirborneEventV3]:
        t = int(features_row["frame"])
        self._update_recent_rows(features_row)
        completed_event = None

        if self.state == self.GROUND:
            self._update_ground_baseline(features_row)
            if t <= self._cooldown_until:
                return completed_event

            recent_density = self._compute_recent_density()
            result = evaluate_direct_launch_rule(features_row, recent_density, self.rule_config)
            if result is not None and self._passes_consistency_check():
                score, breakdown = result
                self._start_airborne(
                    start_frame=t,
                    fired_at_frame=t,
                    score=score,
                    breakdown=breakdown,
                    baseline_y=self.ground_baseline_y,
                    current_last_y=features_row.get("last_y"),
                )
                return completed_event

            self._maybe_start_launch_candidate(features_row)

        elif self.state == self.LAUNCH_CANDIDATE:
            candidate = self._candidate
            if candidate is None:
                self.state = self.GROUND
                return completed_event

            frames_since_ground = t - candidate.last_ground_frame
            if frames_since_ground > self.rule_config.candidate_max_missing_frames + 1:
                self._candidate = None
                self.state = self.GROUND
                self._update_ground_baseline(features_row)
                return completed_event

            if bool(features_row.get("current_has_detection", False)):
                result = evaluate_reappearance_launch_rule(features_row, candidate, self.rule_config)
                if result is not None:
                    score, breakdown = result
                    self._start_airborne(
                        start_frame=candidate.missing_start_frame,
                        fired_at_frame=t,
                        score=score,
                        breakdown=breakdown,
                        baseline_y=candidate.baseline_y,
                        current_last_y=features_row.get("last_y"),
                    )
                else:
                    self._candidate = None
                    self.state = self.GROUND
                    self._update_ground_baseline(features_row)

        elif self.state == self.AIRBORNE:
            self._update_ground_baseline(features_row)
            airborne_duration = t - int(self._active_event_start)

            last_y = features_row.get("last_y")
            frames_since = int(features_row.get("frames_since_last_det", 999))
            if (
                last_y is not None
                and not (isinstance(last_y, float) and np.isnan(last_y))
                and frames_since == 0
            ):
                if self._min_y_during_airborne is None:
                    self._min_y_during_airborne = float(last_y)
                else:
                    self._min_y_during_airborne = min(self._min_y_during_airborne, float(last_y))

            if self._active_event_baseline is not None and self._min_y_during_airborne is not None:
                ascent = self._active_event_baseline - self._min_y_during_airborne
            else:
                ascent = 0.0
            has_ascended = ascent >= self.sm_config.min_ascent_before_landing_px

            if has_ascended and self._is_ball_near_ground(features_row):
                self._near_ground_streak += 1
            else:
                self._near_ground_streak = 0

            if self._near_ground_streak >= self.sm_config.landing_consecutive_detections:
                completed_event = self._complete_event(t, "landed")
                self._cooldown_until = t + self.sm_config.post_event_cooldown_frames
                self.state = self.GROUND
                self._update_ground_baseline(features_row)
            elif airborne_duration > self.sm_config.max_airborne_frames:
                completed_event = self._complete_event(t, "max_duration")
                self._cooldown_until = t + self.sm_config.post_event_cooldown_frames
                self.state = self.GROUND

        return completed_event

    def _complete_event(self, end_frame: int, reason: str) -> CompletedAirborneEventV3:
        start = int(self._active_event_start)
        end = int(end_frame)
        completed = CompletedAirborneEventV3(
            start_frame=start,
            end_frame=end,
            duration_frames=end - start + 1,
            fired_at_frame=int(self._active_event_fired_at),
            fire_score=float(self._active_event_score),
            fire_breakdown=dict(self._active_event_breakdown),
            end_reason=reason,
        )
        self._reset_event_state()
        return completed

    def _reset_event_state(self):
        self._active_event_start = None
        self._active_event_fired_at = None
        self._active_event_score = None
        self._active_event_breakdown = None
        self._active_event_baseline = None
        self._min_y_during_airborne = None
        self._near_ground_streak = 0


def detect_airborne_events_with_state_machine_v3(
    features_df: pd.DataFrame,
    rule_config: AirborneRuleV2Config,
    sm_config: StateMachineV3Config,
) -> list[CompletedAirborneEventV3]:
    sm = AirborneStateMachineV3(rule_config, sm_config)
    completed_events = []

    feature_dicts = features_df.to_dict("records")
    for row in feature_dicts:
        completed = sm.step(row)
        if completed is not None:
            completed_events.append(completed)

    if sm.state == sm.AIRBORNE and sm._active_event_start is not None:
        last_frame = int(feature_dicts[-1]["frame"])
        completed_events.append(sm._complete_event(last_frame, "still_active_at_end_of_video"))

    return completed_events


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("features_csv")
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    events = detect_airborne_events_with_state_machine_v3(
        df,
        AirborneRuleV2Config(),
        StateMachineV3Config(),
    )
    for event in events:
        print(
            f"start={event.start_frame} end={event.end_frame} "
            f"fired_at={event.fired_at_frame} reason={event.end_reason} "
            f"score={event.fire_score:.2f} breakdown={event.fire_breakdown}"
        )
