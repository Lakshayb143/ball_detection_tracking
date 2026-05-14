import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium
from collections import deque


from dataclasses import dataclass
from typing import Optional


# ============================================================
# Config
# ============================================================
MODEL_PATH = "checkpoints/ball_1120.pth"
VIDEO_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1.mp4"
OUTPUT_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v3.mp4"
DETECTION_JSON_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v3.json"
CONFIDENCE = 0.01
BALL_CLASS_ID = 0
MODEL_RESOLUTION = 1120
ENABLE_RFDETR_OPTIMIZE = True
DEFAULT_FPS_IF_MISSING = 30.0
BENCHMARK_INTERPOLATED_BOX_SIZE = 20.0
BENCHMARK_MATCH_IOU = 0.01
BENCHMARK_EVAL_SCORE_THRESHOLD = 0.0
BENCHMARK_BALL_CATEGORY_ID = 1


# Gating
MAHALANOBIS_GATE = 20         # chi-squared 99% for 2 DOF
MAX_GAP_FRAMES = 13            # max consecutive frames to extrapolate before giving up

# Outlier confirmation
OUTLIER_CONFIRM_FRAMES = 4       # need this many consecutive outliers before resetting
HISTORY_FRAMES = 5

# Physics — pixels/sec^2 downward at the field plane.
# Calibrate from one clip: pick a clean shot/cross, fit a parabola to the ball,
# and read off the second derivative of y(t) in pixels per second^2.
# 600 px/s^2 is a reasonable starting guess for a 1080p broadcast wide shot.
GRAVITY_PX_PER_S2 = 789.0


MAX_GAP_BY_REGIME = {
    "High Pass": 60,
    "Cross":     60,
    "Shot":      50,
    "Free Kick": 60,
    "Goal":      40,
    "Header":    25,
    "Throw In":  40,
    "Pass":      15,
    "Drive":     15,
    "Ball Player Block":        15,
    "Player Successful Tackle": 15,
    "Out":       0,    # immediately give up if ball is out
    "Default":   13,   # no active regime
}
DEFAULT_MAX_GAP_FRAMES = 13


# ============================================================
# Action regime configuration
# ============================================================
# Each action triggers a "regime" that lasts for some number of frames
# after the event. During the regime, the tracker applies action-aware
# rejection and motion priors. After the regime ends, behavior reverts
# to the default (detector-trust) mode.


ACTIONS_JSON_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_actions.json"
ACTION_REJECTION_THRESHOLD = 2.0   # implausibility above this → reject

@dataclass
class ActionRegime:
    """Defines how the tracker should behave during this action's regime."""
    name: str
    duration_frames: int               # how long after the event this regime is active
    is_airborne: bool                  # does the ball spend most of this regime in the air?
    expects_high_velocity: bool        # does the ball reach high speeds (>500 px/s)?
    near_foot_required: bool           # should the ball be close to a player's foot?
    rejection_zone: Optional[str] = None  # "ground" / "player_low_body" / None


ACTION_REGIMES = {
    # Airborne actions — ball spends most of the regime above ground level
    "High Pass":   ActionRegime("High Pass",   duration_frames=60, is_airborne=True,  expects_high_velocity=True,  near_foot_required=False, rejection_zone="ground"),
    "Cross":       ActionRegime("Cross",       duration_frames=50, is_airborne=True,  expects_high_velocity=True,  near_foot_required=False, rejection_zone="ground"),
    "Shot":        ActionRegime("Shot",        duration_frames=40, is_airborne=True,  expects_high_velocity=True,  near_foot_required=False, rejection_zone="ground"),
    "Free Kick":   ActionRegime("Free Kick",   duration_frames=60, is_airborne=True,  expects_high_velocity=True,  near_foot_required=False, rejection_zone="ground"),
    "Goal":        ActionRegime("Goal",        duration_frames=40, is_airborne=True,  expects_high_velocity=True,  near_foot_required=False, rejection_zone=None),

    # Aerial-but-short — ball goes up briefly
    "Header":      ActionRegime("Header",      duration_frames=20, is_airborne=True,  expects_high_velocity=False, near_foot_required=False, rejection_zone=None),
    "Throw In":    ActionRegime("Throw In",    duration_frames=40, is_airborne=True,  expects_high_velocity=False, near_foot_required=False, rejection_zone=None),

    # Ground actions — ball stays near players
    "Pass":        ActionRegime("Pass",        duration_frames=30, is_airborne=False, expects_high_velocity=False, near_foot_required=True,  rejection_zone=None),
    "Drive":       ActionRegime("Drive",       duration_frames=30, is_airborne=False, expects_high_velocity=False, near_foot_required=True,  rejection_zone=None),

    # Contact events — ball changes possession or trajectory
    "Ball Player Block":        ActionRegime("Ball Player Block",        duration_frames=15, is_airborne=False, expects_high_velocity=False, near_foot_required=True, rejection_zone=None),
    "Player Successful Tackle": ActionRegime("Player Successful Tackle", duration_frames=15, is_airborne=False, expects_high_velocity=False, near_foot_required=True, rejection_zone=None),

    # Out of play — stop tracking
    "Out":         ActionRegime("Out",         duration_frames=999, is_airborne=False, expects_high_velocity=False, near_foot_required=False, rejection_zone=None),
}

# Default regime when no action event is active
DEFAULT_REGIME = ActionRegime("Default", duration_frames=0, is_airborne=False,
                              expects_high_velocity=False, near_foot_required=False,
                              rejection_zone=None)


# ============================================================
# Action timeline — loaded from JSON, queries by frame
# ============================================================
class ActionTimeline:
    """
    Loads instantaneous action events from a JSON file and, for any given
    frame, returns the currently active regime.

    JSON format:
        {
          "fps": 30,
          "events": [
            {"frame": 90,  "action": "High Pass"},
            {"frame": 121, "action": "Drive"},
            {"frame": 166, "action": "Shot"}
          ]
        }

    A regime is "active" between [event.frame, event.frame + regime.duration_frames).
    If multiple regimes overlap (e.g., a Header during a High Pass), the most recent
    event wins — the assumption is that any subsequent action implies the previous
    regime has ended.
    """

    def __init__(self, json_path: Optional[str] = None):
        self.events = []
        if json_path is not None and json_path != "":
            self._load(json_path)

    def _load(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)
        events = data.get("events", [])
        # Sort by frame ascending
        events.sort(key=lambda e: e["frame"])
        self.events = events
        print(f"[action_timeline] loaded {len(events)} events from {json_path}")

    def get_regime(self, frame_idx: int) -> ActionRegime:
        """Returns the active regime at this frame, or DEFAULT_REGIME if none."""
        # Find the most recent event whose regime hasn't expired yet
        active = None
        for evt in self.events:
            if evt["frame"] > frame_idx:
                break  # events are sorted; future events don't apply
            regime = ACTION_REGIMES.get(evt["action"], DEFAULT_REGIME)
            if frame_idx < evt["frame"] + regime.duration_frames:
                active = (evt, regime)
            # else: this regime has expired, but a later one might still be active
        if active is None:
            return DEFAULT_REGIME
        evt, regime = active
        # also useful: how many frames since this event fired
        return regime

    def get_active_event(self, frame_idx: int):
        """Returns (event_dict, regime, frames_since_event) or (None, DEFAULT, 0)."""
        for evt in reversed(self.events):
            if evt["frame"] > frame_idx:
                continue
            regime = ACTION_REGIMES.get(evt["action"], DEFAULT_REGIME)
            frames_since = frame_idx - evt["frame"]
            if frames_since < regime.duration_frames:
                return evt, regime, frames_since
        return None, DEFAULT_REGIME, 0


# ============================================================
# Action-conditional rejection
# ============================================================
class ActionConditionalFilter:
    """
    Given a candidate ball detection and the current action regime,
    decides whether to accept, reject, or downweight it.

    Returns an "implausibility score" in [0, inf):
      0     → fully consistent with current regime
      0–1   → mildly inconsistent
      >1    → reject (can be combined with Mahalanobis to form total rejection)

    Use the score additively with Mahalanobis distance, or as a hard reject
    above a threshold (e.g., > 2.0 → reject).
    """

    def __init__(self, frame_height: int):
        self.frame_height = frame_height
        # Frame fraction below which we consider "ground level" — tune per camera
        self.GROUND_FRACTION = 0.7
        # Pixel distance from predicted ball position above which "near-foot" fails
        # (only used if near_foot_required is True)
        self.NEAR_FOOT_PX = 80

    def score(self, detection_xy, kf_predicted_xy, kf_predicted_velocity,
              regime: ActionRegime, player_positions=None) -> float:
        """
        detection_xy:           candidate detection center (x, y) in pixels
        kf_predicted_xy:        KF prior at this frame (x, y) — None if track not initialized
        kf_predicted_velocity:  KF velocity prior (vx, vy) — None if track not initialized
        regime:                 current ActionRegime
        player_positions:       optional list of (x, y) player foot positions
        Returns: implausibility score (higher = more suspicious).
        """
        if regime.name == "Default":
            return 0.0  # no action context, trust detection

        score = 0.0
        det_x, det_y = float(detection_xy[0]), float(detection_xy[1])

        # Rule 1: Airborne regimes reject ground-level detections.
        # During a shot/cross/high-pass, a "ball detection" near the bottom
        # of the frame (likely a shoe) is highly suspect.
        if regime.is_airborne and regime.rejection_zone == "ground":
            ground_y = self.frame_height * self.GROUND_FRACTION
            if det_y > ground_y:
                # How far below the threshold? Linear penalty.
                penalty = (det_y - ground_y) / (self.frame_height - ground_y)
                score += 2.0 * penalty   # full penalty if at very bottom

        # Rule 2: If KF is tracking and current detection implies a velocity
        # discontinuity inconsistent with the regime, penalize.
        if kf_predicted_xy is not None and regime.is_airborne:
            displacement = np.linalg.norm(np.array([det_x, det_y]) - np.array(kf_predicted_xy))
            # During airborne regime, ball typically moves predictably under gravity.
            # A detection >150 px from prediction implies either a teleport or a wrong object.
            if displacement > 150:
                score += min(2.0, (displacement - 150) / 100.0)

        # Rule 3: Near-foot required (Pass/Drive). If detection is far from
        # any player, it's probably wrong. Only applied if we have player data.
        if regime.near_foot_required and player_positions is not None and len(player_positions) > 0:
            min_dist = min(
                np.linalg.norm(np.array([det_x, det_y]) - np.array([p[0], p[1]]))
                for p in player_positions
            )
            if min_dist > self.NEAR_FOOT_PX:
                score += 1.5 * (min_dist / self.NEAR_FOOT_PX - 1)

        # Rule 4: "Out" regime — ball has left play, reject everything.
        if regime.name == "Out":
            score += 10.0  # effectively always reject

        return float(score)

# ============================================================
# Kalman Filter — constant acceleration with gravity as control input
# ============================================================
class BallKalmanFilter:
    """
    State: [x, y, vx, vy, ax, ay] in pixels and pixels/sec.
    Gravity is applied as a control input on the y-axis acceleration component.
    """

    def __init__(self, dt, gravity_px_s2=GRAVITY_PX_PER_S2):
        self.dt = dt
        self.gravity = gravity_px_s2

        dt2 = 0.5 * dt ** 2
        self.A = np.array([
            [1, 0, dt, 0,  dt2, 0  ],
            [0, 1, 0,  dt, 0,   dt2],
            [0, 0, 1,  0,  dt,  0  ],
            [0, 0, 0,  1,  0,   dt ],
            [0, 0, 0,  0,  1,   0  ],
            [0, 0, 0,  0,  0,   1  ],
        ])

        # Control matrix: gravity acts as a constant downward acceleration.
        # In image coords, +y is down, so gravity is positive.
        self.B = np.array([[0], [dt2], [0], [dt], [0], [0]])
        self.u = np.array([[self.gravity]])

        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]])

        # Process noise — tuned so airborne motion is "mostly gravity" but
        # ground/contact moments have enough freedom for impulsive changes.
        self.Q = np.diag([1.0, 1.0, 1214.5, 1214.5, 5.0, 5.0]) # best till now
        # self.Q = np.diag([1.0, 1.0, 25.0, 25.0, 400.0, 400.0])

        # Measurement noise — RF-DETR is roughly 2–5 px on the ball center.
        self.R = np.eye(2) * 1339.0 # best till now
        # self.R = np.eye(2) * 9.0

        self.x_hat = np.zeros((6, 1))
        self.P = np.eye(6) * 1000

    def initialize(self, position, velocity=None):
        self.x_hat = np.zeros((6, 1))
        self.x_hat[0, 0] = position[0]
        self.x_hat[1, 0] = position[1]
        if velocity is not None:
            self.x_hat[2, 0] = velocity[0]
            self.x_hat[3, 0] = velocity[1]
        self.P = np.eye(6) * 100

    def predict(self):
        self.x_hat = self.A @ self.x_hat + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.position(), self.innovation_cov()

    def update(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape(2, 1)
        y = z - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def position(self):
        return self.x_hat[:2].flatten()

    def velocity(self):
        return self.x_hat[2:4].flatten()

    def innovation_cov(self):
        return self.H @ self.P @ self.H.T + self.R

    def mahalanobis(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape(2, 1)
        y = z - self.H @ self.x_hat
        S = self.innovation_cov()
        return float((y.T @ np.linalg.inv(S) @ y).item())


# ============================================================
# Outlier confirmation — only reset after sustained inconsistency
# ============================================================
class OutlierConfirmer:
    """
    A single bad-looking detection is not enough to reset the filter.
    We require N consecutive sustained outliers before declaring track loss.
    A 'sustained outlier' here means: detection exists but its Mahalanobis
    distance is way beyond the gate, AND a second nearby detection isn't
    pulling the filter back. We keep this minimal — most rejection happens
    via the Mahalanobis gate, not here.
    """

    def __init__(self, max_consecutive=OUTLIER_CONFIRM_FRAMES):
        self.max_consecutive = max_consecutive
        self.consecutive_outliers = 0

    def report(self, was_outlier):
        if was_outlier:
            self.consecutive_outliers += 1
        else:
            self.consecutive_outliers = 0
        return self.consecutive_outliers >= self.max_consecutive

    def reset(self):
        self.consecutive_outliers = 0


# ============================================================
# Video processor
# ============================================================
class VideoProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {VIDEO_PATH}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS_IF_MISSING
        self.dt = 1.0 / self.fps

        self.model = RFDETRMedium(pretrain_weights=MODEL_PATH, resolution=MODEL_RESOLUTION)
        if ENABLE_RFDETR_OPTIMIZE:
            self.model.optimize_for_inference()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        self.kf = BallKalmanFilter(dt=self.dt)
        self.outlier_confirmer = OutlierConfirmer()

        self.track_initialized = False
        self.frames_since_detection = 0     # consecutive frames with no accepted detection
        self.hit_streak = 0
        
        self.action_timeline = ActionTimeline(ACTIONS_JSON_PATH)
        self.action_filter = ActionConditionalFilter(frame_height=self.frame_height)

    # --------------------------------------------------------
    # Detection selection
    # --------------------------------------------------------
    # def _select_best_detection(self, ball_detections):
    #     """
    #     Returns the best detection given the current KF prediction.
    #     - If track is initialized: prefer detections inside the Mahalanobis gate,
    #       break ties by highest confidence.
    #     - If not initialized: just take the highest-confidence detection.
    #     - Returns None if nothing passes the confidence threshold.
    #     """
    #     if len(ball_detections.xyxy) == 0:
    #         return None

    #     conf_mask = ball_detections.confidence >= CONFIDENCE
    #     if not np.any(conf_mask):
    #         return None
    #     filtered = ball_detections[conf_mask]
    #     centers = filtered.get_anchors_coordinates(sv.Position.CENTER)

    #     if not self.track_initialized:
    #         idx = int(np.argmax(filtered.confidence))
    #         return {"center": centers[idx], "sv": filtered[idx:idx + 1], "in_gate": False}

    #     # Score each detection by Mahalanobis distance under current KF uncertainty
    #     m_dists = np.array([self.kf.mahalanobis(c) for c in centers])
    #     in_gate_mask = m_dists < MAHALANOBIS_GATE

    #     if np.any(in_gate_mask):
    #         # Among in-gate detections, pick the one with smallest Mahalanobis distance
    #         in_gate_indices = np.where(in_gate_mask)[0]
    #         best_in_gate = in_gate_indices[np.argmin(m_dists[in_gate_indices])]
    #         return {"center": centers[best_in_gate], "sv": filtered[best_in_gate:best_in_gate + 1], "in_gate": True}

    #     # No detection in gate. Return the highest-confidence one but flag it as outlier.
    #     # The outlier confirmer will decide whether to reset.
    #     idx = int(np.argmax(filtered.confidence))
    #     return {"center": centers[idx], "sv": filtered[idx:idx + 1], "in_gate": False}


    def _select_best_detection(self, ball_detections, frame_count):
        if len(ball_detections.xyxy) == 0:
            return None

        conf_mask = ball_detections.confidence >= CONFIDENCE
        if not np.any(conf_mask):
            return None
        filtered = ball_detections[conf_mask]
        centers = filtered.get_anchors_coordinates(sv.Position.CENTER)

        regime = self.action_timeline.get_regime(frame_count)

        if not self.track_initialized:
            # Even on init, reject if action says we're in "Out" regime
            if regime.name == "Out":
                return None
            idx = int(np.argmax(filtered.confidence))
            return {"center": centers[idx], "sv": filtered[idx:idx + 1], "in_gate": False}

        predicted_xy = self.kf.position()
        predicted_vel = self.kf.velocity()

        # Compute implausibility for each detection
        action_scores = np.array([
            self.action_filter.score(c, predicted_xy, predicted_vel, regime)
            for c in centers
        ])
        m_dists = np.array([self.kf.mahalanobis(c) for c in centers])

        # Hard reject: action says implausible
        valid_action_mask = action_scores < ACTION_REJECTION_THRESHOLD
        if not np.any(valid_action_mask):
            # All detections rejected by action filter
            print(f"Frame {frame_count}: ALL detections rejected by action filter (regime={regime.name})")
            return None

        # Among action-valid detections, pick by Mahalanobis (in-gate first, then closest)
        valid_indices = np.where(valid_action_mask)[0]
        in_gate_mask = m_dists[valid_indices] < MAHALANOBIS_GATE
        if np.any(in_gate_mask):
            best = valid_indices[in_gate_mask][np.argmin(m_dists[valid_indices[in_gate_mask]])]
            return {"center": centers[best], "sv": filtered[best:best + 1], "in_gate": True}
    
        # No in-gate after action filter — return best-by-confidence among action-valid
        best = valid_indices[np.argmax(filtered.confidence[valid_indices])]
        return {"center": centers[best], "sv": filtered[best:best + 1], "in_gate": False}

    
    def _max_gap_for_frame(self, frame_count):
        """Returns the active MAX_GAP_FRAMES based on current action regime."""
        regime = self.action_timeline.get_regime(frame_count)
        return MAX_GAP_BY_REGIME.get(regime.name, DEFAULT_MAX_GAP_FRAMES)

    # --------------------------------------------------------
    # Per-frame logic
    # --------------------------------------------------------
    def _step(self, best_detection, frame_count):
        """
        Returns: (output_position, is_interpolated, accepted_detection)
        - output_position: the position to display this frame (or None if track lost)
        - is_interpolated: True if no accepted detection this frame
        - accepted_detection: the supervision Detections object to draw a box on, or None
        """
        # Always run KF predict first — this advances the state by one frame
        # and applies gravity. Returns the prior for this frame.
        if self.track_initialized:
            predicted_pos, _ = self.kf.predict()
        else:
            predicted_pos = None

        max_gap = self._max_gap_for_frame(frame_count)   # NEW

        # Case A: no detection at all this frame
        if best_detection is None:
            if not self.track_initialized:
                return None, False, None
            self.frames_since_detection += 1
            self.hit_streak = 0
            if self.frames_since_detection > max_gap:        # CHANGED
                self.track_initialized = False
                self.outlier_confirmer.reset()
                print(f"Frame {frame_count}: track lost — gap exceeded {max_gap} frames during outlier streak")
                return None, False, None
            print(f"Frame {frame_count}: INTERPOLATED at {predicted_pos.round(1)}  (gap={self.frames_since_detection}/{max_gap})")
            return predicted_pos, True, None

        # Case B: detection exists
        center = best_detection["center"]
        in_gate = best_detection["in_gate"]

        if not self.track_initialized:
            # First detection — initialize KF with position only (no velocity yet)
            self.kf.initialize(center)
            self.track_initialized = True
            self.frames_since_detection = 0
            self.hit_streak = 1
            self.outlier_confirmer.reset()
            print(f"Frame {frame_count}: ACCEPTED (init) at {center.round(1)}")
            return center, False, best_detection["sv"]

        if in_gate:
            # Normal accepted detection
            self.kf.update(center)
            self.frames_since_detection = 0
            self.hit_streak += 1
            self.outlier_confirmer.reset()
            print(f"Frame {frame_count}: ACCEPTED at {center.round(1)}  m_dist passed gate")
            return self.kf.position(), False, best_detection["sv"]

        # Detection exists but outside the gate
        should_reset = self.outlier_confirmer.report(was_outlier=True)
        self.frames_since_detection += 1
        self.hit_streak = 0

        if should_reset:
            # Sustained inconsistency — re-initialize on the new detection
            self.kf.initialize(center)
            self.track_initialized = True
            self.frames_since_detection = 0
            self.hit_streak = 1
            self.outlier_confirmer.reset()
            print(f"Frame {frame_count}: RESET — re-initializing on outlier-streak detection at {center.round(1)}")
            return center, False, best_detection["sv"]

        if self.frames_since_detection > MAX_GAP_FRAMES:
            self.track_initialized = False
            self.outlier_confirmer.reset()
            print(f"Frame {frame_count}: track lost — gap exceeded {MAX_GAP_FRAMES} frames during outlier streak")
            return None, False, None

        print(f"Frame {frame_count}: REJECTED detection (outside gate) at {center.round(1)}; interpolating")
        return predicted_pos, True, None

    # --------------------------------------------------------
    # Annotation
    # --------------------------------------------------------
    def _annotate(self, frame, output_position, is_interpolated, accepted_detection):
        if accepted_detection is not None:
            frame = self.box_annotator.annotate(scene=frame, detections=accepted_detection)
            frame = self.label_annotator.annotate(scene=frame, detections=accepted_detection, labels=["Ball"])
        elif output_position is not None and is_interpolated:
            x, y = int(output_position[0]), int(output_position[1])
            cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)
            cv2.putText(frame, "INTERPOLATED", (x + 15, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame

    def _build_detection_record(self, frame_idx, accepted_detection):
        if accepted_detection is None or len(accepted_detection.xyxy) == 0:
            return {
                "frame_idx": int(frame_idx),
                "x": None,
                "y": None,
                "confidence": None,
            }

        center = accepted_detection.get_anchors_coordinates(sv.Position.CENTER)[0]
        confidence = accepted_detection.confidence[0]
        return {
            "frame_idx": int(frame_idx),
            "x": float(center[0]),
            "y": float(center[1]),
            "confidence": float(confidence),
        }

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------
    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, self.fps, (self.frame_width, self.frame_height))
        frame_count = 0
        detection_records = []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.model.predict(frame, confidence=CONFIDENCE)
            ball_detections = detections[detections.class_id == BALL_CLASS_ID]
            best_detection = self._select_best_detection(ball_detections, frame_count)
            output_position, is_interpolated, accepted = self._step(best_detection, frame_count)
            annotated = self._annotate(frame.copy(), output_position, is_interpolated, accepted)

            detection_records.append(self._build_detection_record(frame_count, accepted))
            out_writer.write(annotated)
            frame_count += 1

        out_writer.release()
        self.cap.release()
        with open(DETECTION_JSON_PATH, "w", encoding="utf-8") as json_file:
            json.dump(detection_records, json_file, indent=2)
        print(f"Detection JSON written to {DETECTION_JSON_PATH}")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    VideoProcessor().run()
