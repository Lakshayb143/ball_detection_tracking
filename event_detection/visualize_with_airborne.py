"""
Real-time visualization of v4 ball tracker gated by airborne state machine.

Shows:
  - Ball bbox (green): accepted detection during GROUND state
  - Red circle: KF-interpolated ball position during GROUND
  - "AIRBORNE EVENT" big yellow text overlay during AIRBORNE state (no ball annotations)
  - Top-left HUD: state, score, baseline_y, frame index
  - Thin horizontal line at ground baseline y for reference

Workflow per frame:
  1. Run ball detector
  2. Update trajectory features for this frame (online buffer)
  3. Step the state machine
  4. If state == AIRBORNE: skip tracker, render airborne overlay
  5. If state == GROUND: run tracker (KF predict/update/interpolate), render ball
"""

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium

# Reuse existing modules
from airborne_rule import AirborneRuleConfig
from airborne_state_machine import (
    AirborneStateMachine,
    StateMachineConfig,
    CompletedAirborneEvent,
)


# ============================================================
# Config
# ============================================================
BALL_MODEL_PATH = "/home/lakshay/lx/ball_detection_tracking/checkpoints/ball_samy_1120.pth"
VIDEO_PATH = "/home/lakshay/lx/ball_detection_tracking/france_vs_argentina/clip1.mp4"
OUTPUT_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_visualized.mp4"

CONFIDENCE = 0.01
BALL_CLASS_ID = 0
MODEL_RESOLUTION = 1120

# Trajectory feature buffer (must match what your rule was tuned for)
BUFFER_L = 30

# Tracker params (reuse the v4 baseline values, both phases off)
GRAVITY_PX_PER_S2 = 789.0
MAHALANOBIS_GATE = 20
MAX_GAP_FRAMES = 13
OUTLIER_CONFIRM_FRAMES = 4
DEFAULT_FPS = 30.0


# ============================================================
# Online trajectory feature buffer
# ============================================================
class OnlineFeatureBuffer:
    """
    Builds the same per-frame trajectory features as extract_trajectory_features.py
    but online, one frame at a time.
    """

    def __init__(self, buffer_l: int):
        self.buffer_l = buffer_l
        # Each entry is (frame_idx, x, y, conf) for frames with detections.
        # Frames without detections aren't added; we infer gaps from frame numbers.
        self.dets = deque(maxlen=buffer_l + 5)

    def step(self, frame_idx: int, ball_xy_conf):
        """Add this frame's detection (or None) and return feature row for this frame."""
        if ball_xy_conf is not None:
            x, y, conf = ball_xy_conf
            self.dets.append((frame_idx, float(x), float(y), float(conf)))

        # Trim out detections that fall outside the buffer window
        buffer_start = max(0, frame_idx - self.buffer_l + 1)
        while self.dets and self.dets[0][0] < buffer_start:
            self.dets.popleft()

        n = len(self.dets)
        ys = [d[2] for d in self.dets]

        if n > 0:
            median_y = float(np.median(ys))
            y_range = float(max(ys) - min(ys))
            last_frame, last_x, last_y, last_conf = self.dets[-1]
            frames_since = frame_idx - last_frame
        else:
            median_y = float("nan")
            y_range = float("nan")
            last_y = float("nan")
            last_x = float("nan")
            frames_since = self.buffer_l

        if n >= 2:
            f1, _, y1, _ = self.dets[-2]
            f2, _, y2, _ = self.dets[-1]
            dy_per_frame_last_pair = float((y2 - y1) / max(1, f2 - f1))
            dy_buffer_first_to_last = float(self.dets[-1][2] - self.dets[0][2])
        else:
            dy_per_frame_last_pair = float("nan")
            dy_buffer_first_to_last = float("nan")

        return {
            "frame": frame_idx,
            "n_detections_in_buffer": n,
            "detection_density": n / self.buffer_l,
            "median_y_in_buffer": median_y,
            "y_range_in_buffer": y_range,
            "frames_since_last_det": frames_since,
            "last_y": last_y,
            "dy_per_frame_last_pair": dy_per_frame_last_pair,
            "dy_buffer_first_to_last": dy_buffer_first_to_last,
            # Pose features default to "nothing" — wire in later if you want
            "pose_any_kicking": False,
            "pose_max_kicking_conf": 0.0,
            "pose_dist_ball_to_kicker": float("nan"),
        }


# ============================================================
# Minimal v4 tracker components (copied to keep this file self-contained)
# ============================================================
class BallKalmanFilter:
    def __init__(self, dt, gravity_px_s2=GRAVITY_PX_PER_S2):
        self.dt = dt
        self.gravity = gravity_px_s2
        dt2 = 0.5 * dt ** 2
        self.A = np.array([
            [1, 0, dt, 0,  dt2, 0],
            [0, 1, 0,  dt, 0,   dt2],
            [0, 0, 1,  0,  dt,  0],
            [0, 0, 0,  1,  0,   dt],
            [0, 0, 0,  0,  1,   0],
            [0, 0, 0,  0,  0,   1],
        ])
        self.B = np.array([[0], [dt2], [0], [dt], [0], [0]])
        self.u = np.array([[self.gravity]])
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        self.Q = np.diag([1.0, 1.0, 1214.5, 1214.5, 5.0, 5.0])
        self.R = np.eye(2) * 1339.0
        self.x_hat = np.zeros((6, 1))
        self.P = np.eye(6) * 1000

    def initialize(self, position):
        self.x_hat = np.zeros((6, 1))
        self.x_hat[0, 0] = position[0]
        self.x_hat[1, 0] = position[1]
        self.P = np.eye(6) * 100

    def predict(self):
        self.x_hat = self.A @ self.x_hat + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_hat[:2].flatten()

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(2, 1)
        y = z - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def position(self):
        return self.x_hat[:2].flatten()

    def mahalanobis(self, z):
        z = np.asarray(z, dtype=float).reshape(2, 1)
        y = z - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.R
        return float((y.T @ np.linalg.inv(S) @ y).item())


class OutlierConfirmer:
    def __init__(self, max_consecutive=OUTLIER_CONFIRM_FRAMES):
        self.max_consecutive = max_consecutive
        self.n = 0

    def report(self, was_outlier):
        self.n = self.n + 1 if was_outlier else 0
        return self.n >= self.max_consecutive

    def reset(self):
        self.n = 0


class GroundTracker:
    """v4 baseline tracker (both phases off): KF + Mahalanobis gate + outlier confirmation."""
    def __init__(self, dt):
        self.kf = BallKalmanFilter(dt=dt)
        self.outlier_confirmer = OutlierConfirmer()
        self.track_initialized = False
        self.frames_since_detection = 0

    def reset(self):
        self.kf = BallKalmanFilter(dt=self.kf.dt)
        self.outlier_confirmer = OutlierConfirmer()
        self.track_initialized = False
        self.frames_since_detection = 0

    def step(self, ball_detections):
        """Returns (output_position, is_interpolated, accepted_sv) or (None, False, None)."""
        if self.track_initialized:
            predicted = self.kf.predict()
        else:
            predicted = None

        if len(ball_detections.xyxy) == 0:
            if not self.track_initialized:
                return None, False, None
            self.frames_since_detection += 1
            if self.frames_since_detection > MAX_GAP_FRAMES:
                self.track_initialized = False
                self.outlier_confirmer.reset()
                return None, False, None
            return predicted, True, None

        conf_mask = ball_detections.confidence >= CONFIDENCE
        if not np.any(conf_mask):
            if self.track_initialized:
                self.frames_since_detection += 1
                if self.frames_since_detection > MAX_GAP_FRAMES:
                    self.track_initialized = False
                return predicted if self.track_initialized else None, True, None
            return None, False, None

        filtered = ball_detections[conf_mask]
        centers = filtered.get_anchors_coordinates(sv.Position.CENTER)

        if not self.track_initialized:
            idx = int(np.argmax(filtered.confidence))
            center = centers[idx]
            self.kf.initialize(center)
            self.track_initialized = True
            self.frames_since_detection = 0
            self.outlier_confirmer.reset()
            return center, False, filtered[idx:idx+1]

        m_dists = np.array([self.kf.mahalanobis(c) for c in centers])
        in_gate_mask = m_dists < MAHALANOBIS_GATE
        if np.any(in_gate_mask):
            in_gate_indices = np.where(in_gate_mask)[0]
            best = in_gate_indices[np.argmin(m_dists[in_gate_indices])]
            center = centers[best]
            self.kf.update(center)
            self.frames_since_detection = 0
            self.outlier_confirmer.reset()
            return self.kf.position(), False, filtered[best:best+1]

        # Outside gate
        idx = int(np.argmax(filtered.confidence))
        center = centers[idx]
        self.frames_since_detection += 1
        should_reset = self.outlier_confirmer.report(was_outlier=True)
        if should_reset:
            self.kf.initialize(center)
            self.frames_since_detection = 0
            self.outlier_confirmer.reset()
            return center, False, filtered[idx:idx+1]
        if self.frames_since_detection > MAX_GAP_FRAMES:
            self.track_initialized = False
            self.outlier_confirmer.reset()
            return None, False, None
        return predicted, True, None


# ============================================================
# Annotation
# ============================================================
def annotate_frame(
    frame: np.ndarray,
    state: str,
    state_machine: AirborneStateMachine,
    frame_idx: int,
    output_position,
    is_interpolated: bool,
    accepted_sv,
    box_annotator,
    label_annotator,
    completed_event_just_now,
):
    if state == AirborneStateMachine.AIRBORNE:
        # No ball annotations during airborne. Big yellow text overlay.
        h, w = frame.shape[:2]
        elapsed = frame_idx - state_machine._active_event_start

        # Yellow translucent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 3), (w, 2 * h // 3), (0, 200, 255), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # Big text
        text = "AIRBORNE EVENT"
        sub_text = f"frame {elapsed} / {state_machine.sm_config.max_airborne_frames}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 2.5, 6)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        cv2.putText(frame, text, (text_x, text_y), font, 2.5, (0, 0, 0), 8)
        cv2.putText(frame, text, (text_x, text_y), font, 2.5, (0, 220, 255), 4)
        sub_size = cv2.getTextSize(sub_text, font, 1.0, 3)[0]
        cv2.putText(frame, sub_text, ((w - sub_size[0]) // 2, text_y + 50),
                    font, 1.0, (0, 0, 0), 5)
        cv2.putText(frame, sub_text, ((w - sub_size[0]) // 2, text_y + 50),
                    font, 1.0, (255, 255, 255), 2)

    else:
        # GROUND or UNCERTAIN: render ball detection or interpolation
        if accepted_sv is not None and len(accepted_sv.xyxy) > 0:
            frame = box_annotator.annotate(scene=frame, detections=accepted_sv)
            frame = label_annotator.annotate(
                scene=frame, detections=accepted_sv, labels=["Ball"]
            )
        elif output_position is not None and is_interpolated:
            x, y = int(output_position[0]), int(output_position[1])
            cv2.circle(frame, (x, y), 12, (0, 0, 255), 2)
            cv2.putText(frame, "INTERPOLATED", (x + 15, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Ground baseline reference line (always shown)
    if state_machine.ground_baseline_y is not None:
        baseline_y = int(state_machine.ground_baseline_y)
        cv2.line(frame, (0, baseline_y), (frame.shape[1], baseline_y),
                 (200, 200, 200), 1, lineType=cv2.LINE_AA)
        cv2.putText(frame, "ground baseline", (10, baseline_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Top-left HUD
    hud_lines = [
        f"Frame: {frame_idx}",
        f"State: {state}",
        f"Baseline y: {int(state_machine.ground_baseline_y) if state_machine.ground_baseline_y else 'n/a'}",
    ]
    if state == AirborneStateMachine.AIRBORNE and state_machine._active_event_score:
        hud_lines.append(f"Fire score: {state_machine._active_event_score:.2f}")

    pad = 10
    line_h = 25
    box_h = pad * 2 + line_h * len(hud_lines)
    cv2.rectangle(frame, (0, 0), (320, box_h), (0, 0, 0), -1)
    for i, line in enumerate(hud_lines):
        cv2.putText(frame, line, (pad, pad + line_h * (i + 1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Brief flash if an event just completed
    if completed_event_just_now is not None:
        msg = f"EVENT ENDED ({completed_event_just_now.duration_frames} fr, {completed_event_just_now.end_reason})"
        cv2.putText(frame, msg, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


# ============================================================
# Main
# ============================================================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Could not open {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Loading ball model: {BALL_MODEL_PATH}")
    model = RFDETRMedium(pretrain_weights=BALL_MODEL_PATH, resolution=MODEL_RESOLUTION)
    model.optimize_for_inference()

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    feature_buffer = OnlineFeatureBuffer(buffer_l=BUFFER_L)
    state_machine = AirborneStateMachine(
        rule_config=AirborneRuleConfig(),
        sm_config=StateMachineConfig(),
    )
    tracker = GroundTracker(dt=1.0 / fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    print(f"\nProcessing {n_frames} frames...")
    completed_events_log = []
    last_completion = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Run ball detector
        detections = model.predict(frame, confidence=CONFIDENCE)
        ball_detections = detections[detections.class_id == BALL_CLASS_ID]

        # 2. Build feature row for state machine
        if len(ball_detections.xyxy) > 0:
            best_idx = int(np.argmax(ball_detections.confidence))
            best = ball_detections[best_idx:best_idx+1]
            center = best.get_anchors_coordinates(sv.Position.CENTER)[0]
            ball_xy_conf = (float(center[0]), float(center[1]), float(best.confidence[0]))
        else:
            ball_xy_conf = None

        features_row = feature_buffer.step(frame_idx, ball_xy_conf)

        # 3. Step state machine
        completed_event = state_machine.step(features_row)
        if completed_event is not None:
            completed_events_log.append(completed_event)
            last_completion = completed_event
            print(f"  [event] start={completed_event.start_frame} end={completed_event.end_frame} "
                  f"duration={completed_event.duration_frames} reason={completed_event.end_reason}")

        # 4. If airborne, suppress tracker. Otherwise step tracker.
        if state_machine.state == AirborneStateMachine.AIRBORNE:
            # Reset tracker so we don't accumulate stale state across the airborne window.
            # The next ground frame will re-initialize it cleanly.
            if tracker.track_initialized:
                tracker.reset()
            output_position = None
            is_interpolated = False
            accepted_sv = None
        else:
            output_position, is_interpolated, accepted_sv = tracker.step(ball_detections)

        # 5. Annotate
        completion_to_show = last_completion if (last_completion and frame_idx - last_completion.end_frame < 15) else None
        annotated = annotate_frame(
            frame=frame.copy(),
            state=state_machine.state,
            state_machine=state_machine,
            frame_idx=frame_idx,
            output_position=output_position,
            is_interpolated=is_interpolated,
            accepted_sv=accepted_sv,
            box_annotator=box_annotator,
            label_annotator=label_annotator,
            completed_event_just_now=None,   # we use last_completion instead, sticky for 15 frames
        )

        out_writer.write(annotated)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  frame {frame_idx}/{n_frames} state={state_machine.state}")

    out_writer.release()
    cap.release()

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total frames: {frame_idx}")
    print(f"Detected airborne events: {len(completed_events_log)}")
    for e in completed_events_log:
        print(f"  start={e.start_frame:>4} end={e.end_frame:>4} "
              f"duration={e.duration_frames:>3} reason={e.end_reason}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()