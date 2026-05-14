import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium
from collections import deque


# ============================================================
# Config
# ============================================================
MODEL_PATH = "checkpoints/ball_1120.pth"
VIDEO_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1.mp4"
OUTPUT_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output.mp4"
DETECTION_JSON_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output.json"
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
MAHALANOBIS_GATE = 20        # chi-squared 99% for 2 DOF
MAX_GAP_FRAMES = 13            # max consecutive frames to extrapolate before giving up

# Outlier confirmation
OUTLIER_CONFIRM_FRAMES = 4       # need this many consecutive outliers before resetting
HISTORY_FRAMES = 5

# Physics — pixels/sec^2 downward at the field plane.
# Calibrate from one clip: pick a clean shot/cross, fit a parabola to the ball,
# and read off the second derivative of y(t) in pixels per second^2.
# 600 px/s^2 is a reasonable starting guess for a 1080p broadcast wide shot.
GRAVITY_PX_PER_S2 = 789.0


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
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.dt = 1.0 / self.fps

        self.model = RFDETRMedium(pretrain_weights=MODEL_PATH, resolution=1120)
        self.model.optimize_for_inference()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        self.kf = BallKalmanFilter(dt=self.dt)
        self.outlier_confirmer = OutlierConfirmer()

        self.track_initialized = False
        self.frames_since_detection = 0     # consecutive frames with no accepted detection
        self.hit_streak = 0                 # consecutive frames with accepted detection

    # --------------------------------------------------------
    # Detection selection
    # --------------------------------------------------------
    def _select_best_detection(self, ball_detections):
        """
        Returns the best detection given the current KF prediction.
        - If track is initialized: prefer detections inside the Mahalanobis gate,
          break ties by highest confidence.
        - If not initialized: just take the highest-confidence detection.
        - Returns None if nothing passes the confidence threshold.
        """
        if len(ball_detections.xyxy) == 0:
            return None

        conf_mask = ball_detections.confidence >= CONFIDENCE
        if not np.any(conf_mask):
            return None
        filtered = ball_detections[conf_mask]
        centers = filtered.get_anchors_coordinates(sv.Position.CENTER)

        if not self.track_initialized:
            idx = int(np.argmax(filtered.confidence))
            return {"center": centers[idx], "sv": filtered[idx:idx + 1], "in_gate": False}

        # Score each detection by Mahalanobis distance under current KF uncertainty
        m_dists = np.array([self.kf.mahalanobis(c) for c in centers])
        in_gate_mask = m_dists < MAHALANOBIS_GATE

        if np.any(in_gate_mask):
            # Among in-gate detections, pick the one with smallest Mahalanobis distance
            in_gate_indices = np.where(in_gate_mask)[0]
            best_in_gate = in_gate_indices[np.argmin(m_dists[in_gate_indices])]
            return {"center": centers[best_in_gate], "sv": filtered[best_in_gate:best_in_gate + 1], "in_gate": True}

        # No detection in gate. Return the highest-confidence one but flag it as outlier.
        # The outlier confirmer will decide whether to reset.
        idx = int(np.argmax(filtered.confidence))
        return {"center": centers[idx], "sv": filtered[idx:idx + 1], "in_gate": False}

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

        # Case A: no detection at all this frame
        if best_detection is None:
            if not self.track_initialized:
                return None, False, None
            self.frames_since_detection += 1
            self.hit_streak = 0
            if self.frames_since_detection > MAX_GAP_FRAMES:
                # Give up — gap is too long
                self.track_initialized = False
                self.outlier_confirmer.reset()
                print(f"Frame {frame_count}: track lost — gap exceeded {MAX_GAP_FRAMES} frames")
                return None, False, None
            print(f"Frame {frame_count}: INTERPOLATED at {predicted_pos.round(1)}  (gap={self.frames_since_detection})")
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
            best_detection = self._select_best_detection(ball_detections)
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
