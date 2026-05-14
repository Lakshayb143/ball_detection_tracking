from __future__ import annotations

from collections import deque

import numpy as np
import supervision as sv


BALL_MODEL_PATH = "checkpoints/ball_samy_1120.pth"
PLAYER_MODEL_PATH = "checkpoints/player.pth"
IMAGE_DIR_PATH = "train"
OUTPUT_PATH = "clip1_fresh_runs/ball_tracking_lb_preview.mp4"
PREDICTION_FILE_PATH = "clip1_fresh_runs/ball_tracking_lb_predictions.txt"

BALL_MODEL_RESOLUTION = 1120
PLAYER_MODEL_RESOLUTION = 0
ENABLE_RFDETR_OPTIMIZE = True

CONFIDENCE = 0.1
PLAYER_CONFIDENCE = 0.5
BALL_CLASS_ID = 0
PLAYER_CLASS_IDS = [1, 2, 3]
FPS = 30
BALL_TRACKER_BUFFER_SIZE = 13

POSITION_THRESHOLD = 50.0
VELOCITY_THRESHOLD = 100.0
HISTORY_FRAMES = 3

INTERPOLATION_VELOCITY_THRESHOLD = 20.0
MAX_INTERPOLATION_FRAMES = 2

MAX_OPTICAL_FLOW_GAP = 4
OPTICAL_FLOW_ERROR_THRESHOLD = 30.0
OPTICAL_FLOW_MAX_MOVEMENT = 10.0
OPTICAL_FLOW_WIN_SIZE = (15, 15)
OPTICAL_FLOW_MAX_LEVEL = 2
OPTICAL_FLOW_CRITERIA_EPS = 0.03
OPTICAL_FLOW_CRITERIA_COUNT = 10

MAX_LOST_SECONDS = 0.75
STABLE_TRACK_THRESHOLD = 5
VALIDATION_GATE_THRESHOLD = 25.0
INTERPOLATED_BOX_SIZE = 20.0

DEFAULT_FPS_IF_MISSING = float(FPS)
BENCHMARK_INTERPOLATED_BOX_SIZE = float(INTERPOLATED_BOX_SIZE)
BENCHMARK_MATCH_IOU = 0.01
BENCHMARK_EVAL_SCORE_THRESHOLD = 0.0
BENCHMARK_BALL_CATEGORY_ID = 1


def is_point_in_boxes(point, boxes: np.ndarray) -> bool:
    if boxes.size == 0:
        return False
    return bool(
        np.any(
            (point[0] >= boxes[:, 0])
            & (point[0] <= boxes[:, 2])
            & (point[1] >= boxes[:, 1])
            & (point[1] <= boxes[:, 3])
        )
    )


class BallTracker:
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return sv.Detections.empty()

        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)
        if not self.buffer:
            return sv.Detections.empty()

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = int(np.argmin(distances))
        return detections[[index]]

    def reset(self) -> None:
        self.buffer.clear()


class OutlierDetector:
    def __init__(self, position_threshold=50.0, velocity_threshold=100.0, max_frames=3):
        self.position_threshold = position_threshold
        self.velocity_threshold = velocity_threshold
        self.position_buffer = deque(maxlen=max_frames)
        self.velocity_buffer = deque(maxlen=max_frames)
        self.outlier_frames = 0
        self.outlier_wait_frames = 4
        self.tracking_suspended = False
        self.suspension_frames = 0
        self.max_suspension_frames = 3

    def add_frame(self, position, velocity=None):
        self.position_buffer.append(position.copy())
        if velocity is not None:
            self.velocity_buffer.append(velocity.copy())

    def is_outlier(self, new_position, new_velocity=None):
        if len(self.position_buffer) < 3:
            return False, True

        avg_position = np.mean(self.position_buffer, axis=0)
        is_position_outlier = np.linalg.norm(new_position - avg_position) > self.position_threshold

        is_velocity_outlier = False
        if new_velocity is not None and len(self.velocity_buffer) >= 2:
            avg_velocity = np.mean(self.velocity_buffer, axis=0)
            is_velocity_outlier = np.linalg.norm(new_velocity - avg_velocity) > self.velocity_threshold

        is_outlier = bool(is_position_outlier or is_velocity_outlier)
        if is_outlier:
            self.outlier_frames += 1
            if self.outlier_frames >= self.outlier_wait_frames:
                self._reset_tracking()
                return True, False
            self.tracking_suspended = True
            self.suspension_frames = self.outlier_frames
            return True, False

        self.outlier_frames = 0
        self.tracking_suspended = False
        self.suspension_frames = 0
        return False, True

    def _reset_tracking(self):
        self.outlier_frames = 0
        self.tracking_suspended = False
        self.suspension_frames = 0
        self.position_buffer.clear()
        self.velocity_buffer.clear()

    def should_reset_tracking(self):
        if self.tracking_suspended and self.suspension_frames >= self.max_suspension_frames:
            self._reset_tracking()
            return True
        return False


class OpticalKalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        dt2 = 0.5 * dt**2
        self.A = np.array(
            [
                [1, 0, dt, 0, dt2, 0],
                [0, 1, 0, dt, 0, dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        self.Q = np.eye(6, dtype=np.float32) * 0.1
        self.R = np.eye(2, dtype=np.float32) * 5.0
        self.x_hat = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0

    def predict(self):
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_hat[:2].flatten()

    def update(self, measurement):
        measurement = measurement.reshape(2, 1)
        residual = measurement - self.H @ self.x_hat
        innovation = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation)
        self.x_hat = self.x_hat + kalman_gain @ residual
        self.P = (np.eye(6, dtype=np.float32) - kalman_gain @ self.H) @ self.P

    def initialize_state(self, measurement):
        self.x_hat.fill(0.0)
        self.x_hat[:2] = measurement.reshape(2, 1)
        self.P = np.eye(6, dtype=np.float32) * 100.0

    def set_process_noise(self, accel_noise):
        self.Q[4, 4] = self.Q[5, 5] = accel_noise


class InterpolationTracker:
    def __init__(self, velocity_threshold=50.0, max_gap_frames=2):
        self.velocity_threshold = velocity_threshold
        self.max_gap_frames = max_gap_frames
        self.interpolation_kf = OpticalKalmanFilter(dt=1.0 / FPS)
        self.accepted_positions = deque(maxlen=5)
        self.interpolation_active = False
        self.interpolation_frames_remaining = 0
        self.last_accepted_position = None
        self.last_accepted_velocity = None

    def add_accepted_prediction(self, position, velocity=None):
        self.accepted_positions.append(position.copy())
        self.last_accepted_position = position.copy()
        if velocity is not None:
            self.last_accepted_velocity = velocity.copy()
        self.interpolation_active = False
        self.interpolation_frames_remaining = 0

    def should_interpolate(self, current_velocity=None):
        if len(self.accepted_positions) < 2:
            return False
        if current_velocity is not None and np.linalg.norm(current_velocity) > self.velocity_threshold:
            return False
        return True

    def start_interpolation(self):
        if len(self.accepted_positions) >= 2 and self.last_accepted_position is not None:
            self.interpolation_kf.initialize_state(self.last_accepted_position)
            self.interpolation_active = True
            self.interpolation_frames_remaining = self.max_gap_frames
            return True
        return False

    def get_interpolated_position(self):
        if not self.interpolation_active or self.interpolation_frames_remaining <= 0:
            return None
        predicted_pos = self.interpolation_kf.predict()
        self.interpolation_frames_remaining -= 1
        if self.interpolation_frames_remaining <= 0:
            self.interpolation_active = False
        return predicted_pos

    def stop_interpolation(self):
        self.interpolation_active = False
        self.interpolation_frames_remaining = 0


class AdaptiveKalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        dt2 = 0.5 * dt**2
        self.A = np.array(
            [
                [1, 0, dt, 0, dt2, 0],
                [0, 1, 0, dt, 0, dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        self.Q = np.eye(6, dtype=np.float32) * 0.1
        self.R = np.eye(2, dtype=np.float32) * 5.0
        self.x_hat = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0

    def predict(self):
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_hat[:2].flatten()

    def set_process_noise(self, accel_noise):
        self.Q[4, 4] = self.Q[5, 5] = accel_noise

    def update(self, measurement):
        measurement = measurement.reshape(2, 1)
        residual = measurement - self.H @ self.x_hat
        innovation = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation)
        self.x_hat = self.x_hat + kalman_gain @ residual
        self.P = (np.eye(6, dtype=np.float32) - kalman_gain @ self.H) @ self.P

    def initialize_state(self, measurement):
        self.x_hat.fill(0.0)
        self.x_hat[:2] = measurement.reshape(2, 1)
        self.P = np.eye(6, dtype=np.float32) * 100.0
