import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium


# ============================================================
# Config
# ============================================================
MODEL_PATH = "checkpoints/ball_samy_1120.pth"
PLAYER_MODEL_PATH = "checkpoints/player.pth"
VIDEO_PATH = "/home/lakshay/lx/ball_detection_tracking/france_vs_argentina/clip2.mp4"
OUTPUT_PATH = "/home/lakshay/lx/ball_detection_tracking/clip2_output_v5.mp4"
DETECTION_JSON_PATH = "/home/lakshay/lx/ball_detection_tracking/clip2_v5.json"
ACTIONS_JSON_PATH = "/home/lakshay/lx/ball_detection_tracking/clip2_actions.json"

CONFIDENCE = 0.01
PLAYER_CONFIDENCE = 0.5
BALL_CLASS_ID = 0
# Player-related class IDs to use for the airborne-overlap rejection rule.
# The script treats player+referee as "player" and excludes goalkeeper
# from the concern. Adjust the list below to match your player model's class ids.
PLAYER_CLASS_IDS = (1, 2, 3)              # goalkeeper, player, referee (skip the parent class 0)
PLAYER_REJECTION_CLASS_IDS = (2,)       # player and referee — excludes goalkeeper

MODEL_RESOLUTION = 1120
ENABLE_RFDETR_OPTIMIZE = True
DEFAULT_FPS_IF_MISSING = 30.0
BENCHMARK_INTERPOLATED_BOX_SIZE = 20.0
BENCHMARK_MATCH_IOU = 0.01
BENCHMARK_EVAL_SCORE_THRESHOLD = 0.0
BENCHMARK_BALL_CATEGORY_ID = 1

# Gating — Euclidean distance (pixels) from last accepted position.
# A detection farther than this is treated as an outlier candidate.
EUCLIDEAN_GATE = 100

# Outlier confirmation
OUTLIER_CONFIRM_FRAMES = 4


# ============================================================
# Ablation switches
# ============================================================
ENABLE_PHASE_1 = False    # Airborne regime: reject ball detections inside player bboxes
ENABLE_PHASE_2 = False    # Per-regime max_gap

# Used when Phase 2 is OFF — these are your previous baseline values
BASELINE_MAX_GAP_FRAMES = 13


# ============================================================
# Phase 2: action-regime -> max-gap
# ============================================================
@dataclass
class RegimeDynamics:
    """Action-conditional gap settings."""

    max_gap: int


REGIME_DYNAMICS = {
    "High Pass":                RegimeDynamics(max_gap=60),
    "Cross":                    RegimeDynamics(max_gap=60),
    "Shot":                     RegimeDynamics(max_gap=50),
    "Free Kick":                RegimeDynamics(max_gap=60),
    "Goal":                     RegimeDynamics(max_gap=40),
    "Header":                   RegimeDynamics(max_gap=25),
    "Throw In":                 RegimeDynamics(max_gap=40),
    "Pass":                     RegimeDynamics(max_gap=15),
    "Drive":                    RegimeDynamics(max_gap=15),
    "Ball Player Block":        RegimeDynamics(max_gap=15),
    "Player Successful Tackle": RegimeDynamics(max_gap=15),
    "Out":                      RegimeDynamics(max_gap=0),
    "Default":                  RegimeDynamics(max_gap=13),
}

# Which regimes count as "airborne" for Phase 1's player-overlap rejection rule.
AIRBORNE_REGIMES = {"High Pass", "airborne"}


# ============================================================
# Action regime metadata
# ============================================================
@dataclass
class ActionRegime:
    name: str
    duration_frames: int


ACTION_REGIMES = {
    # new unified label — duration ignored (end_frame is explicit in the JSON)
    "airborne": ActionRegime("airborne", duration_frames=0),
    # legacy labels kept for backward compat with any old-format JSONs
    "High Pass": ActionRegime("High Pass", duration_frames=60),
    "Cross": ActionRegime("Cross", duration_frames=50),
    "Shot": ActionRegime("Shot", duration_frames=40),
    "Free Kick": ActionRegime("Free Kick", duration_frames=60),
    "Goal": ActionRegime("Goal", duration_frames=40),
    "Header": ActionRegime("Header", duration_frames=20),
    "Throw In": ActionRegime("Throw In", duration_frames=40),
    "Pass": ActionRegime("Pass", duration_frames=30),
    "Drive": ActionRegime("Drive", duration_frames=30),
    "Ball Player Block": ActionRegime("Ball Player Block", duration_frames=15),
    "Player Successful Tackle": ActionRegime(
        "Player Successful Tackle", duration_frames=15
    ),
    "Out": ActionRegime("Out", duration_frames=999),
}
DEFAULT_REGIME = ActionRegime("Default", duration_frames=0)


class ActionTimeline:
    def __init__(self, json_path: Optional[str] = None):
        self.events = []
        if json_path:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("events", [])
            normalised = []
            for evt in raw:
                if "start_frame" in evt:
                    # new format: explicit span
                    normalised.append({
                        "start_frame": int(evt["start_frame"]),
                        "end_frame":   int(evt["end_frame"]),
                        "action":      evt["action"],
                    })
                else:
                    # old format: derive end from action duration
                    start = int(evt["frame"])
                    dur   = ACTION_REGIMES.get(evt["action"], DEFAULT_REGIME).duration_frames
                    normalised.append({
                        "start_frame": start,
                        "end_frame":   start + dur,
                        "action":      evt["action"],
                    })
            self.events = sorted(normalised, key=lambda e: e["start_frame"])
            print(f"[action_timeline] loaded {len(self.events)} events from {json_path}")

    def get_regime(self, frame_idx: int) -> ActionRegime:
        active = None
        for evt in self.events:
            if evt["start_frame"] > frame_idx:
                break
            if frame_idx <= evt["end_frame"]:
                active = ACTION_REGIMES.get(evt["action"], DEFAULT_REGIME)
        return active if active else DEFAULT_REGIME


# ============================================================
# Outlier confirmer
# ============================================================
class OutlierConfirmer:
    def __init__(self, max_consecutive=OUTLIER_CONFIRM_FRAMES):
        self.max_consecutive = max_consecutive
        self.consecutive_outliers = 0

    def report(self, was_outlier):
        self.consecutive_outliers = self.consecutive_outliers + 1 if was_outlier else 0
        return self.consecutive_outliers >= self.max_consecutive

    def reset(self):
        self.consecutive_outliers = 0


# ============================================================
# Phase 1: detection-inside-player check
# ============================================================
def detection_inside_any_player(det_center, player_bboxes) -> bool:
    """
    Returns True if the detection center falls inside any player bbox.
    Used during airborne regimes to reject ball detections that are most likely
    a player's shoe/sock/jersey rather than the actual ball.
    """
    if len(player_bboxes) == 0:
        return False
    cx, cy = float(det_center[0]), float(det_center[1])
    for x1, y1, x2, y2 in player_bboxes:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


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

        self.phase1_rejection_log = []

        self.model = RFDETRMedium(pretrain_weights=MODEL_PATH, resolution=MODEL_RESOLUTION)
        if ENABLE_RFDETR_OPTIMIZE:
            self.model.optimize_for_inference()

        self.player_model = RFDETRMedium(pretrain_weights=PLAYER_MODEL_PATH)
        if ENABLE_RFDETR_OPTIMIZE:
            try:
                self.player_model.optimize_for_inference()
            except Exception as e:
                print(f"[WARN] could not optimize player model: {e}")

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        self.outlier_confirmer = OutlierConfirmer()

        self.track_initialized = False
        self.last_position: Optional[np.ndarray] = None
        self.frames_since_detection = 0
        self.hit_streak = 0

        self.action_timeline = ActionTimeline(ACTIONS_JSON_PATH)
        self.current_max_gap = REGIME_DYNAMICS["Default"].max_gap

        # Diagnostic counters
        self.phase1_rejections = 0
        self.phase1_active_frames = 0

    def _apply_regime(self, frame_count):
        regime = self.action_timeline.get_regime(frame_count)

        if ENABLE_PHASE_2:
            dynamics = REGIME_DYNAMICS.get(regime.name, REGIME_DYNAMICS["Default"])
            self.current_max_gap = dynamics.max_gap
        else:
            self.current_max_gap = BASELINE_MAX_GAP_FRAMES

        return regime

    def _get_player_bboxes(self, frame):
        """
        Runs the player detector and returns rejection-relevant player bboxes
        (xyxy), excluding goalkeeper from the rejection logic.
        """
        detections = self.player_model.predict(frame, confidence=PLAYER_CONFIDENCE)
        if len(detections) == 0:
            return np.empty((0, 4), dtype=np.float32)
        keep = np.isin(detections.class_id, PLAYER_REJECTION_CLASS_IDS)
        if not np.any(keep):
            return np.empty((0, 4), dtype=np.float32)
        return detections.xyxy[keep].astype(np.float32)

    def _select_best_detection(self, ball_detections, player_bboxes, regime, frame_count):
        if len(ball_detections.xyxy) == 0:
            return None

        conf_mask = ball_detections.confidence >= CONFIDENCE
        if not np.any(conf_mask):
            return None
        filtered = ball_detections[conf_mask]
        centers = filtered.get_anchors_coordinates(sv.Position.CENTER)

        # Phase 1: airborne-regime player-overlap rejection.
        if ENABLE_PHASE_1 and regime.name in AIRBORNE_REGIMES and len(player_bboxes) > 0:
            self.phase1_active_frames += 1
            keep_indices = []
            for i, center in enumerate(centers):
                if not detection_inside_any_player(center, player_bboxes):
                    keep_indices.append(i)
                else:
                    self.phase1_rejections += 1
                    rejection_entry = {
                        "frame": int(frame_count),
                        "regime": regime.name,
                        "rejected_x": float(center[0]),
                        "rejected_y": float(center[1]),
                        "rejected_conf": float(filtered.confidence[i]),
                        "n_other_candidates": int(len(centers) - 1),
                    }
                    self.phase1_rejection_log.append(rejection_entry)
                    print(
                        f"Frame {frame_count}: PHASE1 REJECT - det at "
                        f"{center.round(1)} inside player bbox during {regime.name}"
                    )
            if not keep_indices:
                return None
            filtered = filtered[keep_indices]
            centers = centers[keep_indices]

        if not self.track_initialized:
            if regime.name == "Out":
                return None
            idx = int(np.argmax(filtered.confidence))
            return {"center": centers[idx], "sv": filtered[idx : idx + 1]}

        # Pick highest-confidence detection; outlier check happens in _step.
        idx = int(np.argmax(filtered.confidence))
        return {"center": centers[idx], "sv": filtered[idx : idx + 1]}

    def _step(self, best_detection, frame_count):
        max_gap = self.current_max_gap

        if best_detection is None:
            if not self.track_initialized:
                return None, None
            self.frames_since_detection += 1
            self.hit_streak = 0
            if self.frames_since_detection > max_gap:
                self.track_initialized = False
                self.last_position = None
                self.outlier_confirmer.reset()
                print(f"Frame {frame_count}: track lost - gap exceeded {max_gap} frames")
            return None, None

        center = best_detection["center"]

        if not self.track_initialized:
            self.last_position = center.copy()
            self.track_initialized = True
            self.frames_since_detection = 0
            self.hit_streak = 1
            self.outlier_confirmer.reset()
            print(f"Frame {frame_count}: ACCEPTED (init) at {center.round(1)}")
            return center, best_detection["sv"]

        dist = float(np.linalg.norm(center - self.last_position))
        in_gate = dist < EUCLIDEAN_GATE

        if in_gate:
            self.last_position = center.copy()
            self.frames_since_detection = 0
            self.hit_streak += 1
            self.outlier_confirmer.reset()
            print(f"Frame {frame_count}: ACCEPTED at {center.round(1)} (dist={dist:.1f}px)")
            return center, best_detection["sv"]

        should_reset = self.outlier_confirmer.report(was_outlier=True)
        self.frames_since_detection += 1
        self.hit_streak = 0

        if should_reset:
            self.last_position = center.copy()
            self.track_initialized = True
            self.frames_since_detection = 0
            self.hit_streak = 1
            self.outlier_confirmer.reset()
            print(f"Frame {frame_count}: RESET at {center.round(1)} (dist={dist:.1f}px)")
            return center, best_detection["sv"]

        if self.frames_since_detection > max_gap:
            self.track_initialized = False
            self.last_position = None
            self.outlier_confirmer.reset()
            print(f"Frame {frame_count}: track lost - outlier streak exceeded {max_gap} frames")
            return None, None

        print(
            f"Frame {frame_count}: REJECTED detection (outside gate) at "
            f"{center.round(1)} (dist={dist:.1f}px)"
        )
        return None, None

    def _annotate(self, frame, output_position, accepted_detection, player_bboxes):
        for x1, y1, x2, y2 in player_bboxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 100), 1)

        if accepted_detection is not None:
            frame = self.box_annotator.annotate(scene=frame, detections=accepted_detection)
            frame = self.label_annotator.annotate(
                scene=frame, detections=accepted_detection, labels=["Ball"]
            )
        return frame

    def _build_detection_record(self, frame_idx, accepted_detection):
        if accepted_detection is None or len(accepted_detection.xyxy) == 0:
            return {"frame_idx": int(frame_idx), "x": None, "y": None, "confidence": None}
        center = accepted_detection.get_anchors_coordinates(sv.Position.CENTER)[0]
        return {
            "frame_idx": int(frame_idx),
            "x": float(center[0]),
            "y": float(center[1]),
            "confidence": float(accepted_detection.confidence[0]),
        }

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(
            OUTPUT_PATH, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        frame_count = 0
        detection_records = []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            regime = self._apply_regime(frame_count)

            ball_detections = self.model.predict(frame, confidence=CONFIDENCE)
            ball_detections = ball_detections[ball_detections.class_id == BALL_CLASS_ID]
            player_bboxes = self._get_player_bboxes(frame) if ENABLE_PHASE_1 else np.empty((0, 4), dtype=np.float32)

            best_detection = self._select_best_detection(
                ball_detections, player_bboxes, regime, frame_count
            )
            output_position, accepted = self._step(best_detection, frame_count)

            annotated = self._annotate(
                frame.copy(), output_position, accepted, player_bboxes
            )
            detection_records.append(self._build_detection_record(frame_count, accepted))
            out_writer.write(annotated)
            frame_count += 1

        out_writer.release()
        self.cap.release()

        with open(DETECTION_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(detection_records, f, indent=2)

        rejection_log_path = DETECTION_JSON_PATH.replace(".json", "_phase1_rejections.json")
        with open(rejection_log_path, "w", encoding="utf-8") as f:
            json.dump(self.phase1_rejection_log, f, indent=2)
        print(f"Phase 1 rejection log written to {rejection_log_path}")

        print("\n=== Diagnostic Summary ===")
        print(
            f"Phase 1 active on {self.phase1_active_frames} frames "
            "(airborne regime + player bboxes present)"
        )
        print(f"Phase 1 rejected {self.phase1_rejections} ball detections as inside-player")
        print(f"Detection JSON written to {DETECTION_JSON_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",          default=VIDEO_PATH)
    parser.add_argument("--output-video",   default=OUTPUT_PATH)
    parser.add_argument("--output-json",    default=DETECTION_JSON_PATH)
    parser.add_argument("--actions-json",   default=ACTIONS_JSON_PATH)
    args = parser.parse_args()

    VIDEO_PATH          = args.video
    OUTPUT_PATH         = args.output_video
    DETECTION_JSON_PATH = args.output_json
    ACTIONS_JSON_PATH   = args.actions_json

    VideoProcessor().run()
