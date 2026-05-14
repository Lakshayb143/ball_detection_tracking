"""
Pose features for airborne event detection.

Given a frame and player bboxes, runs ViTPose++ Huge on each player crop and
extracts kicking-pose features per player.

Output features (per player, per frame):
  - left_leg_extension_angle  : hip-knee-ankle angle on left leg (degrees, 180 = fully extended)
  - right_leg_extension_angle : same for right leg
  - left_ankle_y              : pixel y of left ankle (None if not detected)
  - right_ankle_y             : pixel y of right ankle (None if not detected)
  - is_kicking_pose           : bool — heuristic: one leg extended past threshold,
                                 other leg planted, ankle of extended leg above hip
  - kicking_leg               : "left", "right", or None
  - confidence                : min keypoint confidence across the leg keypoints used

Usage:
    extractor = PoseFeatureExtractor()
    pose_per_player = extractor.extract(frame_bgr, player_bboxes_xyxy)
    # pose_per_player is a list of dicts, one per player bbox

The pose model is loaded once on construction and reused. Loading takes a few
seconds; per-frame inference depends on player count (most cost is the ViT forward
pass on each crop).
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Optional, Sequence, Tuple
from dataclasses import dataclass


# ============================================================
# Config
# ============================================================
DEFAULT_MODEL_NAME = "usyd-community/vitpose-plus-huge"
DEFAULT_DATASET_INDEX = 5    # 5 = COCO-WholeBody (includes feet keypoints)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kicking detection thresholds — tunable. Defaults reasoned from soccer biomechanics:
# - Extended leg: hip-knee-ankle angle near 180 (fully extended). Threshold 150° is generous.
# - Planted leg: angle clearly bent. Threshold 140° max.
# - Ankle of kicking leg should be above the hip (lower y in image space) at follow-through.
KICKING_EXTENDED_ANGLE_DEG = 150.0
KICKING_PLANTED_ANGLE_DEG = 140.0
KICKING_KEYPOINT_CONF_MIN = 0.3
KICKING_ANKLE_ABOVE_HIP_REQUIRED = False   # keep False; covers anticipation phase too


# ============================================================
# Keypoint indices
# ============================================================
# COCO-WholeBody (dataset_index=5) keypoint order matches COCO body keypoints
# for the first 17, plus feet keypoints at indices 17-22. We only need the body
# joints for kicking detection.
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


# ============================================================
# Output container
# ============================================================
@dataclass
class PlayerPoseFeatures:
    bbox_xyxy: np.ndarray
    left_leg_extension_angle: Optional[float]
    right_leg_extension_angle: Optional[float]
    left_ankle_xy: Optional[Tuple[float, float]]
    right_ankle_xy: Optional[Tuple[float, float]]
    is_kicking_pose: bool
    kicking_leg: Optional[str]
    confidence: float
    keypoints_xy: np.ndarray              # full (K, 2) keypoint array in image space
    keypoint_scores: np.ndarray           # (K,) confidence per keypoint


# ============================================================
# Geometry helpers
# ============================================================
def angle_between_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Returns the angle at vertex b formed by segments b->a and b->c, in degrees.
    Returns NaN if any vector has zero length.
    """
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos_angle = float(np.dot(v1, v2) / (n1 * n2))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def leg_angle_and_conf(keypoints: np.ndarray, scores: np.ndarray,
                       hip_idx: int, knee_idx: int, ankle_idx: int,
                       min_conf: float) -> Tuple[Optional[float], float]:
    """
    Returns (angle_degrees, min_confidence_across_3_keypoints) or (None, 0.0)
    if any of the three keypoints fails the confidence threshold.
    """
    confs = [scores[hip_idx], scores[knee_idx], scores[ankle_idx]]
    min_c = float(min(confs))
    if min_c < min_conf:
        return None, min_c
    angle = angle_between_three_points(
        keypoints[hip_idx], keypoints[knee_idx], keypoints[ankle_idx]
    )
    if np.isnan(angle):
        return None, min_c
    return angle, min_c


def is_kicking_from_legs(
    left_angle: Optional[float],
    right_angle: Optional[float],
    left_ankle_y: Optional[float],
    right_ankle_y: Optional[float],
    left_hip_y: Optional[float],
    right_hip_y: Optional[float],
    extended_threshold: float,
    planted_threshold: float,
    require_ankle_above_hip: bool,
) -> Tuple[bool, Optional[str]]:
    """
    Heuristic: a kick is when one leg is clearly extended and the other is clearly
    planted (bent, supporting weight). Optionally require the kicking ankle to be
    above the hip (deep in follow-through).
    """
    candidates = []
    if (left_angle is not None and right_angle is not None
            and left_angle >= extended_threshold and right_angle <= planted_threshold):
        if not require_ankle_above_hip or (
            left_ankle_y is not None and left_hip_y is not None
            and left_ankle_y < left_hip_y
        ):
            candidates.append(("left", left_angle))

    if (left_angle is not None and right_angle is not None
            and right_angle >= extended_threshold and left_angle <= planted_threshold):
        if not require_ankle_above_hip or (
            right_ankle_y is not None and right_hip_y is not None
            and right_ankle_y < right_hip_y
        ):
            candidates.append(("right", right_angle))

    if not candidates:
        return False, None
    candidates.sort(key=lambda t: -t[1])
    return True, candidates[0][0]


# ============================================================
# Pose feature extractor
# ============================================================
class PoseFeatureExtractor:
    """
    Wraps ViTPose++ for inference on player crops.

    The model is loaded once on construction. Call .extract(frame, bboxes) per
    frame; returns a list of PlayerPoseFeatures, one entry per input bbox.
    Bboxes that produce zero confident keypoints still get an entry, with all
    fields set to None / False.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dataset_index: int = DEFAULT_DATASET_INDEX,
        device: str = DEFAULT_DEVICE,
        extended_threshold: float = KICKING_EXTENDED_ANGLE_DEG,
        planted_threshold: float = KICKING_PLANTED_ANGLE_DEG,
        keypoint_conf_min: float = KICKING_KEYPOINT_CONF_MIN,
        require_ankle_above_hip: bool = KICKING_ANKLE_ABOVE_HIP_REQUIRED,
    ):
        # Lazy import so the module imports even if transformers isn't present
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self.device = torch.device(device)
        self.dataset_index = int(dataset_index)
        self.extended_threshold = float(extended_threshold)
        self.planted_threshold = float(planted_threshold)
        self.keypoint_conf_min = float(keypoint_conf_min)
        self.require_ankle_above_hip = bool(require_ankle_above_hip)

        print(f"[pose] Loading {model_name} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"[pose] Loaded.")

    @torch.inference_mode()
    def extract(
        self,
        frame_bgr: np.ndarray,
        player_bboxes_xyxy: np.ndarray,
    ) -> List[PlayerPoseFeatures]:
        """
        Run pose estimation on every player bbox in the given frame.
        frame_bgr is the raw OpenCV BGR frame; bboxes are xyxy in image coords.
        """
        if len(player_bboxes_xyxy) == 0:
            return []

        # Convert BGR -> RGB and to PIL for the processor
        from PIL import Image
        rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(rgb)

        # The HF processor expects bboxes in xywh per image
        bboxes_xywh = self._xyxy_to_xywh(player_bboxes_xyxy)
        inputs = self.processor(
            image, boxes=[bboxes_xywh.tolist()], return_tensors="pt"
        ).to(self.device)

        # ViTPose++ expects a dataset index per bbox to select the MoE expert
        dataset_index = torch.full(
            (len(bboxes_xywh),), self.dataset_index, dtype=torch.int64, device=self.device
        )
        outputs = self.model(**inputs, dataset_index=dataset_index)

        results = self.processor.post_process_pose_estimation(
            outputs, boxes=[bboxes_xywh.tolist()], threshold=0.0
        )[0]   # one image -> one list of per-bbox results

        per_player_features = []
        for bbox, result in zip(player_bboxes_xyxy, results):
            keypoints = result["keypoints"].cpu().numpy() if hasattr(result["keypoints"], "cpu") else np.asarray(result["keypoints"])
            scores = result["scores"].cpu().numpy() if hasattr(result["scores"], "cpu") else np.asarray(result["scores"])
            features = self._build_features(bbox, keypoints, scores)
            per_player_features.append(features)

        return per_player_features

    def _xyxy_to_xywh(self, bboxes_xyxy: np.ndarray) -> np.ndarray:
        bboxes = np.asarray(bboxes_xyxy, dtype=np.float32)
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        return np.column_stack([bboxes[:, 0], bboxes[:, 1], widths, heights])

    def _build_features(
        self,
        bbox_xyxy: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> PlayerPoseFeatures:
        # Compute leg angles
        left_angle, left_conf = leg_angle_and_conf(
            keypoints, scores, KP_LEFT_HIP, KP_LEFT_KNEE, KP_LEFT_ANKLE,
            self.keypoint_conf_min,
        )
        right_angle, right_conf = leg_angle_and_conf(
            keypoints, scores, KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE,
            self.keypoint_conf_min,
        )

        # Ankle and hip positions
        left_ankle_xy = (
            (float(keypoints[KP_LEFT_ANKLE, 0]), float(keypoints[KP_LEFT_ANKLE, 1]))
            if scores[KP_LEFT_ANKLE] >= self.keypoint_conf_min else None
        )
        right_ankle_xy = (
            (float(keypoints[KP_RIGHT_ANKLE, 0]), float(keypoints[KP_RIGHT_ANKLE, 1]))
            if scores[KP_RIGHT_ANKLE] >= self.keypoint_conf_min else None
        )
        left_hip_y = (
            float(keypoints[KP_LEFT_HIP, 1])
            if scores[KP_LEFT_HIP] >= self.keypoint_conf_min else None
        )
        right_hip_y = (
            float(keypoints[KP_RIGHT_HIP, 1])
            if scores[KP_RIGHT_HIP] >= self.keypoint_conf_min else None
        )

        # Kicking heuristic
        is_kicking, kicking_leg = is_kicking_from_legs(
            left_angle=left_angle,
            right_angle=right_angle,
            left_ankle_y=left_ankle_xy[1] if left_ankle_xy else None,
            right_ankle_y=right_ankle_xy[1] if right_ankle_xy else None,
            left_hip_y=left_hip_y,
            right_hip_y=right_hip_y,
            extended_threshold=self.extended_threshold,
            planted_threshold=self.planted_threshold,
            require_ankle_above_hip=self.require_ankle_above_hip,
        )

        # Aggregate confidence — min across the 6 leg keypoints we care about
        leg_confs = [
            scores[KP_LEFT_HIP], scores[KP_RIGHT_HIP],
            scores[KP_LEFT_KNEE], scores[KP_RIGHT_KNEE],
            scores[KP_LEFT_ANKLE], scores[KP_RIGHT_ANKLE],
        ]
        leg_confidence = float(min(leg_confs))

        return PlayerPoseFeatures(
            bbox_xyxy=np.asarray(bbox_xyxy, dtype=np.float32),
            left_leg_extension_angle=left_angle,
            right_leg_extension_angle=right_angle,
            left_ankle_xy=left_ankle_xy,
            right_ankle_xy=right_ankle_xy,
            is_kicking_pose=is_kicking,
            kicking_leg=kicking_leg,
            confidence=leg_confidence,
            keypoints_xy=keypoints,
            keypoint_scores=scores,
        )


# ============================================================
# Convenience: scan a frame for "any kicking player near point P"
# ============================================================
def any_kicking_player_near_point(
    pose_features: Sequence[PlayerPoseFeatures],
    target_xy: Tuple[float, float],
    distance_threshold_px: float,
) -> Optional[PlayerPoseFeatures]:
    """
    Returns the kicking player whose extended-leg ankle is closest to target_xy
    and within the threshold, or None.

    target_xy is typically the last known ball position. This is the predicate
    you'd plug into the airborne rule: 'is some player kicking, near the ball?'
    """
    best = None
    best_dist = float("inf")
    tx, ty = target_xy

    for p in pose_features:
        if not p.is_kicking_pose:
            continue
        if p.kicking_leg == "left" and p.left_ankle_xy is not None:
            ax, ay = p.left_ankle_xy
        elif p.kicking_leg == "right" and p.right_ankle_xy is not None:
            ax, ay = p.right_ankle_xy
        else:
            continue
        dist = float(np.hypot(ax - tx, ay - ty))
        if dist < best_dist and dist <= distance_threshold_px:
            best = p
            best_dist = dist
    return best


# ============================================================
# Quick test harness
# ============================================================
if __name__ == "__main__":
    import cv2
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pose_features.py <image_path> [<bbox_xyxy_csv>]")
        print("  bbox_xyxy_csv: comma-separated 'x1,y1,x2,y2;x1,y1,x2,y2;...'")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read {sys.argv[1]}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        bboxes = []
        for chunk in sys.argv[2].split(";"):
            parts = [float(x) for x in chunk.split(",")]
            bboxes.append(parts)
        bboxes = np.array(bboxes, dtype=np.float32)
    else:
        # Default: full image as one "player"
        h, w = img.shape[:2]
        bboxes = np.array([[0, 0, w, h]], dtype=np.float32)

    extractor = PoseFeatureExtractor()
    features = extractor.extract(img, bboxes)
    for i, f in enumerate(features):
        print(f"\n--- Player {i} (bbox {f.bbox_xyxy}) ---")
        print(f"  Left leg angle:  {f.left_leg_extension_angle}")
        print(f"  Right leg angle: {f.right_leg_extension_angle}")
        print(f"  Left ankle:      {f.left_ankle_xy}")
        print(f"  Right ankle:     {f.right_ankle_xy}")
        print(f"  Kicking pose:    {f.is_kicking_pose} (leg: {f.kicking_leg})")
        print(f"  Confidence:      {f.confidence:.3f}")