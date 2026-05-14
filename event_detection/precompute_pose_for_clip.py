"""
Precompute pose features for an entire video and dump to JSON.

Runs the player detector on every frame, then ViTPose++ Huge on each player
crop. Output is a JSON keyed by frame index, with per-player pose features.

Once this is written, rule iteration is fast — load the JSON, no model calls.

Output JSON format:
{
  "metadata": {
    "video_path": "...",
    "n_frames": 481,
    "fps": 30.0,
    "model_name": "usyd-community/vitpose-plus-huge",
    "dataset_index": 5,
    "kicking_extended_angle_deg": 150.0,
    "kicking_planted_angle_deg": 140.0
  },
  "frames": {
    "0": {
      "players": [
        {
          "bbox_xyxy": [x1, y1, x2, y2],
          "class_id": 2,
          "left_leg_extension_angle": 165.3,
          "right_leg_extension_angle": 95.1,
          "left_ankle_xy": [..., ...],
          "right_ankle_xy": [..., ...],
          "is_kicking_pose": true,
          "kicking_leg": "left",
          "confidence": 0.81
        },
        ...
      ]
    },
    "1": {...},
    ...
  }
}
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium

from pose_features import PoseFeatureExtractor, PlayerPoseFeatures


# ============================================================
# Config — edit these
# ============================================================
VIDEO_PATH = "/home/lakshay/lx/ball_detection_tracking/france_vs_argentina/clip1.mp4"
PLAYER_MODEL_PATH = "/home/lakshay/lx/ball_detection_tracking/checkpoints/player.pth"
OUTPUT_JSON = "/home/lakshay/lx/ball_detection_tracking/clip1_pose_features.json"

PLAYER_CONFIDENCE = 0.5
# class IDs we want to run pose on. From your map: {1: gk, 2: player, 3: ref}.
# Excluding ref (3) since refs don't kick the ball; including gk (1) is fine
# since GKs do kick (goal kicks, distribution).
POSE_TARGET_CLASS_IDS = (1, 2)

# Pad player bboxes outward before pose so feet aren't cut off
BBOX_PADDING_FRACTION = 0.10

# Save progress every N frames so you don't lose everything if it crashes
CHECKPOINT_EVERY = 50

# Optional: skip frames that are far from any hand-labeled action event,
# to save time during initial development. Set to None to process all frames.
ACTIONS_JSON_FOR_GATING = None   # e.g. "/path/to/clip1_actions.json"
GATE_WINDOW_FRAMES = 90          # only process frames within this many of any event


# ============================================================
# Helpers
# ============================================================
def pad_bboxes(bboxes_xyxy: np.ndarray, frame_w: int, frame_h: int,
               pad_fraction: float) -> np.ndarray:
    """Expand bboxes outward by a fraction of their dimensions, clipped to frame."""
    if len(bboxes_xyxy) == 0:
        return bboxes_xyxy
    bboxes = np.asarray(bboxes_xyxy, dtype=np.float32).copy()
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    pad_x = widths * pad_fraction
    pad_y = heights * pad_fraction
    bboxes[:, 0] = np.clip(bboxes[:, 0] - pad_x, 0, frame_w)
    bboxes[:, 1] = np.clip(bboxes[:, 1] - pad_y, 0, frame_h)
    bboxes[:, 2] = np.clip(bboxes[:, 2] + pad_x, 0, frame_w)
    bboxes[:, 3] = np.clip(bboxes[:, 3] + pad_y, 0, frame_h)
    return bboxes


def load_action_gate(actions_path: str, window_frames: int) -> set:
    """Returns the set of frame indices to process (within window of any event)."""
    with open(actions_path) as f:
        events = json.load(f).get("events", [])
    keep = set()
    for evt in events:
        f = evt["frame"]
        for k in range(max(0, f - window_frames), f + window_frames + 1):
            keep.add(k)
    return keep


def features_to_dict(feature: PlayerPoseFeatures, class_id: int) -> dict:
    return {
        "bbox_xyxy": feature.bbox_xyxy.tolist(),
        "class_id": int(class_id),
        "left_leg_extension_angle": feature.left_leg_extension_angle,
        "right_leg_extension_angle": feature.right_leg_extension_angle,
        "left_ankle_xy": list(feature.left_ankle_xy) if feature.left_ankle_xy else None,
        "right_ankle_xy": list(feature.right_ankle_xy) if feature.right_ankle_xy else None,
        "is_kicking_pose": bool(feature.is_kicking_pose),
        "kicking_leg": feature.kicking_leg,
        "confidence": float(feature.confidence),
    }


# ============================================================
# Main
# ============================================================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Could not open {VIDEO_PATH}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[precompute] Video: {VIDEO_PATH}")
    print(f"[precompute] {n_frames} frames @ {fps:.2f} fps, {frame_w}x{frame_h}")

    print(f"[precompute] Loading player model from {PLAYER_MODEL_PATH}")
    player_model = RFDETRMedium(pretrain_weights=PLAYER_MODEL_PATH)
    try:
        player_model.optimize_for_inference()
    except Exception as e:
        print(f"[precompute] could not optimize player model: {e}")

    print(f"[precompute] Loading pose model")
    pose_extractor = PoseFeatureExtractor()

    # Frame gating
    frame_gate = None
    if ACTIONS_JSON_FOR_GATING is not None:
        frame_gate = load_action_gate(ACTIONS_JSON_FOR_GATING, GATE_WINDOW_FRAMES)
        print(f"[precompute] Gating: {len(frame_gate)} of {n_frames} frames "
              f"within {GATE_WINDOW_FRAMES} frames of an event")

    # Output container
    output = {
        "metadata": {
            "video_path": str(VIDEO_PATH),
            "n_frames": n_frames,
            "fps": float(fps),
            "frame_width": frame_w,
            "frame_height": frame_h,
            "model_name": "usyd-community/vitpose-plus-huge",
            "dataset_index": 5,
            "kicking_extended_angle_deg": pose_extractor.extended_threshold,
            "kicking_planted_angle_deg": pose_extractor.planted_threshold,
            "keypoint_conf_min": pose_extractor.keypoint_conf_min,
            "bbox_padding_fraction": BBOX_PADDING_FRACTION,
            "pose_target_class_ids": list(POSE_TARGET_CLASS_IDS),
        },
        "frames": {},
    }

    # Diagnostic counters
    n_processed = 0
    n_skipped_gated = 0
    n_skipped_no_players = 0
    n_kicking_total = 0
    t_start = time.time()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_gate is not None and frame_idx not in frame_gate:
            n_skipped_gated += 1
            output["frames"][str(frame_idx)] = {"players": [], "skipped": "gated_out"}
            frame_idx += 1
            continue

        # 1. Detect players
        player_dets = player_model.predict(frame, confidence=PLAYER_CONFIDENCE)
        if len(player_dets) == 0:
            n_skipped_no_players += 1
            output["frames"][str(frame_idx)] = {"players": []}
            frame_idx += 1
            continue

        keep_mask = np.isin(player_dets.class_id, POSE_TARGET_CLASS_IDS)
        if not np.any(keep_mask):
            n_skipped_no_players += 1
            output["frames"][str(frame_idx)] = {"players": []}
            frame_idx += 1
            continue

        kept_bboxes = player_dets.xyxy[keep_mask].astype(np.float32)
        kept_class_ids = player_dets.class_id[keep_mask]
        padded_bboxes = pad_bboxes(kept_bboxes, frame_w, frame_h, BBOX_PADDING_FRACTION)

        # 2. Run pose
        try:
            pose_features = pose_extractor.extract(frame, padded_bboxes)
        except Exception as e:
            print(f"[precompute] pose failed on frame {frame_idx}: {e}")
            output["frames"][str(frame_idx)] = {"players": [], "error": str(e)}
            frame_idx += 1
            continue

        # 3. Pack
        players_out = []
        for feat, cls_id in zip(pose_features, kept_class_ids):
            players_out.append(features_to_dict(feat, cls_id))
            if feat.is_kicking_pose:
                n_kicking_total += 1
        output["frames"][str(frame_idx)] = {"players": players_out}

        n_processed += 1
        frame_idx += 1

        # Progress + periodic checkpoint
        if frame_idx % CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t_start
            rate = n_processed / elapsed if elapsed > 0 else 0
            remaining = (n_frames - frame_idx) / max(rate, 1e-6)
            print(f"[precompute] frame {frame_idx}/{n_frames}  "
                  f"processed={n_processed}  kicking_so_far={n_kicking_total}  "
                  f"rate={rate:.2f} fps  est_remaining={remaining/60:.1f} min")
            with open(OUTPUT_JSON, "w") as f:
                json.dump(output, f)

    cap.release()

    # Final write
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n[precompute] DONE")
    print(f"  Total frames:           {n_frames}")
    print(f"  Processed:              {n_processed}")
    print(f"  Skipped (gated out):    {n_skipped_gated}")
    print(f"  Skipped (no players):   {n_skipped_no_players}")
    print(f"  Kicking poses detected: {n_kicking_total}")
    print(f"  Wall time:              {elapsed/60:.2f} min")
    print(f"  Effective rate:         {n_processed/elapsed:.2f} fps")
    print(f"  Output:                 {OUTPUT_JSON}")


if __name__ == "__main__":
    main()