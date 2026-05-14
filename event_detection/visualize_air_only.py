"""
Simple airborne visualization.

Reads the trajectory_features.csv, runs the state machine, and overlays a
top-right "AIRBORNE EVENT" label on the video during airborne windows.

That's it. No ball rendering. No tracker. No detector calls. Just airborne
yes/no shown clearly on each frame.
"""

import cv2
import pandas as pd

from airborne_rule import AirborneRuleConfig
from airborne_state_machine import (
    AirborneStateMachine,
    StateMachineConfig,
)


# ============================================================
# Config
# ============================================================
VIDEO_PATH = "/home/lakshay/lx/ball_detection_tracking/france_vs_argentina/clip2.mp4"
FEATURES_CSV = "trajectory_features_c2.csv"
OUTPUT_PATH = "/home/lakshay/lx/ball_detection_tracking/clip2_airborne_overlay.mp4"


# ============================================================
# Main
# ============================================================
def main():
    # Load features and build per-frame airborne lookup using the state machine
    df = pd.read_csv(FEATURES_CSV)
    sm = AirborneStateMachine(AirborneRuleConfig(), StateMachineConfig())

    is_airborne_per_frame = {}
    for row in df.to_dict("records"):
        sm.step(row)
        is_airborne_per_frame[int(row["frame"])] = (sm.state == AirborneStateMachine.AIRBORNE)

    print(f"Airborne in {sum(is_airborne_per_frame.values())} of {len(is_airborne_per_frame)} frames")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Could not open {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    # Top-right label box
    label = "AIRBORNE EVENT"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    pad = 12
    box_w, box_h = text_w + 2 * pad, text_h + 2 * pad
    box_x1 = w - box_w - 20
    box_y1 = 20
    box_x2 = box_x1 + box_w
    box_y2 = box_y1 + box_h

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_airborne_per_frame.get(frame_idx, False):
            # Solid colored box, white text
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 165, 255), -1)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)
            cv2.putText(frame, label,
                        (box_x1 + pad, box_y1 + pad + text_h),
                        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    out.release()
    cap.release()
    print(f"Output written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()