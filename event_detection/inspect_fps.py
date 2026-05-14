"""
Investigate each false-positive prediction:
  - what was happening in the trajectory features at that moment
  - was there a labeled event nearby that this might really be related to
  - what was the score breakdown
"""

import pandas as pd
from extract_trajectory_features import load_detections, load_actions, extract_features
from airborne_rule import AirborneRuleConfig, detect_airborne_events_offline, evaluate_rule_at_frame

DETECTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v4_A.json"
ACTIONS_PATH    = "/home/lakshay/lx/ball_detection_tracking/clip1_actions.json"
FP_FRAMES       = [52, 296, 331, 424]   # from the latest eval

detections = load_detections(DETECTIONS_PATH)
df = extract_features(detections, buffer_l=30)
config = AirborneRuleConfig()
events = load_actions(ACTIONS_PATH)
print(f"GT events: {[(e['frame'], e['action']) for e in events]}\n")

for fp_frame in FP_FRAMES:
    print("=" * 60)
    print(f"FP at frame {fp_frame}")
    print("=" * 60)

    # Show 5 frames before and 10 after
    window = df[(df["frame"] >= fp_frame - 5) & (df["frame"] <= fp_frame + 10)].copy()
    print(f"\nFeatures around frame {fp_frame}:")
    print(f"  {'frame':>5} {'dy_pf':>8} {'dy_buf':>8} {'gap':>4} {'last_y':>7} {'density':>7}")
    for _, r in window.iterrows():
        marker = " <- FP" if int(r['frame']) == fp_frame else ""
        dy_pf = f"{r['dy_per_frame_last_pair']:.2f}" if not pd.isna(r['dy_per_frame_last_pair']) else "n/a"
        dy_buf = f"{r['dy_buffer_first_to_last']:.1f}" if not pd.isna(r['dy_buffer_first_to_last']) else "n/a"
        last_y = f"{r['last_y']:.0f}" if not pd.isna(r['last_y']) else "n/a"
        print(f"  {int(r['frame']):>5} {dy_pf:>8} {dy_buf:>8} {int(r['frames_since_last_det']):>4} {last_y:>7} {r['detection_density']:>7.2f}{marker}")

    # Score breakdown at the fire frame
    fire_row = df[df["frame"] == fp_frame].iloc[0].to_dict()
    i = df.index[df["frame"] == fp_frame][0]
    lookback_start = max(0, i - config.ground_window_frames + 1)
    recent_n = sum(1 for r in df.iloc[lookback_start:i+1].to_dict("records")
                   if r.get("frames_since_last_det", 999) == 0)
    recent_density = recent_n / max(1, (i - lookback_start + 1))
    result = evaluate_rule_at_frame(fire_row, recent_density, config)
    if result:
        print(f"\nScore at fire: {result[0]:.2f}, breakdown={result[1]}")

    # Distance to nearest GT event
    nearest = min(events, key=lambda e: abs(e['frame'] - fp_frame))
    dist = nearest['frame'] - fp_frame
    print(f"\nNearest GT event: {nearest['action']} at frame {nearest['frame']} ({'after' if dist > 0 else 'before'} by {abs(dist)} frames)")
    print()