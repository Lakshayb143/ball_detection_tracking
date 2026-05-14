"""
Debug WHY events fire/suppress, and sweep L to see how latency affects detection.

Output 1: per-event-start trace at L=30 (what's blocking each event)
Output 2: latency sweep at L in {15, 30, 45, 60} (recall/precision/start-error per L)
"""

import sys
import numpy as np
import pandas as pd

# Adjust import paths to match your event_detection/ directory
from extract_trajectory_features import load_detections, load_actions, extract_features, load_pose_features
from airborne_rule import AirborneRuleConfig, detect_airborne_events_offline, evaluate_rule_at_frame
from evaluate_airborne_detector import load_gt_airborne_events, evaluate_airborne_detector


# ============================================================
# Config
# ============================================================
DETECTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v4_A.json"
ACTIONS_PATH    = "/home/lakshay/lx/ball_detection_tracking/clip1_actions.json"
POSE_JSON_PATH  = None   # set to your pose JSON path when ready to include pose
FPS = 30.0

# Event-start frames you want to trace
DEBUG_FRAMES = {15, 85, 128, 180, 224}

# Latencies to sweep (in frames)
L_VALUES = [15, 30, 45, 60]


# ============================================================
# Debug version of the offline detector — same logic, with prints
# ============================================================
def detect_with_debug(features_df, config, debug_frames):
    feature_dicts = features_df.to_dict("records")
    events = []
    active_event_until = -1
    cooldown_until = -1

    for i, row in enumerate(feature_dicts):
        t = int(row["frame"])

        in_active = t <= active_event_until
        in_cooldown = t <= cooldown_until

        if t in debug_frames:
            print(f"\n[DEBUG] frame {t}:")
            print(f"  active_event_until={active_event_until}, cooldown_until={cooldown_until}")
            print(f"  in_active={in_active}, in_cooldown={in_cooldown}")
            print(f"  dy_per_frame_last_pair={row.get('dy_per_frame_last_pair')}")
            print(f"  dy_buffer_first_to_last={row.get('dy_buffer_first_to_last')}")
            print(f"  frames_since_last_det={row.get('frames_since_last_det')}")
            print(f"  detection_density (buffer)={row.get('detection_density'):.2f}")

        if in_active or in_cooldown:
            if t in debug_frames:
                print(f"  -> SUPPRESSED (event at {active_event_until} blocks this frame)")
            continue

        # Compute recent density same way as production rule
        lookback_start = max(0, i - config.ground_window_frames + 1)
        recent_rows = feature_dicts[lookback_start:i + 1]
        recent_n_dets = sum(
            1 for r in recent_rows if r.get("frames_since_last_det", 999) == 0
        )
        recent_density = recent_n_dets / max(1, len(recent_rows))

        result = evaluate_rule_at_frame(row, recent_density, config)

        if t in debug_frames:
            print(f"  recent_density (last {config.ground_window_frames} frames)={recent_density:.2f}")
            if result is None:
                print(f"  -> NO FIRE (gate failed or score below threshold {config.fire_threshold})")
            else:
                score, breakdown = result
                print(f"  -> FIRED with score={score:.2f}")
                print(f"     breakdown: {breakdown}")

        if result is None:
            continue

        score, breakdown = result
        event_type = config.default_event_type
        duration = config.event_durations.get(event_type, config.event_duration_default)

        events.append({
            "start_frame": t, "duration_frames": duration,
            "score": score, "breakdown": breakdown,
        })
        active_event_until = t + duration
        cooldown_until = active_event_until + config.post_event_cooldown_frames

    return events


# ============================================================
# Main: debug at L=30, then sweep
# ============================================================
def main():
    print("=" * 70)
    print("PART 1 — DEBUG TRACE AT L=30")
    print("=" * 70)

    detections = load_detections(DETECTIONS_PATH)
    pose_features = load_pose_features(POSE_JSON_PATH) if POSE_JSON_PATH else {}
    df_30 = extract_features(detections, buffer_l=30, pose_features=pose_features)
    config = AirborneRuleConfig()

    print(f"Config: fire_threshold={config.fire_threshold}, "
          f"upward_dy_threshold={config.upward_dy_per_frame_threshold}, "
          f"weight_upward={config.weight_upward_velocity}, "
          f"event_durations={config.event_durations}, "
          f"post_event_cooldown={config.post_event_cooldown_frames}, "
          f"ground_window={config.ground_window_frames}, "
          f"min_density_for_ground={config.min_density_for_ground}")
    print(f"Tracing fires/suppressions at frames: {sorted(DEBUG_FRAMES)}")

    events = detect_with_debug(df_30, config, DEBUG_FRAMES)
    print(f"\n[DEBUG SUMMARY] {len(events)} total events fired:")
    for e in events:
        print(f"  start={e['start_frame']:>4}, duration={e['duration_frames']:>3}, "
              f"score={e['score']:.2f}, breakdown={e['breakdown']}")

    # ============================================================
    print("\n\n" + "=" * 70)
    print("PART 2 — LATENCY SWEEP")
    print("=" * 70)

    gt_events = load_gt_airborne_events(ACTIONS_PATH)
    total_frames = max(detections.keys()) + 1

    print(f"\n  {'L (s)':>6}  {'L (fr)':>7}  {'Recall':>7}  {'Prec':>6}  "
          f"{'Mean err':>9}  {'FP/min':>7}  {'TP':>3}  {'FP':>3}")
    rows = []
    for L in L_VALUES:
        df_L = extract_features(detections, buffer_l=L, pose_features=pose_features)
        preds = detect_airborne_events_offline(df_L, config)
        result = evaluate_airborne_detector(
            predictions=preds, gt_events=gt_events,
            total_frames=total_frames, fps=FPS,
        )
        n_fp = result.n_pred - result.n_matched
        rows.append({
            "L_frames": L, "L_seconds": L / FPS,
            "recall": result.recall, "precision": result.precision,
            "mean_err": result.mean_start_error,
            "fp_per_min": result.fp_per_minute,
            "tp": result.n_matched, "fp": n_fp,
        })
        print(f"  {L/FPS:>6.2f}  {L:>7d}  {result.recall*100:>6.1f}%  "
              f"{result.precision*100:>5.1f}%  {result.mean_start_error:>9.1f}  "
              f"{result.fp_per_minute:>7.2f}  {result.n_matched:>3}  {n_fp:>3}")

        # Show per-GT status for this L
        print(f"    Per-GT:")
        for s in result.per_gt_status:
            if s["detected"]:
                print(f"      [DETECTED] {s['gt_action']:<12} gt={s['gt_start']:>4} "
                      f"pred={s['predicted_start']:>4} err={s['start_error_frames']}")
            else:
                print(f"      [MISSED]   {s['gt_action']:<12} gt={s['gt_start']:>4}")

    pd.DataFrame(rows).to_csv("latency_sweep.csv", index=False)
    print(f"\nSweep CSV written to latency_sweep.csv")


if __name__ == "__main__":
    main()