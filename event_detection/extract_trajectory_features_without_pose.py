"""
Trajectory feature extractor for airborne event detection.

For each frame t, computes features over a rolling buffer of L frames:
  [t-L+1, ..., t]. The features are designed to separate "ball is on
  ground / in possession" from "ball was just launched into the air."

Inputs:
  - Detection JSON (from your v4 tracker — must contain frame_idx, x, y, confidence)
  - Actions JSON (hand-annotated airborne events — used only for plot overlay)

Outputs:
  - trajectory_features.csv: per-frame features
  - trajectory_features.png: features plotted over time with airborne windows shaded
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# Config
# ============================================================
# DETECTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v4_A.json"
DETECTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip2.json"
ACTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip2_actions.json"
PLAYER_DETECTIONS_PATH = None  # Set to None for now; we'll add this later if useful

OUTPUT_CSV = "trajectory_features_c2.csv"
OUTPUT_PLOT = "trajectory_features_c2.png"

# Buffer length — your latency budget in frames.
# 15 = 0.5s at 30fps, 30 = 1s at 30fps, 45 = 1.5s.
BUFFER_L = 30

# How long each labeled action is "active" for plot shading
ACTION_DURATIONS = {
    "High Pass": 60, "Cross": 50, "Shot": 40, "Free Kick": 60,
    "Header": 20, "Throw In": 40,
    "Pass": 30, "Drive": 30,
    "Ball Player Block": 15, "Player Successful Tackle": 15,
}
AIRBORNE_ACTIONS = {"High Pass", "Cross", "Shot", "Free Kick", "Header"}


# ============================================================
# Loaders
# ============================================================
def load_detections(path):
    """Returns dict: frame_idx -> (x, y, conf) or None if no detection."""
    with open(path) as f:
        data = json.load(f)
    det = {}
    for entry in data:
        f = entry["frame_idx"]
        x, y = entry.get("x"), entry.get("y")
        conf = entry.get("confidence")
        if x is not None and y is not None:
            det[f] = (float(x), float(y), float(conf) if conf is not None else 0.0)
        else:
            det[f] = None
    return det


def load_actions(path):
    with open(path) as f:
        data = json.load(f)
    return sorted(data.get("events", []), key=lambda e: e["frame"])


# ============================================================
# Feature extraction
# ============================================================
def extract_features(detections, buffer_l):
    """
    For each frame t, compute features over the buffer [t-L+1, ..., t].

    Features:
      n_detections_in_buffer  : int      — how many frames in buffer have a detection
      median_y_in_buffer      : float    — median y of detections in buffer (or NaN)
      y_range_in_buffer       : float    — max y - min y in buffer (or NaN)
      frames_since_last_det   : int      — gap since most recent detection within buffer
      last_y                  : float    — y of most recent detection in buffer
      dy_last_pair            : float    — y difference between last two detections
      dy_per_frame_last_pair  : float    — dy_last_pair / frame gap between them
      dy_buffer_first_to_last : float    — y of last detection minus y of first detection
                                            in the buffer (overall trend over L frames)
      detection_density       : float    — n_detections / L (fraction of frames with ball)
    """
    if not detections:
        return pd.DataFrame()

    max_frame = max(detections.keys())
    rows = []

    for t in range(max_frame + 1):
        buffer_start = max(0, t - buffer_l + 1)
        buffer_frames = list(range(buffer_start, t + 1))
        # Pull (frame, x, y, conf) for frames with valid detection
        buffer_dets = [
            (f, *detections[f])
            for f in buffer_frames
            if f in detections and detections[f] is not None
        ]

        n_dets = len(buffer_dets)
        ys = np.array([d[2] for d in buffer_dets]) if buffer_dets else np.array([])

        if n_dets > 0:
            median_y = float(np.median(ys))
            y_range = float(ys.max() - ys.min())
            last_frame, last_x, last_y, last_conf = buffer_dets[-1]
            frames_since_last = t - last_frame
        else:
            median_y = np.nan
            y_range = np.nan
            last_y = np.nan
            frames_since_last = buffer_l

        if n_dets >= 2:
            f1, x1, y1, c1 = buffer_dets[-2]
            f2, x2, y2, c2 = buffer_dets[-1]
            dy_last_pair = float(y2 - y1)
            dy_per_frame_last_pair = float(dy_last_pair / max(1, f2 - f1))
            dy_buffer_first_to_last = float(buffer_dets[-1][2] - buffer_dets[0][2])
        else:
            dy_last_pair = np.nan
            dy_per_frame_last_pair = np.nan
            dy_buffer_first_to_last = np.nan

        rows.append({
            "frame": t,
            "n_detections_in_buffer": n_dets,
            "detection_density": n_dets / buffer_l,
            "median_y_in_buffer": median_y,
            "y_range_in_buffer": y_range,
            "frames_since_last_det": frames_since_last,
            "last_y": last_y,
            "dy_last_pair": dy_last_pair,
            "dy_per_frame_last_pair": dy_per_frame_last_pair,
            "dy_buffer_first_to_last": dy_buffer_first_to_last,
        })

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================
def plot_features(df, events, buffer_l, output_path):
    """
    Plots features over time with airborne windows shaded.
    Negative dy means ball moved up in image (image-y is down-positive).
    """
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

    # Build airborne window list for shading
    airborne_windows = []
    other_event_windows = []
    for evt in events:
        action = evt["action"]
        start = evt["frame"]
        duration = ACTION_DURATIONS.get(action, 30)
        end = start + duration
        if action in AIRBORNE_ACTIONS:
            airborne_windows.append((start, end, action))
        else:
            other_event_windows.append((start, end, action))

    def shade_axis(ax):
        for start, end, action in airborne_windows:
            ax.axvspan(start, end, alpha=0.25, color="red", zorder=0)
        for start, end, action in other_event_windows:
            ax.axvspan(start, end, alpha=0.10, color="blue", zorder=0)

    # ---- Panel 1: detection density and gap ----
    ax = axes[0]
    ax.plot(df["frame"], df["detection_density"], label="detection density", color="black", lw=1)
    ax.set_ylabel("density (0-1)")
    ax.set_title(f"Detection density and gap (buffer L = {buffer_l} frames)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left")
    shade_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(df["frame"], df["frames_since_last_det"], label="frames since last det",
             color="orange", lw=1, alpha=0.6)
    ax2.set_ylabel("frames since last det", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.legend(loc="upper right")

    # ---- Panel 2: last_y and median_y ----
    ax = axes[1]
    ax.plot(df["frame"], df["last_y"], label="last y", color="navy", lw=1)
    ax.plot(df["frame"], df["median_y_in_buffer"], label="median y in buffer",
            color="teal", lw=1, alpha=0.7)
    ax.set_ylabel("y (px) — lower = higher in image")
    ax.invert_yaxis()
    ax.set_title("Ball y-coordinate (image space)")
    ax.legend(loc="upper left")
    shade_axis(ax)

    # ---- Panel 3: y range over buffer ----
    ax = axes[2]
    ax.plot(df["frame"], df["y_range_in_buffer"], color="purple", lw=1)
    ax.set_ylabel("y range (px)")
    ax.set_title("Spread of ball y over buffer (large = motion in y; small = static)")
    shade_axis(ax)

    # ---- Panel 4: dy_per_frame between last two detections ----
    ax = axes[3]
    ax.plot(df["frame"], df["dy_per_frame_last_pair"], color="darkgreen", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("dy / frame (px)")
    ax.set_title("Most recent y-velocity (negative = ball moving up in image = airborne signal)")
    shade_axis(ax)

    # ---- Panel 5: cumulative buffer trend ----
    ax = axes[4]
    ax.plot(df["frame"], df["dy_buffer_first_to_last"], color="firebrick", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("y_last - y_first (px)")
    ax.set_title("Buffer-level y change (strong negative = ball climbed during the L frames)")
    ax.set_xlabel("frame")
    shade_axis(ax)

    # Legend for shading
    legend_handles = [
        Patch(facecolor="red", alpha=0.25, label="airborne event (hand-labeled)"),
        Patch(facecolor="blue", alpha=0.10, label="ground event (hand-labeled)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=120)
    print(f"Plot saved to {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    print(f"Loading detections from {DETECTIONS_PATH}")
    detections = load_detections(DETECTIONS_PATH)
    print(f"  {len(detections)} frames, "
          f"{sum(1 for v in detections.values() if v is not None)} with detections")

    print(f"\nLoading actions from {ACTIONS_PATH}")
    events = load_actions(ACTIONS_PATH)
    print(f"  {len(events)} events")
    for evt in events:
        airborne = "airborne" if evt["action"] in AIRBORNE_ACTIONS else "ground"
        print(f"  frame {evt['frame']:>4}, {evt['action']:<25} ({airborne})")

    print(f"\nExtracting features with buffer L = {BUFFER_L}")
    df = extract_features(detections, BUFFER_L)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  features written to {OUTPUT_CSV}")

    print(f"\nPlotting...")
    plot_features(df, events, BUFFER_L, OUTPUT_PLOT)

    # Summary stats: feature distributions inside vs outside airborne windows
    print("\n" + "=" * 70)
    print("FEATURE DISTRIBUTIONS: inside vs outside airborne windows")
    print("=" * 70)
    is_airborne = np.zeros(len(df), dtype=bool)
    for evt in events:
        if evt["action"] in AIRBORNE_ACTIONS:
            start = evt["frame"]
            end = start + ACTION_DURATIONS.get(evt["action"], 30)
            mask = (df["frame"] >= start) & (df["frame"] < end)
            is_airborne |= mask.values

    feature_cols = [
        "detection_density",
        "frames_since_last_det",
        "last_y",
        "y_range_in_buffer",
        "dy_per_frame_last_pair",
        "dy_buffer_first_to_last",
    ]

    print(f"\n  {'feature':<28} {'airborne (median)':>18} {'ground (median)':>18}")
    for col in feature_cols:
        airborne_vals = df.loc[is_airborne, col].dropna()
        ground_vals = df.loc[~is_airborne, col].dropna()
        a_med = airborne_vals.median() if len(airborne_vals) > 0 else float("nan")
        g_med = ground_vals.median() if len(ground_vals) > 0 else float("nan")
        print(f"  {col:<28} {a_med:>18.2f} {g_med:>18.2f}")


if __name__ == "__main__":
    main()