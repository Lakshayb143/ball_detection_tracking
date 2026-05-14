"""
Trajectory feature extractor for airborne event detection.

For each frame t, computes features over a rolling buffer of L frames.
If a pose JSON is provided, also adds pose features per frame.

Inputs:
  - Detection JSON (from your v4 tracker)
  - Actions JSON (hand-labeled, used for plot overlay only)
  - Pose JSON (from precompute_pose_for_clip.py, optional)

Outputs:
  - trajectory_features.csv: per-frame features (incl. pose if provided)
  - trajectory_features.png: features over time with airborne windows shaded
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# Config
# ============================================================
DETECTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v4_A.json"
ACTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_actions.json"
POSE_JSON_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_pose_features.json"  # set to None to skip

OUTPUT_CSV = "trajectory_features.csv"
OUTPUT_PLOT = "trajectory_features.png"

BUFFER_L = 30

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


def load_pose_features(pose_json_path):
    """
    Returns dict: frame_idx -> {
        "any_kicking": bool,
        "max_kicking_conf": float,
        "kicking_ankle_xy": (x, y) or None,   # ankle of the kicking leg
        "n_kicking_players": int,
    }
    If a frame has no pose entry (or was gated out during precompute), it gets
    a zero entry — same as "no kicking detected."
    """
    if pose_json_path is None:
        return {}
    with open(pose_json_path) as f:
        data = json.load(f)

    out = {}
    for frame_str, frame_data in data.get("frames", {}).items():
        f = int(frame_str)
        players = frame_data.get("players", [])
        kicking_players = [p for p in players if p.get("is_kicking_pose")]

        if not kicking_players:
            out[f] = {
                "any_kicking": False,
                "max_kicking_conf": 0.0,
                "kicking_ankle_xy": None,
                "n_kicking_players": 0,
            }
            continue

        best = max(kicking_players, key=lambda p: p.get("confidence", 0.0))
        kicking_leg = best.get("kicking_leg")
        ankle_xy = (best.get("left_ankle_xy") if kicking_leg == "left"
                    else best.get("right_ankle_xy"))
        out[f] = {
            "any_kicking": True,
            "max_kicking_conf": float(best.get("confidence", 0.0)),
            "kicking_ankle_xy": tuple(ankle_xy) if ankle_xy else None,
            "n_kicking_players": len(kicking_players),
        }
    return out


# ============================================================
# Feature extraction
# ============================================================
def extract_features(detections, buffer_l, pose_features=None):
    """
    Per-frame features over a rolling buffer.
    pose_features (optional): dict frame_idx -> pose info from load_pose_features().
    """
    if not detections:
        return pd.DataFrame()

    pose_features = pose_features or {}
    max_frame = max(detections.keys())
    rows = []

    for t in range(max_frame + 1):
        buffer_start = max(0, t - buffer_l + 1)
        buffer_frames = list(range(buffer_start, t + 1))
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
            last_x_val = float(last_x)
            last_y_val = float(last_y)
        else:
            median_y = np.nan
            y_range = np.nan
            last_y_val = np.nan
            last_x_val = np.nan
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

        # Pose features for this frame
        pose_t = pose_features.get(t, {
            "any_kicking": False,
            "max_kicking_conf": 0.0,
            "kicking_ankle_xy": None,
            "n_kicking_players": 0,
        })

        # Distance from last ball detection to nearest kicking ankle.
        # This is what makes pose actionable: a kicker far from the ball is
        # less informative than a kicker right at the ball.
        dist_ball_to_kicker = np.nan
        if pose_t["any_kicking"] and pose_t["kicking_ankle_xy"] is not None:
            if not np.isnan(last_x_val) and not np.isnan(last_y_val):
                ax, ay = pose_t["kicking_ankle_xy"]
                dist_ball_to_kicker = float(
                    np.hypot(ax - last_x_val, ay - last_y_val)
                )

        rows.append({
            "frame": t,
            "n_detections_in_buffer": n_dets,
            "detection_density": n_dets / buffer_l,
            "median_y_in_buffer": median_y,
            "y_range_in_buffer": y_range,
            "frames_since_last_det": frames_since_last,
            "last_x": last_x_val,
            "last_y": last_y_val,
            "dy_last_pair": dy_last_pair,
            "dy_per_frame_last_pair": dy_per_frame_last_pair,
            "dy_buffer_first_to_last": dy_buffer_first_to_last,
            # Pose features
            "pose_any_kicking": bool(pose_t["any_kicking"]),
            "pose_max_kicking_conf": float(pose_t["max_kicking_conf"]),
            "pose_n_kicking_players": int(pose_t["n_kicking_players"]),
            "pose_dist_ball_to_kicker": dist_ball_to_kicker,
        })

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================
def plot_features(df, events, buffer_l, output_path, has_pose):
    n_panels = 6 if has_pose else 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 2.5 * n_panels), sharex=True)

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
        for start, end, _ in airborne_windows:
            ax.axvspan(start, end, alpha=0.25, color="red", zorder=0)
        for start, end, _ in other_event_windows:
            ax.axvspan(start, end, alpha=0.10, color="blue", zorder=0)

    # Panel 1: detection density and gap
    ax = axes[0]
    ax.plot(df["frame"], df["detection_density"], label="detection density",
            color="black", lw=1)
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

    # Panel 2: last_y
    ax = axes[1]
    ax.plot(df["frame"], df["last_y"], label="last y", color="navy", lw=1)
    ax.plot(df["frame"], df["median_y_in_buffer"], label="median y in buffer",
            color="teal", lw=1, alpha=0.7)
    ax.set_ylabel("y (px) - lower = higher in image")
    ax.invert_yaxis()
    ax.set_title("Ball y-coordinate (image space)")
    ax.legend(loc="upper left")
    shade_axis(ax)

    # Panel 3: y range
    ax = axes[2]
    ax.plot(df["frame"], df["y_range_in_buffer"], color="purple", lw=1)
    ax.set_ylabel("y range (px)")
    ax.set_title("Spread of ball y over buffer")
    shade_axis(ax)

    # Panel 4: dy per frame
    ax = axes[3]
    ax.plot(df["frame"], df["dy_per_frame_last_pair"], color="darkgreen", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("dy / frame (px)")
    ax.set_title("Most recent y-velocity (negative = ball moving up = airborne signal)")
    shade_axis(ax)

    # Panel 5: cumulative buffer y change
    ax = axes[4]
    ax.plot(df["frame"], df["dy_buffer_first_to_last"], color="firebrick", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("y_last - y_first (px)")
    ax.set_title("Buffer-level y change (strong negative = ball climbed during L frames)")
    shade_axis(ax)

    # Panel 6 (optional): pose
    if has_pose:
        ax = axes[5]
        ax.plot(df["frame"], df["pose_max_kicking_conf"],
                label="max kicking conf", color="darkorange", lw=1)
        ax.fill_between(df["frame"], 0, df["pose_max_kicking_conf"].fillna(0),
                        where=df["pose_any_kicking"].astype(bool),
                        color="darkorange", alpha=0.3, label="any kicking")
        ax.set_ylabel("kicking conf")
        ax.set_title("Pose: kicking-pose confidence per frame "
                     "(orange shading = at least one kicking player)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("frame")
        ax.legend(loc="upper left")
        shade_axis(ax)
    else:
        axes[4].set_xlabel("frame")

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

    pose_features = {}
    has_pose = False
    if POSE_JSON_PATH is not None:
        try:
            print(f"\nLoading pose features from {POSE_JSON_PATH}")
            pose_features = load_pose_features(POSE_JSON_PATH)
            has_pose = len(pose_features) > 0
            n_kicking = sum(1 for p in pose_features.values() if p["any_kicking"])
            print(f"  {len(pose_features)} frames, {n_kicking} with kicking pose")
        except FileNotFoundError:
            print(f"  pose JSON not found, continuing without pose features")
            has_pose = False

    print(f"\nLoading actions from {ACTIONS_PATH}")
    events = load_actions(ACTIONS_PATH)
    print(f"  {len(events)} events")
    for evt in events:
        airborne = "airborne" if evt["action"] in AIRBORNE_ACTIONS else "ground"
        print(f"  frame {evt['frame']:>4}, {evt['action']:<25} ({airborne})")

    print(f"\nExtracting features with buffer L = {BUFFER_L}")
    df = extract_features(detections, BUFFER_L, pose_features=pose_features)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  features written to {OUTPUT_CSV}")

    print(f"\nPlotting...")
    plot_features(df, events, BUFFER_L, OUTPUT_PLOT, has_pose=has_pose)

    # Summary stats
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
        "detection_density", "frames_since_last_det", "last_y",
        "y_range_in_buffer", "dy_per_frame_last_pair",
        "dy_buffer_first_to_last",
    ]
    if has_pose:
        feature_cols += ["pose_max_kicking_conf", "pose_dist_ball_to_kicker"]

    print(f"\n  {'feature':<32} {'airborne (median)':>18} {'ground (median)':>18}")
    for col in feature_cols:
        airborne_vals = df.loc[is_airborne, col].dropna()
        ground_vals = df.loc[~is_airborne, col].dropna()
        a_med = airborne_vals.median() if len(airborne_vals) > 0 else float("nan")
        g_med = ground_vals.median() if len(ground_vals) > 0 else float("nan")
        print(f"  {col:<32} {a_med:>18.2f} {g_med:>18.2f}")


if __name__ == "__main__":
    main()