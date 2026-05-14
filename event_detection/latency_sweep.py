"""
Latency sweep for the airborne event detector.

Runs the full pipeline (feature extraction -> rule -> evaluation) for multiple
buffer-lookahead values L. Produces:
  - latency_sweep_results.csv : one row per L with all metrics
  - latency_sweep_results.png : accuracy-vs-latency curve

Use this to pick the production latency budget. The relationship is monotone
in expectation (more lookahead = better recall and precision), so the curve
tells your team how much accuracy is sacrificed at each latency tier.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt

from extract_trajectory_features import (
    load_detections,
    load_actions,
    extract_features,
)
from airborne_rule import detect_airborne_events_offline, AirborneRuleConfig
from evaluate_airborne_detector import (
    load_gt_airborne_events,
    evaluate_airborne_detector,
)


# ============================================================
# Config
# ============================================================
DETECTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v4_A.json"
ACTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_actions.json"
FPS = 30.0

# Latency values to sweep (in frames at 30 fps).
# 15 = 0.5s, 30 = 1.0s, 45 = 1.5s, 60 = 2.0s, 90 = 3.0s
L_VALUES = [15, 30, 45, 60, 90]

# Rule config — same across all L values so we isolate the effect of latency.
# The rule may need tuning at very small L (less smoothing room).
RULE_CONFIG = AirborneRuleConfig()

OUTPUT_CSV = "latency_sweep_results.csv"
OUTPUT_PLOT = "latency_sweep_results.png"


# ============================================================
# Run one L value end-to-end
# ============================================================
def run_one_l(L: int, detections, gt_events, total_frames):
    """Returns a dict of metrics for this latency value."""
    features_df = extract_features(detections, buffer_l=L)
    predictions = detect_airborne_events_offline(features_df, RULE_CONFIG)
    result = evaluate_airborne_detector(
        predictions=predictions,
        gt_events=gt_events,
        total_frames=total_frames,
        fps=FPS,
    )
    return {
        "L_frames": L,
        "L_seconds": L / FPS,
        "n_gt": result.n_gt,
        "n_pred": result.n_pred,
        "n_matched": result.n_matched,
        "recall": result.recall,
        "precision": result.precision,
        "mean_start_error_frames": result.mean_start_error,
        "median_start_error_frames": result.median_start_error,
        "fp_per_minute": result.fp_per_minute,
    }, result


# ============================================================
# Plot
# ============================================================
def plot_sweep(df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: Recall and Precision
    ax = axes[0, 0]
    ax.plot(df["L_seconds"], df["recall"] * 100, "o-", label="Recall (PRIMARY)",
            color="firebrick", linewidth=2, markersize=8)
    ax.plot(df["L_seconds"], df["precision"] * 100, "s-", label="Precision",
            color="steelblue", linewidth=2, markersize=8)
    ax.set_xlabel("Latency budget L (seconds)")
    ax.set_ylabel("Metric (%)")
    ax.set_title("Recall and precision vs. latency")
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Mean start-frame error
    ax = axes[0, 1]
    ax.plot(df["L_seconds"], df["mean_start_error_frames"], "o-",
            color="darkgreen", linewidth=2, markersize=8, label="Mean")
    ax.plot(df["L_seconds"], df["median_start_error_frames"], "s--",
            color="seagreen", linewidth=2, markersize=8, label="Median")
    ax.set_xlabel("Latency budget L (seconds)")
    ax.set_ylabel("Frames")
    ax.set_title("Start-frame error of detected events\n(lower = boundaries closer to GT)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: FP per minute
    ax = axes[1, 0]
    ax.plot(df["L_seconds"], df["fp_per_minute"], "o-",
            color="darkorange", linewidth=2, markersize=8)
    ax.set_xlabel("Latency budget L (seconds)")
    ax.set_ylabel("FP / minute")
    ax.set_title("False positive rate\n(spurious 'airborne' events emitted per minute)")
    ax.grid(alpha=0.3)

    # Panel 4: count summary
    ax = axes[1, 1]
    width = 0.25
    x = range(len(df))
    ax.bar([i - width for i in x], df["n_gt"], width, label="GT events", color="black")
    ax.bar(x, df["n_matched"], width, label="Detected (TP)", color="forestgreen")
    ax.bar([i + width for i in x], df["n_pred"] - df["n_matched"], width,
           label="False positives", color="firebrick")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l:.1f}s" for l in df["L_seconds"]])
    ax.set_xlabel("Latency budget L")
    ax.set_ylabel("Event count")
    ax.set_title("Event counts: GT vs detected vs false positives")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    print(f"\nPlot saved to {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    print(f"Loading detections from {DETECTIONS_PATH}")
    detections = load_detections(DETECTIONS_PATH)
    print(f"  {len(detections)} frames")

    print(f"Loading GT actions from {ACTIONS_PATH}")
    gt_events = load_gt_airborne_events(ACTIONS_PATH)
    print(f"  {len(gt_events)} GT airborne events")

    total_frames = max(detections.keys()) + 1 if detections else 0

    print(f"\nSweeping L values: {L_VALUES}")
    print(f"  ({total_frames} frames total, {total_frames/FPS:.1f} seconds at {FPS} fps)\n")

    rows = []
    full_results = {}
    for L in L_VALUES:
        print(f"--- L = {L} frames ({L/FPS:.2f}s) ---")
        row, full_result = run_one_l(L, detections, gt_events, total_frames)
        rows.append(row)
        full_results[L] = full_result
        print(f"  n_pred={row['n_pred']}, matched={row['n_matched']}/{row['n_gt']}, "
              f"recall={row['recall']*100:.1f}%, precision={row['precision']*100:.1f}%, "
              f"FP/min={row['fp_per_minute']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSweep results written to {OUTPUT_CSV}")

    # Detailed summary
    print("\n" + "=" * 70)
    print("LATENCY SWEEP SUMMARY")
    print("=" * 70)
    print(f"  {'L (s)':>6}  {'Recall':>7}  {'Prec':>6}  "
          f"{'Mean err':>9}  {'FP/min':>7}  {'TP':>3}  {'FP':>3}")
    for _, row in df.iterrows():
        print(f"  {row['L_seconds']:>6.2f}  "
              f"{row['recall']*100:>6.1f}%  "
              f"{row['precision']*100:>5.1f}%  "
              f"{row['mean_start_error_frames']:>9.1f}  "
              f"{row['fp_per_minute']:>7.2f}  "
              f"{row['n_matched']:>3}  {row['n_pred'] - row['n_matched']:>3}")

    # Sanity checks: which L value would the team likely pick?
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find smallest L that achieves recall >= 0.9 (since recall is the must-have)
    high_recall_rows = df[df["recall"] >= 0.9].sort_values("L_seconds")
    if len(high_recall_rows) > 0:
        best_lo_lat = high_recall_rows.iloc[0]
        print(f"  Smallest L with recall >= 90%: "
              f"{best_lo_lat['L_seconds']:.2f}s "
              f"(recall {best_lo_lat['recall']*100:.1f}%, precision {best_lo_lat['precision']*100:.1f}%)")
    else:
        print("  No L achieved recall >= 90%. Rule needs tuning before this is shippable.")

    # Plateau detection: where does recall stop improving?
    if len(df) >= 2:
        df_sorted = df.sort_values("L_seconds").reset_index(drop=True)
        max_recall = df_sorted["recall"].max()
        plateau_rows = df_sorted[df_sorted["recall"] >= max_recall - 0.01]
        if len(plateau_rows) > 0:
            min_plateau = plateau_rows.iloc[0]
            print(f"  Recall saturates at: {min_plateau['L_seconds']:.2f}s "
                  f"(recall {min_plateau['recall']*100:.1f}%). "
                  f"Beyond this, more latency stops helping.")

    plot_sweep(df, OUTPUT_PLOT)


if __name__ == "__main__":
    main()