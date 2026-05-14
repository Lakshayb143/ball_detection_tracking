"""
Batch airborne event evaluation pipeline.

For each clip:
  1. Extract trajectory features (no pose) from the detection JSON
  2. Run airborne state machine to produce predicted events
  3. Evaluate predictions against GT with the full metric suite
  4. Save per-clip results to outputs/airborne_eval/<clip>/

Then write a consolidated summary across all clips to:
  outputs/airborne_eval/summary.json
  outputs/airborne_eval/summary.csv

Usage (run from repo root):
  python event_detection/run_airborne_eval.py                  # all clips with both det + GT
  python event_detection/run_airborne_eval.py --clips 1 2 3    # specific clips
  python event_detection/run_airborne_eval.py --no-plots        # skip matplotlib plots (faster)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ED_DIR     = Path(__file__).resolve().parent
ROOT       = ED_DIR.parent
DET_DIR    = ROOT / "detections_v5"
GT_DIR     = ROOT / "ground_truths"
OUT_DIR    = ROOT / "outputs" / "airborne_eval_v5"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ED_DIR))

from extract_trajectory_features_without_pose import load_detections, extract_features
from airborne_rule import AirborneRuleConfig
from airborne_state_machine import StateMachineConfig, detect_airborne_events_with_state_machine
from evaluate_airborne_detector import (
    load_gt_airborne_events,
    evaluate_airborne_detector,
    print_eval_result,
    EvalResult,
)


# ============================================================
# Helpers
# ============================================================

def _plot_features(df, gt_events, clip_name, out_path):
    """Minimal plot: feature panels with GT airborne windows shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f"{clip_name} — trajectory features", y=1.0)

    def shade(ax):
        for g in gt_events:
            ax.axvspan(g.start_frame, g.end_frame, alpha=0.25, color="red", zorder=0)

    panels = [
        ("detection_density",        "density (0-1)",    "Detection density"),
        ("dy_per_frame_last_pair",    "dy/frame (px)",    "y-velocity (neg = ball rising)"),
        ("frames_since_last_det",     "frames",           "Frames since last detection"),
        ("y_range_in_buffer",         "y range (px)",     "y spread over buffer"),
        ("dy_buffer_first_to_last",   "y_last-y_first",   "Buffer-level y change"),
    ]
    for ax, (col, ylabel, title) in zip(axes, panels):
        ax.plot(df["frame"], df[col], lw=1)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        shade(ax)

    axes[-1].set_xlabel("frame")
    legend_handles = [Patch(facecolor="red", alpha=0.25, label="GT airborne")]
    fig.legend(handles=legend_handles, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def _result_to_dict(clip_name: str, result: EvalResult) -> dict:
    return {
        "clip": clip_name,
        "n_gt":    result.n_gt,
        "n_pred":  result.n_pred,
        "n_matched": result.n_matched,
        "recall":     round(result.recall, 4),
        "precision":  round(result.precision, 4),
        "mean_start_error":  round(result.mean_start_error, 2)  if not np.isnan(result.mean_start_error)  else None,
        "median_start_error": round(result.median_start_error, 2) if not np.isnan(result.median_start_error) else None,
        "mean_end_error":    round(result.mean_end_error, 2)    if not np.isnan(result.mean_end_error)    else None,
        "median_end_error":  round(result.median_end_error, 2)  if not np.isnan(result.median_end_error)  else None,
        "mean_iou":   round(result.mean_iou, 4)   if not np.isnan(result.mean_iou)   else None,
        "median_iou": round(result.median_iou, 4) if not np.isnan(result.median_iou) else None,
        "frame_recall":    round(result.frame_recall, 4),
        "frame_precision": round(result.frame_precision, 4),
        "frame_f1":        round(result.frame_f1, 4),
        "per_gt":  result.per_gt_status,
        "per_pred": result.per_pred_status,
    }


# ============================================================
# Per-clip pipeline
# ============================================================

def run_clip(clip_name: str, save_plots: bool) -> dict | None:
    det_path = DET_DIR / f"{clip_name}.json"
    gt_path  = GT_DIR  / f"{clip_name}_actions.json"
    clip_out = OUT_DIR / clip_name
    clip_out.mkdir(parents=True, exist_ok=True)

    if not det_path.exists():
        print(f"[skip] {clip_name} — no detection JSON at {det_path}")
        return None
    if not gt_path.exists():
        print(f"[skip] {clip_name} — no GT JSON at {gt_path}")
        return None

    print(f"\n{'='*60}")
    print(f"  {clip_name}")
    print(f"{'='*60}")

    # 1. Load detections + extract features
    detections = load_detections(str(det_path))
    total_frames = max(detections.keys()) + 1
    print(f"  detections loaded: {total_frames} frames, "
          f"{sum(1 for v in detections.values() if v is not None)} with ball")

    features_csv = clip_out / "features.csv"
    df = extract_features(detections, buffer_l=30)
    df.to_csv(features_csv, index=False)
    print(f"  features -> {features_csv}")

    # 2. Load GT
    gt_events = load_gt_airborne_events(str(gt_path))
    print(f"  GT events: {len(gt_events)}")

    # 3. Detect with state machine
    rule_config = AirborneRuleConfig()
    sm_config   = StateMachineConfig()
    completed   = detect_airborne_events_with_state_machine(df, rule_config, sm_config)
    predictions = [e.to_airborne_event() for e in completed]
    print(f"  predictions: {len(predictions)}")

    # 4. Evaluate
    result = evaluate_airborne_detector(predictions, gt_events, total_frames)
    print_eval_result(result)

    # 5. Save per-clip JSON
    result_dict = _result_to_dict(clip_name, result)
    eval_json = clip_out / "eval_result.json"
    with open(eval_json, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"  eval -> {eval_json}")

    # 6. Optional plot
    if save_plots:
        plot_path = clip_out / "features_plot.png"
        _plot_features(df, gt_events, clip_name, plot_path)
        print(f"  plot -> {plot_path}")

    return result_dict


# ============================================================
# Consolidated summary
# ============================================================

def write_summary(all_results: list[dict]):
    summary_cols = [
        "clip", "n_gt", "n_pred", "n_matched",
        "recall", "precision",
        "mean_start_error", "mean_end_error",
        "mean_iou", "median_iou",
        "frame_recall", "frame_precision", "frame_f1",
    ]

    # Strip per-event detail for the summary JSON
    summary_rows = [{k: r[k] for k in summary_cols} for r in all_results]

    # Add macro averages row
    numeric_cols = summary_cols[1:]
    avgs = {"clip": "MACRO_AVG"}
    for col in numeric_cols:
        vals = [r[col] for r in summary_rows if r[col] is not None]
        avgs[col] = round(float(np.mean(vals)), 4) if vals else None
    summary_rows.append(avgs)

    summary_json = OUT_DIR / "summary.json"
    summary_csv  = OUT_DIR / "summary.csv"

    with open(summary_json, "w") as f:
        json.dump(summary_rows, f, indent=2)

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print(f"\n{'='*60}")
    print("CONSOLIDATED SUMMARY")
    print(f"{'='*60}")
    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False))
    print(f"\n  -> {summary_json}")
    print(f"  -> {summary_csv}")


# ============================================================
# Entry point
# ============================================================

def find_clips(clip_filter: list[int] | None) -> list[str]:
    if clip_filter:
        return [f"clip{n}" for n in clip_filter]
    # Auto-discover: any clip with both detection + GT
    names = set()
    for p in DET_DIR.glob("*.json"):
        names.add(p.stem)
    return sorted(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="+", type=int, metavar="N",
                        help="Clip numbers to evaluate (e.g. 1 2 3). Default: all with det+GT.")
    parser.add_argument("--no-plots", action="store_true", help="Skip feature plots.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clip_names = find_clips(args.clips)
    if not clip_names:
        print("No clips found.")
        return

    all_results = []
    for clip_name in clip_names:
        result = run_clip(clip_name, save_plots=not args.no_plots)
        if result is not None:
            all_results.append(result)

    if len(all_results) > 1:
        write_summary(all_results)
    elif len(all_results) == 1:
        print(f"\n(Only 1 clip evaluated — skipping consolidated summary)")


if __name__ == "__main__":
    main()
