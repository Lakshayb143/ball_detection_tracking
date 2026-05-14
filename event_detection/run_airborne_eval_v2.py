"""
Batch airborne event evaluation pipeline v2.

This is a parallel runner for the v2 no-pose detector:
  1. Extract v2 trajectory features from detections_v5
  2. Run the v2 launch-candidate state machine
  3. Evaluate predictions against GT spans

By default, output is written to outputs/airborne_eval_v5_v2 so the current
v5 outputs remain untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ED_DIR = Path(__file__).resolve().parent
ROOT = ED_DIR.parent
DET_DIR = ROOT / "detections_v5"
GT_DIR = ROOT / "ground_truths"
DEFAULT_OUT_DIR = ROOT / "outputs" / "airborne_eval_v5_v2"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ED_DIR))

from airborne_rule_v2 import AirborneRuleV2Config
from airborne_state_machine_v2 import (
    StateMachineV2Config,
    detect_airborne_events_with_state_machine_v2,
)
from evaluate_airborne_detector import (
    EvalResult,
    evaluate_airborne_detector,
    load_gt_airborne_events,
    print_eval_result,
)
from extract_trajectory_features_without_pose_v2 import load_detections, extract_features


def _resolve_gt_path(clip_name: str) -> Path:
    primary = GT_DIR / f"{clip_name}_actions.json"
    if primary.exists():
        return primary
    fallback = ROOT / f"{clip_name}_actions.json"
    if fallback.exists():
        return fallback
    return primary


def _plot_features(df, gt_events, clip_name, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(6, 1, figsize=(16, 16), sharex=True)
    fig.suptitle(f"{clip_name} - trajectory features v2", y=1.0)

    def shade(ax):
        for g in gt_events:
            ax.axvspan(g.start_frame, g.end_frame, alpha=0.25, color="red", zorder=0)

    panels = [
        ("detection_density", "density", "Detection density"),
        ("frames_since_last_det", "frames", "Frames since last detection"),
        ("dy_per_frame_last_pair", "dy/frame", "y velocity, negative means rising"),
        ("y_drop_after_gap", "px", "Upward reappearance y drop after gap"),
        ("parabola_inlier_ratio_15", "ratio", "15-frame parabola inlier ratio"),
        ("dy_buffer_first_to_last", "px", "Buffer-level y change"),
    ]
    for ax, (col, ylabel, title) in zip(axes, panels):
        ax.plot(df["frame"], df[col], lw=1)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        shade(ax)

    axes[-1].set_xlabel("frame")
    fig.legend(
        handles=[Patch(facecolor="red", alpha=0.25, label="GT airborne")],
        loc="upper center",
        ncol=1,
        bbox_to_anchor=(0.5, 1.0),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def _result_to_dict(clip_name: str, result: EvalResult, completed_events) -> dict:
    out = {
        "clip": clip_name,
        "n_gt": result.n_gt,
        "n_pred": result.n_pred,
        "n_matched": result.n_matched,
        "recall": round(result.recall, 4),
        "precision": round(result.precision, 4),
        "mean_start_error": round(result.mean_start_error, 2) if not np.isnan(result.mean_start_error) else None,
        "median_start_error": round(result.median_start_error, 2) if not np.isnan(result.median_start_error) else None,
        "mean_end_error": round(result.mean_end_error, 2) if not np.isnan(result.mean_end_error) else None,
        "median_end_error": round(result.median_end_error, 2) if not np.isnan(result.median_end_error) else None,
        "mean_iou": round(result.mean_iou, 4) if not np.isnan(result.mean_iou) else None,
        "median_iou": round(result.median_iou, 4) if not np.isnan(result.median_iou) else None,
        "frame_recall": round(result.frame_recall, 4),
        "frame_precision": round(result.frame_precision, 4),
        "frame_f1": round(result.frame_f1, 4),
        "per_gt": result.per_gt_status,
        "per_pred": result.per_pred_status,
    }
    out["completed_events"] = [
        {
            "start_frame": e.start_frame,
            "end_frame": e.end_frame,
            "duration_frames": e.duration_frames,
            "fired_at_frame": e.fired_at_frame,
            "fire_score": round(e.fire_score, 4),
            "fire_breakdown": e.fire_breakdown,
            "end_reason": e.end_reason,
        }
        for e in completed_events
    ]
    return out


def run_clip(clip_name: str, out_dir: Path, save_plots: bool) -> dict | None:
    det_path = DET_DIR / f"{clip_name}.json"
    gt_path = _resolve_gt_path(clip_name)
    clip_out = out_dir / clip_name
    clip_out.mkdir(parents=True, exist_ok=True)

    if not det_path.exists():
        print(f"[skip] {clip_name} - no detection JSON at {det_path}")
        return None
    if not gt_path.exists():
        print(f"[skip] {clip_name} - no GT JSON at {gt_path}")
        return None

    print(f"\n{'=' * 60}")
    print(f"  {clip_name} v2")
    print(f"{'=' * 60}")

    detections = load_detections(str(det_path))
    total_frames = max(detections.keys()) + 1
    print(
        f"  detections loaded: {total_frames} frames, "
        f"{sum(1 for v in detections.values() if v is not None)} with ball"
    )

    df = extract_features(detections, buffer_l=30, arc_window_frames=15)
    features_csv = clip_out / "features.csv"
    df.to_csv(features_csv, index=False)
    print(f"  features -> {features_csv}")

    gt_events = load_gt_airborne_events(str(gt_path))
    print(f"  GT events: {len(gt_events)}")

    rule_config = AirborneRuleV2Config()
    sm_config = StateMachineV2Config()
    completed = detect_airborne_events_with_state_machine_v2(df, rule_config, sm_config)
    predictions = [e.to_airborne_event() for e in completed]
    print(f"  predictions: {len(predictions)}")
    for event in completed:
        print(
            f"    pred start={event.start_frame} end={event.end_frame} "
            f"fired_at={event.fired_at_frame} reason={event.end_reason} "
            f"score={event.fire_score:.2f}"
        )

    result = evaluate_airborne_detector(predictions, gt_events, total_frames)
    print_eval_result(result)

    result_dict = _result_to_dict(clip_name, result, completed)
    eval_json = clip_out / "eval_result.json"
    with open(eval_json, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"  eval -> {eval_json}")

    if save_plots:
        plot_path = clip_out / "features_plot.png"
        _plot_features(df, gt_events, clip_name, plot_path)
        print(f"  plot -> {plot_path}")

    return result_dict


def write_summary(all_results: list[dict], out_dir: Path):
    summary_cols = [
        "clip",
        "n_gt",
        "n_pred",
        "n_matched",
        "recall",
        "precision",
        "mean_start_error",
        "mean_end_error",
        "mean_iou",
        "median_iou",
        "frame_recall",
        "frame_precision",
        "frame_f1",
    ]
    summary_rows = [{k: r[k] for k in summary_cols} for r in all_results]

    numeric_cols = summary_cols[1:]
    avgs = {"clip": "MACRO_AVG"}
    for col in numeric_cols:
        vals = [r[col] for r in summary_rows if r[col] is not None]
        avgs[col] = round(float(np.mean(vals)), 4) if vals else None
    summary_rows.append(avgs)

    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"
    with open(summary_json, "w") as f:
        json.dump(summary_rows, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print(f"\n{'=' * 60}")
    print("CONSOLIDATED SUMMARY V2")
    print(f"{'=' * 60}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\n  -> {summary_json}")
    print(f"  -> {summary_csv}")


def find_clips(clip_filter: list[int] | None) -> list[str]:
    if clip_filter:
        return [f"clip{n}" for n in clip_filter]
    names = {p.stem for p in DET_DIR.glob("*.json")}
    return sorted(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="+", type=int, metavar="N")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for clip_name in find_clips(args.clips):
        result = run_clip(clip_name, out_dir=out_dir, save_plots=not args.no_plots)
        if result is not None:
            results.append(result)

    if len(results) > 1:
        write_summary(results, out_dir)
    elif len(results) == 1:
        print("\n(Only 1 clip evaluated - skipping consolidated summary)")


if __name__ == "__main__":
    main()
