"""
Evaluate a list of detected airborne events against hand-labeled ground truth.

Metrics
-------
Event-level (one-to-one matching by max IoU):
  Recall     — fraction of GT events matched to at least one prediction
  Precision  — fraction of predictions matched to at least one GT event

Boundary accuracy (matched pairs only):
  Start-frame error — |pred_start - gt_start|, mean + median
  End-frame error   — |pred_end   - gt_end|,   mean + median

Span quality (matched pairs only):
  Temporal IoU — intersection / union of matched pred and GT spans

Frame-level (across all frames):
  Recall     — fraction of GT-airborne frames predicted as airborne
  Precision  — fraction of predicted-airborne frames that are GT-airborne
  F1         — harmonic mean of the two

GT format (new):
  {"start_frame": N, "end_frame": M, "action": "airborne"}
  Frames are inclusive on both ends.
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from airborne_rule import AirborneEvent


@dataclass
class GTAirborneEvent:
    start_frame: int
    end_frame: int


@dataclass
class EvalResult:
    # Counts
    n_gt: int
    n_pred: int
    n_matched: int

    # Event-level
    recall: float
    precision: float

    # Boundary accuracy (matched only)
    mean_start_error: float
    median_start_error: float
    mean_end_error: float
    median_end_error: float

    # Span quality (matched only)
    mean_iou: float
    median_iou: float

    # Frame-level
    frame_recall: float
    frame_precision: float
    frame_f1: float

    # Per-event detail for human inspection
    per_gt_status: List[Dict]
    per_pred_status: List[Dict]


# ============================================================
# Helpers
# ============================================================

def load_gt_airborne_events(actions_json_path: str) -> List[GTAirborneEvent]:
    with open(actions_json_path) as f:
        data = json.load(f)
    events = []
    for evt in data.get("events", []):
        if evt.get("action", "").lower() != "airborne":
            continue
        events.append(GTAirborneEvent(
            start_frame=int(evt["start_frame"]),
            end_frame=int(evt["end_frame"]),
        ))
    events.sort(key=lambda e: e.start_frame)
    return events


def _pred_end(p: AirborneEvent) -> int:
    # duration_frames is the span length; frames are inclusive so end = start + duration - 1
    return p.start_frame + p.duration_frames - 1


def _temporal_iou(
    a_start: int, a_end: int,
    b_start: int, b_end: int,
) -> float:
    inter_start = max(a_start, b_start)
    inter_end   = min(a_end,   b_end)
    inter = max(0, inter_end - inter_start + 1)
    if inter == 0:
        return 0.0
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - inter
    return inter / union


def _frame_set(start: int, end: int) -> set:
    return set(range(start, end + 1))


# ============================================================
# Matching: greedy max-IoU (highest IoU pair assigned first)
# ============================================================

def match_predictions_to_gt(
    predictions: List[AirborneEvent],
    gt_events: List[GTAirborneEvent],
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    One-to-one matching. All overlapping (pred, gt) pairs are scored by
    temporal IoU; the highest-IoU pair is assigned first, then those two
    are removed from the pool. Ties broken by smaller start-frame distance.

    Returns:
      matches         — list of (pred_index, gt_index)
      unmatched_preds — pred indices with no GT match
      unmatched_gts   — GT indices with no pred match
    """
    candidates = []
    for pi, p in enumerate(predictions):
        pe = _pred_end(p)
        for gi, g in enumerate(gt_events):
            iou = _temporal_iou(p.start_frame, pe, g.start_frame, g.end_frame)
            if iou > 0:
                start_err = abs(p.start_frame - g.start_frame)
                # Sort key: descending IoU (negate), then ascending start error
                candidates.append((-iou, start_err, pi, gi))

    candidates.sort()

    matches: List[Tuple[int, int]] = []
    used_preds: set = set()
    used_gts: set = set()

    for _, _, pi, gi in candidates:
        if pi in used_preds or gi in used_gts:
            continue
        matches.append((pi, gi))
        used_preds.add(pi)
        used_gts.add(gi)

    unmatched_preds = [pi for pi in range(len(predictions)) if pi not in used_preds]
    unmatched_gts   = [gi for gi in range(len(gt_events))   if gi not in used_gts]
    return matches, unmatched_preds, unmatched_gts


# ============================================================
# Main evaluation
# ============================================================

def evaluate_airborne_detector(
    predictions: List[AirborneEvent],
    gt_events: List[GTAirborneEvent],
    total_frames: int,
) -> EvalResult:
    matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(predictions, gt_events)

    n_gt      = len(gt_events)
    n_pred    = len(predictions)
    n_matched = len(matches)

    recall    = n_matched / n_gt   if n_gt   > 0 else 0.0
    precision = n_matched / n_pred if n_pred > 0 else 0.0

    # Boundary + IoU stats over matched pairs
    start_errors, end_errors, ious = [], [], []
    for pi, gi in matches:
        p, g = predictions[pi], gt_events[gi]
        pe = _pred_end(p)
        start_errors.append(abs(p.start_frame - g.start_frame))
        end_errors.append(abs(pe - g.end_frame))
        ious.append(_temporal_iou(p.start_frame, pe, g.start_frame, g.end_frame))

    def _mean(arr):   return float(np.mean(arr))   if arr else float("nan")
    def _median(arr): return float(np.median(arr)) if arr else float("nan")

    # Frame-level metrics
    gt_frames   = set()
    pred_frames = set()
    for g in gt_events:
        gt_frames |= _frame_set(g.start_frame, g.end_frame)
    for p in predictions:
        pred_frames |= _frame_set(p.start_frame, _pred_end(p))

    tp_frames = len(gt_frames & pred_frames)
    frame_recall    = tp_frames / len(gt_frames)   if gt_frames   else 0.0
    frame_precision = tp_frames / len(pred_frames) if pred_frames else 0.0
    if frame_recall + frame_precision > 0:
        frame_f1 = 2 * frame_recall * frame_precision / (frame_recall + frame_precision)
    else:
        frame_f1 = 0.0

    # Per-event detail
    pred_to_gt = {pi: gi for pi, gi in matches}
    gt_to_pred = {gi: pi for pi, gi in matches}

    per_gt_status = []
    for gi, g in enumerate(gt_events):
        if gi in gt_to_pred:
            pi = gt_to_pred[gi]
            p  = predictions[pi]
            pe = _pred_end(p)
            per_gt_status.append({
                "gt_start": g.start_frame,
                "gt_end":   g.end_frame,
                "detected": True,
                "pred_start":        p.start_frame,
                "pred_end":          pe,
                "start_error_frames": abs(p.start_frame - g.start_frame),
                "end_error_frames":   abs(pe - g.end_frame),
                "iou":               _temporal_iou(p.start_frame, pe, g.start_frame, g.end_frame),
            })
        else:
            per_gt_status.append({
                "gt_start": g.start_frame,
                "gt_end":   g.end_frame,
                "detected": False,
                "pred_start": None, "pred_end": None,
                "start_error_frames": None,
                "end_error_frames":   None,
                "iou":               0.0,
            })

    per_pred_status = []
    for pi, p in enumerate(predictions):
        pe = _pred_end(p)
        if pi in pred_to_gt:
            gi = pred_to_gt[pi]
            g  = gt_events[gi]
            per_pred_status.append({
                "pred_start": p.start_frame,
                "pred_end":   pe,
                "matched": True,
                "matched_gt_start": g.start_frame,
                "matched_gt_end":   g.end_frame,
                "iou": _temporal_iou(p.start_frame, pe, g.start_frame, g.end_frame),
            })
        else:
            per_pred_status.append({
                "pred_start": p.start_frame,
                "pred_end":   pe,
                "matched": False,
                "matched_gt_start": None,
                "matched_gt_end":   None,
                "iou": 0.0,
            })

    return EvalResult(
        n_gt=n_gt, n_pred=n_pred, n_matched=n_matched,
        recall=recall, precision=precision,
        mean_start_error=_mean(start_errors), median_start_error=_median(start_errors),
        mean_end_error=_mean(end_errors),     median_end_error=_median(end_errors),
        mean_iou=_mean(ious),                 median_iou=_median(ious),
        frame_recall=frame_recall, frame_precision=frame_precision, frame_f1=frame_f1,
        per_gt_status=per_gt_status,
        per_pred_status=per_pred_status,
    )


# ============================================================
# Pretty printer
# ============================================================

def print_eval_result(result: EvalResult):
    W = 70
    print("=" * W)
    print("AIRBORNE DETECTOR EVAL")
    print("=" * W)
    print(f"  GT events   : {result.n_gt}")
    print(f"  Predictions : {result.n_pred}")
    print(f"  Matched     : {result.n_matched}")

    print()
    print("  -- Event-level --")
    print(f"  Recall      : {result.recall*100:.1f}%  (PRIMARY — must be high)")
    print(f"  Precision   : {result.precision*100:.1f}%")

    print()
    print("  -- Boundary accuracy (matched only) --")
    print(f"  Start error : mean={result.mean_start_error:.1f}f  median={result.median_start_error:.1f}f")
    print(f"  End error   : mean={result.mean_end_error:.1f}f  median={result.median_end_error:.1f}f")

    print()
    print("  -- Span quality (matched only) --")
    print(f"  Temporal IoU: mean={result.mean_iou:.3f}  median={result.median_iou:.3f}")

    print()
    print("  -- Frame-level --")
    print(f"  Recall      : {result.frame_recall*100:.1f}%")
    print(f"  Precision   : {result.frame_precision*100:.1f}%")
    print(f"  F1          : {result.frame_f1*100:.1f}%")

    print()
    print("  -- Per-GT status --")
    for s in result.per_gt_status:
        if s["detected"]:
            print(f"    [HIT]    gt=[{s['gt_start']:>4},{s['gt_end']:>4}]  "
                  f"pred=[{s['pred_start']:>4},{s['pred_end']:>4}]  "
                  f"start_err={s['start_error_frames']:>3}f  "
                  f"end_err={s['end_error_frames']:>3}f  "
                  f"iou={s['iou']:.3f}")
        else:
            print(f"    [MISS]   gt=[{s['gt_start']:>4},{s['gt_end']:>4}]")

    fp_list = [s for s in result.per_pred_status if not s["matched"]]
    if fp_list:
        print()
        print("  -- False positives --")
        for s in fp_list:
            print(f"    pred=[{s['pred_start']:>4},{s['pred_end']:>4}]")

    print("=" * W)


# ============================================================
# Standalone harness
# ============================================================
if __name__ == "__main__":
    import pandas as pd
    from airborne_rule import detect_airborne_events_offline, AirborneRuleConfig

    ACTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip2_actions.json"
    FEATURES_CSV = "trajectory_features_c2.csv"

    df = pd.read_csv(FEATURES_CSV)
    gt_events = load_gt_airborne_events(ACTIONS_PATH)
    config = AirborneRuleConfig()
    predictions = detect_airborne_events_offline(df, config)

    result = evaluate_airborne_detector(
        predictions=predictions,
        gt_events=gt_events,
        total_frames=int(df["frame"].max()) + 1,
    )
    print_eval_result(result)
