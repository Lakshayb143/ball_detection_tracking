"""
Delta diagnostic — compare two tracker output JSONs against ground truth.

Tells you exactly which frames changed between two configs and why,
plus analyzes the quality of Phase 1 rejections if a rejection log is provided.

Use cases:
  - Phase 1 ablation: A_path = config A (both off), B_path = config B (P1 only)
  - Phase 2 ablation: A_path = config A (both off), B_path = config C (P2 only)

Outputs:
  - Console: transition matrix (TP→FP, FP→TP, etc.) and regression breakdown
  - delta_*/regressions/*.jpg — rendered images of the worst regression frames
  - Console: Phase 1 rejection precision (if rejection log is provided)
"""

import json
import os
import numpy as np
import cv2
from collections import defaultdict, Counter


# ============================================================
# Config — edit these
# ============================================================
COCO_PATH = "/home/lakshay/lx/ball_detection_tracking/train/_annotations.coco.json"
ACTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_actions.json"
IMAGES_DIR = "/home/lakshay/lx/ball_detection_tracking/train"

# Two runs to compare. Set the JSON paths from your ablation runs.
RUN_A_NAME = "baseline"          # config A — both phases OFF
RUN_A_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v4_A.json"

RUN_B_NAME = "phase2_only"       # config B — Phase 1 ON, Phase 2 OFF
RUN_B_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v4_C.json"

# Optional: Phase 1 rejection log from the run B tracker (set to None if not available)
PHASE1_REJECTION_LOG = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v4_C_phase1_rejections.json"

OUTPUT_DIR = f"delta_{RUN_A_NAME}_vs_{RUN_B_NAME}"
MATCH_RADIUS_PX = 50
RENDER_WORST_N = 400


# ============================================================
# Loaders
# ============================================================
def load_coco_gt(path):
    with open(path) as f:
        coco = json.load(f)
    ball_cat_id = next(c["id"] for c in coco["categories"] if "ball" in c["name"].lower())
    images_sorted = sorted(coco["images"], key=lambda im: im["file_name"])
    image_to_frame = {img["id"]: idx for idx, img in enumerate(images_sorted)}
    frame_to_filename = {idx: img["file_name"] for idx, img in enumerate(images_sorted)}

    gt = {}
    for ann in coco["annotations"]:
        try:
            if int(ann["category_id"]) != int(ball_cat_id):
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
        except (ValueError, TypeError, KeyError):
            continue
        f = image_to_frame.get(ann["image_id"])
        if f is None:
            continue
        cx, cy = x + w/2, y + h/2
        if f in gt and w*h <= gt[f][4]:
            continue
        gt[f] = (cx, cy, w, h, w*h)
    return ({f: (cx, cy, w, h) for f, (cx, cy, w, h, _) in gt.items()},
            frame_to_filename)


def load_detections(path):
    with open(path) as f:
        data = json.load(f)
    det = {}
    for entry in data:
        x, y = entry.get("x"), entry.get("y")
        if x is not None and y is not None:
            det[entry["frame_idx"]] = (float(x), float(y),
                                       float(entry.get("confidence") or 0))
    return det


def load_phase1_rejections(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================
# Action regime lookup
# ============================================================
DURATIONS = {
    "High Pass": 60, "Cross": 50, "Shot": 40, "Free Kick": 60,
    "Goal": 40, "Header": 20, "Throw In": 40,
    "Pass": 30, "Drive": 30,
    "Ball Player Block": 15, "Player Successful Tackle": 15, "Out": 999,
}


def load_actions(path):
    with open(path) as f:
        return sorted(json.load(f).get("events", []), key=lambda e: e["frame"])


def regime_at(frame_idx, events):
    active = None
    for evt in events:
        if evt["frame"] > frame_idx:
            break
        dur = DURATIONS.get(evt["action"], 0)
        if frame_idx < evt["frame"] + dur:
            active = evt["action"]
    return active if active else "no_regime"


def classify(gt_xy, det_xyc):
    """Returns 'TP', 'FP', 'Miss', or 'TN'."""
    if gt_xy is None and det_xyc is None:
        return "TN"
    if gt_xy is None and det_xyc is not None:
        return "FP"
    if gt_xy is not None and det_xyc is None:
        return "Miss"
    dist = np.hypot(gt_xy[0] - det_xyc[0], gt_xy[1] - det_xyc[1])
    return "TP" if dist < MATCH_RADIUS_PX else "FP"


# ============================================================
# Phase 1 rejection quality analysis
# ============================================================
def analyze_phase1_rejections(rejections, gt):
    print("\n" + "=" * 70)
    print("PHASE 1 REJECTION QUALITY")
    print("=" * 70)

    if not rejections:
        print("No rejection log loaded — skipping this section.")
        print("(Add the rejection logging to your tracker and re-run config B to enable.)")
        return

    correct = 0      # rejected when GT was far away → good
    wrong = 0        # rejected when GT was right there → bad
    no_gt = 0        # no GT on this frame
    near_player = 0  # GT near rejected position but Phase 1 still acted

    print(f"\nAnalyzing {len(rejections)} Phase 1 rejections...\n")
    print(f"  {'frame':>5}  {'regime':>10}  {'rejected pos':>15}  {'GT pos':>15}  {'dist':>6}  {'verdict'}")

    for r in rejections:
        f = r["frame"]
        rx, ry = r["rejected_x"], r["rejected_y"]
        g = gt.get(f)

        if g is None:
            no_gt += 1
            verdict = "NO_GT (unclear)"
            gt_str = "n/a"
            dist_str = "n/a"
        else:
            gx, gy, gw, gh = g
            dist = np.hypot(rx - gx, ry - gy)
            gt_str = f"({gx:.0f},{gy:.0f})"
            dist_str = f"{dist:.0f}px"
            if dist < MATCH_RADIUS_PX:
                wrong += 1
                verdict = "BAD (real ball was here)"
            else:
                correct += 1
                verdict = "GOOD (was a wrong-object)"

        print(f"  {f:>5}  {r['regime']:>10}  ({rx:>5.0f},{ry:>5.0f})  "
              f"{gt_str:>15}  {dist_str:>6}  {verdict}")

    total_with_gt = correct + wrong
    print(f"\n  Total rejections:          {len(rejections)}")
    print(f"  Correct (rejected FP):     {correct}")
    print(f"  Wrong (rejected real ball): {wrong}")
    print(f"  No GT on frame (unclear):  {no_gt}")
    if total_with_gt > 0:
        precision = correct / total_with_gt * 100
        print(f"\n  Rejection precision:       {precision:.1f}% "
              f"({correct}/{total_with_gt} correct, ignoring no-GT)")
        if precision >= 85:
            print("  → Phase 1 is fundamentally sound. Issue is recovery after rejection.")
        elif precision >= 60:
            print("  → Phase 1 is rejecting real balls too often. Add depth check.")
        else:
            print("  → Phase 1 rule is wrong-headed. Real balls live inside player bboxes too often.")


# ============================================================
# Main delta analysis
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gt, frame_to_filename = load_coco_gt(COCO_PATH)
    det_a = load_detections(RUN_A_PATH)
    det_b = load_detections(RUN_B_PATH)
    events = load_actions(ACTIONS_PATH)
    rejections = load_phase1_rejections(PHASE1_REJECTION_LOG)

    all_frames = set(gt.keys()) | set(det_a.keys()) | set(det_b.keys())
    n_frames = max(all_frames) + 1 if all_frames else 0

    print(f"Comparing {RUN_A_NAME} vs {RUN_B_NAME}")
    print(f"Frames: {n_frames}, GT: {len(gt)}, A: {len(det_a)}, B: {len(det_b)}\n")
    if rejections is not None:
        print(f"Phase 1 rejections logged: {len(rejections)}\n")

    # Per-frame outcome and disagreement
    deltas = []  # (frame, regime, outcome_a, outcome_b, det_a, det_b, gt_full)
    transition_counts = Counter()
    summary_a = Counter()
    summary_b = Counter()

    for f in range(n_frames):
        g_full = gt.get(f)
        g = (g_full[0], g_full[1]) if g_full else None
        a = det_a.get(f)
        b = det_b.get(f)
        regime = regime_at(f, events)

        oa = classify(g, a)
        ob = classify(g, b)
        summary_a[oa] += 1
        summary_b[ob] += 1

        if oa != ob:
            deltas.append((f, regime, oa, ob, a, b, g_full))
            transition_counts[(oa, ob)] += 1

    print("=" * 70)
    print("OVERALL OUTCOME COUNTS")
    print("=" * 70)
    print(f"  {'outcome':>8}  {RUN_A_NAME:>15}  {RUN_B_NAME:>15}  {'delta':>7}")
    for outcome in ["TP", "FP", "Miss", "TN"]:
        a_count = summary_a[outcome]
        b_count = summary_b[outcome]
        print(f"  {outcome:>8}  {a_count:>15}  {b_count:>15}  {b_count - a_count:>+7}")

    print("\n" + "=" * 70)
    print(f"OUTCOME TRANSITIONS ({RUN_A_NAME} → {RUN_B_NAME})")
    print("=" * 70)
    print(f"{'A → B':<20} {'count':>6}  meaning")
    transition_meanings = [
        (('TP', 'FP'),   "Phase changed correct → wrong (REGRESSION)"),
        (('TP', 'Miss'), "Phase rejected real ball (REGRESSION)"),
        (('FP', 'TP'),   "Phase fixed a wrong detection (WIN)"),
        (('FP', 'Miss'), "Phase rejected wrong detection (WIN)"),
        (('Miss', 'TP'), "Phase recovered missed ball (WIN)"),
        (('Miss', 'FP'), "Phase invented wrong detection (REGRESSION)"),
        (('TN', 'FP'),   "Phase invented detection in non-GT frame (REGRESSION)"),
        (('FP', 'TN'),   "Phase removed FP in non-GT frame (WIN)"),
    ]
    for (oa, ob), meaning in transition_meanings:
        count = transition_counts[(oa, ob)]
        if count > 0:
            print(f"  {oa:>4} → {ob:<5}    {count:>6}  {meaning}")

    win_keys = {('FP', 'TP'), ('FP', 'Miss'), ('Miss', 'TP'), ('FP', 'TN')}
    loss_keys = {('TP', 'FP'), ('TP', 'Miss'), ('Miss', 'FP'), ('TN', 'FP')}
    wins = sum(transition_counts[k] for k in win_keys)
    losses = sum(transition_counts[k] for k in loss_keys)
    print(f"\n  Total wins: {wins}, total losses: {losses}, net: {wins - losses:+d}")

    # Regressions broken down by regime
    print("\n" + "=" * 70)
    print(f"REGRESSIONS BY REGIME (where {RUN_B_NAME} made things worse)")
    print("=" * 70)
    by_regime_regression = defaultdict(Counter)
    for f, regime, oa, ob, a, b, g_full in deltas:
        if (oa, ob) in loss_keys:
            by_regime_regression[regime][(oa, ob)] += 1
    for regime in sorted(by_regime_regression.keys(),
                         key=lambda r: -sum(by_regime_regression[r].values())):
        total = sum(by_regime_regression[regime].values())
        breakdown = ", ".join(f"{oa}→{ob}: {c}"
                              for (oa, ob), c in by_regime_regression[regime].most_common())
        print(f"  {regime:25}  total {total:>3}   ({breakdown})")

    # Detailed list of regressions
    regressions = [(f, r, oa, ob, a, b, g)
                   for f, r, oa, ob, a, b, g in deltas
                   if (oa, ob) in loss_keys]

    print("\n" + "=" * 70)
    print(f"FIRST 20 REGRESSIONS (detailed)")
    print("=" * 70)
    for f, regime, oa, ob, a, b, g_full in regressions[:20]:
        print(f"\n  frame {f}, regime={regime}, {oa} → {ob}")
        if g_full:
            print(f"    GT at ({g_full[0]:.0f}, {g_full[1]:.0f}), bbox {g_full[2]:.0f}x{g_full[3]:.0f}")
        if a:
            dist = np.hypot(a[0]-g_full[0], a[1]-g_full[1]) if g_full else 0
            print(f"    A det at ({a[0]:.0f}, {a[1]:.0f}), conf={a[2]:.2f}, dist_to_gt={dist:.0f}")
        else:
            print(f"    A: no detection")
        if b:
            dist = np.hypot(b[0]-g_full[0], b[1]-g_full[1]) if g_full else 0
            print(f"    B det at ({b[0]:.0f}, {b[1]:.0f}), conf={b[2]:.2f}, dist_to_gt={dist:.0f}")
        else:
            print(f"    B: no detection")

    # Render worst regressions
    print(f"\n\nRendering top {RENDER_WORST_N} regressions to {OUTPUT_DIR}/regressions/")
    bad_dir = os.path.join(OUTPUT_DIR, "regressions")
    os.makedirs(bad_dir, exist_ok=True)
    rendered = 0
    for rank, (f, regime, oa, ob, a, b, g_full) in enumerate(regressions[:RENDER_WORST_N], 1):
        fname = frame_to_filename.get(f)
        if not fname:
            continue
        img_path = os.path.join(IMAGES_DIR, fname)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        # GT bbox in green
        if g_full:
            gx, gy, gw, gh = g_full
            cv2.rectangle(img, (int(gx-gw/2), int(gy-gh/2)),
                          (int(gx+gw/2), int(gy+gh/2)), (0, 255, 0), 2)
            cv2.putText(img, "GT", (int(gx-gw/2), max(int(gy-gh/2)-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # A detection in blue (the "good" run)
        if a:
            cv2.circle(img, (int(a[0]), int(a[1])), 12, (255, 100, 0), 2)
            cv2.putText(img, f"A ({oa})", (int(a[0])+15, int(a[1])-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        # B detection in red (the "bad" run)
        if b:
            cv2.circle(img, (int(b[0]), int(b[1])), 8, (0, 0, 255), 2)
            cv2.putText(img, f"B ({ob})", (int(b[0])+10, int(b[1])+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Header
        header = f"frame {f}  {regime}  {oa} -> {ob}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, header, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = os.path.join(bad_dir, f"rank{rank:02d}_frame{f:05d}_{oa}to{ob}.jpg")
        cv2.imwrite(out_path, img)
        rendered += 1

    print(f"Rendered {rendered} regression frames")

    # Phase 1 rejection quality (if log provided)
    analyze_phase1_rejections(rejections, gt)

    print(f"\n\n=== Done ===")
    print(f"Outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()