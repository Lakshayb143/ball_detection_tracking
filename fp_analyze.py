"""
Analyze why the action filter isn't helping.

For every frame, classifies:
  - GT presence and y-position
  - Detection presence and y-position
  - Active action regime
  - Frame outcome (TP/FP/Miss/TN)
  - Whether the action filter would have rejected the detection (and whether that was correct)
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# Config
# ============================================================
COCO_PATH = "/home/lakshay/lx/ball_detection_tracking/train/_annotations.coco.json"
DETECTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v3.json"
ACTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_actions.json"
FRAME_HEIGHT = 1080   # adjust if your video is different
GROUND_FRACTION = 0.7
MATCH_RADIUS_PX = 50  # detection within this distance of GT counts as TP


# ============================================================
# Load everything
# ============================================================
def load_coco_gt(path):
    with open(path) as f:
        coco = json.load(f)
    ball_cat_id = next((c["id"] for c in coco["categories"]
                        if "ball" in c["name"].lower()), None)
    images_sorted = sorted(coco["images"], key=lambda im: im["file_name"])
    image_to_frame = {img["id"]: idx for idx, img in enumerate(images_sorted)}

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
        if f in gt and w*h <= gt[f][2]:
            continue
        gt[f] = (cx, cy, w*h)
    return {f: (cx, cy) for f, (cx, cy, _) in gt.items()}


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


# Action regime durations (must match your tracker's config)
DURATIONS = {
    "High Pass": 60, "Cross": 50, "Shot": 40, "Free Kick": 60,
    "Goal": 40, "Header": 20, "Throw In": 40,
    "Pass": 30, "Drive": 30,
    "Ball Player Block": 15, "Player Successful Tackle": 15,
    "Out": 999,
}
AIRBORNE_WITH_GROUND_REJECT = {"High Pass", "Cross", "Shot", "Free Kick"}
AIRBORNE_OTHER = {"Header", "Throw In", "Goal"}


def load_actions(path):
    with open(path) as f:
        return sorted(json.load(f).get("events", []), key=lambda e: e["frame"])


def regime_at(frame_idx, events):
    """Returns (action_name, frames_since) for the active regime, or (None, 0)."""
    active = None
    for evt in events:
        if evt["frame"] > frame_idx:
            break
        dur = DURATIONS.get(evt["action"], 0)
        if frame_idx < evt["frame"] + dur:
            active = (evt["action"], frame_idx - evt["frame"])
    return active if active else (None, 0)


# ============================================================
# Analyze
# ============================================================
def main():
    gt = load_coco_gt(COCO_PATH)
    det = load_detections(DETECTIONS_PATH)
    events = load_actions(ACTIONS_PATH)

    n_frames = max(max(gt.keys(), default=0), max(det.keys(), default=0)) + 1
    ground_y = FRAME_HEIGHT * GROUND_FRACTION

    print(f"Frames analyzed: {n_frames}")
    print(f"GT present on:   {len(gt)}")
    print(f"Det present on:  {len(det)}")
    print(f"Action events:   {len(events)}")
    print(f"Ground threshold y > {ground_y:.0f}\n")

    # Per-frame analysis
    by_regime = defaultdict(lambda: {"TP": 0, "FP": 0, "Miss": 0, "TN": 0})
    fp_breakdown = Counter()
    miss_breakdown = Counter()
    action_filter_would_reject = []   # list of (frame, was_TP, regime)

    for f in range(n_frames):
        g = gt.get(f)
        d = det.get(f)
        regime, frames_since = regime_at(f, events)
        regime_key = regime if regime else "no_regime"

        # Classify outcome
        if g and d:
            dist = np.hypot(g[0]-d[0], g[1]-d[1])
            outcome = "TP" if dist < MATCH_RADIUS_PX else "FP"
        elif d and not g:
            outcome = "FP"
        elif g and not d:
            outcome = "Miss"
        else:
            outcome = "TN"

        by_regime[regime_key][outcome] += 1

        # Classify FPs
        if outcome == "FP" and d:
            in_ground_zone = d[1] > ground_y
            if regime in AIRBORNE_WITH_GROUND_REJECT and in_ground_zone:
                fp_breakdown["airborne_regime + ground_zone (action filter SHOULD catch)"] += 1
            elif regime in AIRBORNE_WITH_GROUND_REJECT and not in_ground_zone:
                fp_breakdown["airborne_regime + sky_zone (action filter would NOT catch)"] += 1
            elif regime in AIRBORNE_OTHER:
                fp_breakdown[f"airborne_other ({regime})"] += 1
            elif regime is None:
                fp_breakdown["no_regime"] += 1
            else:
                fp_breakdown[f"other_regime ({regime})"] += 1

        # Classify Misses
        if outcome == "Miss" and g:
            in_ground_zone = g[1] > ground_y
            zone = "low_in_frame" if in_ground_zone else "high_in_frame"
            r = regime if regime else "no_regime"
            miss_breakdown[f"{r} / {zone}"] += 1

        # Would action filter reject this detection?
        if d and regime in AIRBORNE_WITH_GROUND_REJECT and d[1] > ground_y:
            action_filter_would_reject.append((f, outcome == "TP", regime))

    # ---- Print results ----
    print("=" * 70)
    print("OUTCOME BREAKDOWN BY REGIME")
    print("=" * 70)
    for regime in ["no_regime"] + [e["action"] for e in events]:
        if regime in by_regime:
            r = by_regime[regime]
            total = sum(r.values())
            print(f"\n{regime:25} (total {total} frames)")
            print(f"   TP={r['TP']}  FP={r['FP']}  Miss={r['Miss']}  TN={r['TN']}")

    print("\n" + "=" * 70)
    print("FP BREAKDOWN — where are FPs happening?")
    print("=" * 70)
    for cat, count in fp_breakdown.most_common():
        print(f"  {count:>4}  {cat}")

    print("\n" + "=" * 70)
    print("MISS BREAKDOWN — where are misses happening?")
    print("=" * 70)
    for cat, count in miss_breakdown.most_common():
        print(f"  {count:>4}  {cat}")

    print("\n" + "=" * 70)
    print("ACTION FILTER IMPACT")
    print("=" * 70)
    catches_fp = sum(1 for _, was_tp, _ in action_filter_would_reject if not was_tp)
    kills_tp = sum(1 for _, was_tp, _ in action_filter_would_reject if was_tp)
    print(f"  Detections in airborne+ground zone: {len(action_filter_would_reject)}")
    print(f"    - real FPs caught (good):         {catches_fp}")
    print(f"    - real TPs killed (bad):          {kills_tp}")
    print(f"  Net effect: {catches_fp - kills_tp:+d} FPs")

    if kills_tp > 0:
        print(f"\n  TPs that would be killed (consider raising GROUND_FRACTION):")
        for f, was_tp, r in action_filter_would_reject[:10]:
            if was_tp:
                d = det[f]
                print(f"    frame {f}: regime={r}, det_y={d[1]:.0f} (ground_y={ground_y:.0f})")


if __name__ == "__main__":
    main()