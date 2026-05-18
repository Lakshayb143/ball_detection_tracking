"""
Analyze color distributions in GT ball bounding boxes for clip1.
Load all GT boxes, sample pixels, and report HSV statistics.
"""

import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GT_COCO = REPO_ROOT / "train/_annotations.coco.json"
VIDEO = REPO_ROOT / "clips/clip1.mp4"
BALL_CAT_ID = 1


def load_gt(gt_coco, ball_cat_id):
    """Returns {image_id: [(cx, cy, [x,y,w,h]), ...]}"""
    with open(gt_coco) as f:
        coco = json.load(f)
    gt = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] == ball_cat_id:
            gx, gy, gw, gh = (float(v) for v in ann["bbox"])
            gt[ann["image_id"]].append((gx + gw / 2, gy + gh / 2, ann["bbox"]))
    return gt


def analyze_bbox_colors(frame, bbox, label=""):
    """Extract pixels from bbox and report HSV statistics."""
    gx, gy, gw, gh = (float(v) for v in bbox)
    x1, y1 = int(gx), int(gy)
    x2, y2 = int(gx + gw), int(gy + gh)

    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    h_channel = hsv[:, :, 0].astype(float)
    s_channel = hsv[:, :, 1].astype(float)
    v_channel = hsv[:, :, 2].astype(float)

    # Count white pixels (low saturation, high value)
    white_mask = (s_channel < 40) & (v_channel > 150)
    white_frac = float(np.count_nonzero(white_mask)) / float(white_mask.size)

    # Count green pixels
    green_mask = (h_channel >= 35) & (h_channel <= 85) & (s_channel >= 40) & (v_channel >= 40) & (v_channel <= 200)
    green_frac = float(np.count_nonzero(green_mask)) / float(green_mask.size)

    # Count high saturation pixels (blue jerseys, yellow grass)
    high_sat_mask = s_channel > 100
    high_sat_frac = float(np.count_nonzero(high_sat_mask)) / float(high_sat_mask.size)

    return {
        "bbox_size": (x2 - x1, y2 - y1),
        "white_fraction": white_frac,
        "green_fraction": green_frac,
        "high_saturation_fraction": high_sat_frac,
        "avg_saturation": float(np.mean(s_channel)),
        "avg_value": float(np.mean(v_channel)),
    }


def main():
    print("Loading GT annotations…")
    gt_by_imgid = load_gt(GT_COCO, BALL_CAT_ID)
    print(f"Found {len(gt_by_imgid)} frames with GT annotations")

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")

    frame_idx = 1
    all_stats = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in gt_by_imgid:
            for cx, cy, bbox in gt_by_imgid[frame_idx]:
                stats = analyze_bbox_colors(frame, bbox)
                if stats:
                    stats["frame_idx"] = frame_idx
                    stats["gt_cx"] = cx
                    stats["gt_cy"] = cy
                    all_stats.append(stats)

        frame_idx += 1

    cap.release()

    if not all_stats:
        print("No GT boxes found!")
        return

    # Report statistics
    print(f"\n=== GT Ball Box Color Analysis (n={len(all_stats)}) ===\n")

    white_fracs = [s["white_fraction"] for s in all_stats]
    green_fracs = [s["green_fraction"] for s in all_stats]
    high_sat_fracs = [s["high_saturation_fraction"] for s in all_stats]
    avg_sats = [s["avg_saturation"] for s in all_stats]

    print(f"White pixels (S<40, V>150):")
    print(f"  Mean: {np.mean(white_fracs):.1%}")
    print(f"  Min:  {np.min(white_fracs):.1%}")
    print(f"  Max:  {np.max(white_fracs):.1%}")
    print(f"  Median: {np.median(white_fracs):.1%}")

    print(f"\nGreen pixels (H 35-85):")
    print(f"  Mean: {np.mean(green_fracs):.1%}")
    print(f"  Min:  {np.min(green_fracs):.1%}")
    print(f"  Max:  {np.max(green_fracs):.1%}")

    print(f"\nHigh saturation pixels (S>100, i.e. blue/yellow/green):")
    print(f"  Mean: {np.mean(high_sat_fracs):.1%}")
    print(f"  Min:  {np.min(high_sat_fracs):.1%}")
    print(f"  Max:  {np.max(high_sat_fracs):.1%}")

    print(f"\nAverage saturation across all pixels:")
    print(f"  Mean: {np.mean(avg_sats):.1f}")
    print(f"  Min:  {np.min(avg_sats):.1f}")
    print(f"  Max:  {np.max(avg_sats):.1f}")

    # Percentiles
    print(f"\nWhite fraction percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th: {np.percentile(white_fracs, p):.1%}")

    print(f"\nAverage saturation percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th: {np.percentile(avg_sats, p):.1f}")

    # Show some examples
    print(f"\nExamples (worst 5 white fractions):")
    sorted_stats = sorted(all_stats, key=lambda s: s["white_fraction"])
    for i, s in enumerate(sorted_stats[:5]):
        print(
            f"  Frame {s['frame_idx']:04d}: white={s['white_fraction']:.1%}, "
            f"green={s['green_fraction']:.1%}, high_sat={s['high_saturation_fraction']:.1%}, "
            f"avg_sat={s['avg_saturation']:.1f}"
        )


if __name__ == "__main__":
    main()
