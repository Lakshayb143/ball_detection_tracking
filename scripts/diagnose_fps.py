"""
FP Diagnostic Script — ground tracking false positives.

For each FP frame in a best-run output, reports:
  - Whether the prediction was interpolated
  - Distance from pred to GT ball
  - Whether the GT ball was stationary (possession indicator)
  - KF velocity at the start of the gap (estimated from last 2 accepted frames)

Optionally saves annotated images for every FP frame.

Usage:
  python scripts/diagnose_fps.py
  python scripts/diagnose_fps.py --save-images --video clips/clip1.mp4
  python scripts/diagnose_fps.py --run-dir clip1_fresh_runs/v4_then_ransac_v2_online10__clip1 \
      --tracker-json clip1_fresh_runs/ball_outlier_interpolator_v4__clip1/point_outputs.json \
      --gt-coco train/_annotations.coco.json \
      --output-csv outputs/fp_diagnosis_clip1.csv \
      --save-images --video clips/clip1.mp4 --images-dir outputs/fp_frames_clip1
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# Defaults (all relative to repo root)
# ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = dict(
    run_dir=REPO_ROOT
    / "clip1_fresh_runs/v4_then_ransac_v2_online10__clip1",
    tracker_json=REPO_ROOT
    / "clip1_fresh_runs/ball_outlier_interpolator_v4__clip1/point_outputs.json",
    gt_coco=REPO_ROOT / "train/_annotations.coco.json",
    output_csv=REPO_ROOT / "outputs/fp_diagnosis_clip1.csv",
    images_dir=REPO_ROOT / "outputs/fp_frames_clip1",
    video=REPO_ROOT / "clips/clip1.mp4",
    box_size=20.0,
    match_iou=0.01,
    ball_category_id=1,
    # GT "stationary" threshold — if GT centroid moves less than this across
    # an FP run, we flag it as a possession candidate.
    gt_stationary_px=30.0,
)


# ──────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────

def _iou(pred_cx, pred_cy, gt_bbox_xywh, box_size):
    px1 = pred_cx - box_size / 2
    py1 = pred_cy - box_size / 2
    px2 = pred_cx + box_size / 2
    py2 = pred_cy + box_size / 2
    gx, gy, gw, gh = (float(v) for v in gt_bbox_xywh)
    gx2 = gx + gw
    gy2 = gy + gh
    ix1 = max(px1, gx)
    iy1 = max(py1, gy)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_size * box_size + gw * gh - inter
    return inter / union if union > 0 else 0.0


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


# ──────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────

def load_frame_trace(run_dir):
    path = Path(run_dir) / "frame_trace.csv"
    with open(path) as f:
        return list(csv.DictReader(f))


def load_point_outputs(tracker_json):
    with open(tracker_json) as f:
        pts = json.load(f)
    return {p["frame_idx"]: p for p in pts}


def load_gt(gt_coco, ball_category_id):
    """Returns {image_id: [(cx, cy, bbox_xywh), ...]}"""
    with open(gt_coco) as f:
        coco = json.load(f)
    gt = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] == ball_category_id:
            gx, gy, gw, gh = (float(v) for v in ann["bbox"])
            gt[ann["image_id"]].append((gx + gw / 2, gy + gh / 2, ann["bbox"]))
    return gt


# ──────────────────────────────────────────────────────────────────
# FP identification
# ──────────────────────────────────────────────────────────────────

def find_fp_frames(frame_trace, gt_by_imgid, box_size, match_iou):
    """
    Returns list of dicts, one per FP frame:
      frame_idx, pred_x, pred_y, gt_cx, gt_cy, dist_px,
      is_interpolated, final_source
    """
    fp_frames = []
    for row in frame_trace:
        if row["final_source"] == "no_output":
            continue
        img_id = int(row["image_id"])
        gts = gt_by_imgid.get(img_id)
        if not gts:
            continue  # no GT ball — not a GT-frame FP
        fx = float(row["final_x"])
        fy = float(row["final_y"])
        max_iou = max(_iou(fx, fy, bbox, box_size) for _, _, bbox in gts)
        if max_iou >= match_iou:
            continue  # TP
        # Nearest GT box
        best_gt = min(gts, key=lambda g: _dist(fx, fy, g[0], g[1]))
        gt_cx, gt_cy, gt_bbox = best_gt
        fp_frames.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "pred_x": fx,
                "pred_y": fy,
                "gt_cx": gt_cx,
                "gt_cy": gt_cy,
                "gt_bbox": gt_bbox,
                "dist_px": _dist(fx, fy, gt_cx, gt_cy),
                "is_interpolated": row["raw_interpolated"] == "True",
                "final_source": row["final_source"],
            }
        )
    return fp_frames


# ──────────────────────────────────────────────────────────────────
# Run grouping
# ──────────────────────────────────────────────────────────────────

def group_into_runs(fp_frames):
    """Groups consecutive FP frames (no gaps) into runs."""
    if not fp_frames:
        return []
    runs = []
    current = [fp_frames[0]]
    for fp in fp_frames[1:]:
        if fp["frame_idx"] == current[-1]["frame_idx"] + 1:
            current.append(fp)
        else:
            runs.append(current)
            current = [fp]
    runs.append(current)
    return runs


# ──────────────────────────────────────────────────────────────────
# Velocity at gap start
# ──────────────────────────────────────────────────────────────────

def estimate_velocity_at_gap_start(run_start_frame, point_outputs, lookback=3):
    """
    Finds the last N accepted (non-interpolated) detections before
    run_start_frame and returns (vx, vy, speed_px_per_frame).
    Returns (None, None, None) if not enough history.
    """
    frames_before = [
        p
        for f, p in point_outputs.items()
        if f < run_start_frame
        and not p.get("interpolated", True)
        and p.get("x") is not None
        and p.get("y") is not None
    ]
    if len(frames_before) < 2:
        return None, None, None
    frames_before.sort(key=lambda p: p["frame_idx"])
    recent = frames_before[-lookback:]
    if len(recent) < 2:
        return None, None, None
    f0, f1 = recent[-2], recent[-1]
    dt = f1["frame_idx"] - f0["frame_idx"]
    if dt == 0:
        return None, None, None
    vx = (f1["x"] - f0["x"]) / dt
    vy = (f1["y"] - f0["y"]) / dt
    return vx, vy, math.hypot(vx, vy)


# ──────────────────────────────────────────────────────────────────
# GT motion during a run
# ──────────────────────────────────────────────────────────────────

def gt_range_during_run(run_frames):
    """Max displacement of GT centroid across a run."""
    gt_positions = [(f["gt_cx"], f["gt_cy"]) for f in run_frames]
    if len(gt_positions) < 2:
        return 0.0
    xs = [p[0] for p in gt_positions]
    ys = [p[1] for p in gt_positions]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    return math.hypot(dx, dy)


# ──────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────

POSSESSION_FLAG = "POSSESSION?"
DRIFT_FLAG = "DRIFT"


def classify_run(run, vx, vy, speed, gt_range, gt_stationary_px):
    if gt_range < gt_stationary_px:
        return POSSESSION_FLAG
    return DRIFT_FLAG


def print_report(runs, point_outputs, gt_stationary_px):
    total_fp = sum(len(r) for r in runs)
    print(f"\n{'='*70}")
    print(f"  FP DIAGNOSTIC REPORT")
    print(f"  Total FP frames: {total_fp}  |  Runs: {len(runs)}")
    print(f"{'='*70}\n")

    for i, run in enumerate(runs):
        start_f = run[0]["frame_idx"]
        end_f = run[-1]["frame_idx"]
        n = len(run)
        n_interp = sum(1 for f in run if f["is_interpolated"])
        dist_start = run[0]["dist_px"]
        dist_end = run[-1]["dist_px"]
        gt_range = gt_range_during_run(run)
        vx, vy, speed = estimate_velocity_at_gap_start(start_f, point_outputs)

        classification = classify_run(run, vx, vy, speed, gt_range, gt_stationary_px)
        speed_str = f"{speed:.1f} px/frame" if speed is not None else "n/a"
        vel_str = (
            f"({vx:+.1f}, {vy:+.1f})" if vx is not None else "n/a"
        )

        print(f"  Run #{i+1}  frames {start_f}–{end_f}  [{n} frames]  ── {classification}")
        print(f"    Interpolated:   {n_interp}/{n} frames")
        print(f"    Distance:       {dist_start:.0f} px (start) → {dist_end:.0f} px (end)")
        print(f"    GT movement:    {gt_range:.0f} px across run")
        print(f"    Speed@gap-start: {speed_str}  vel={vel_str}")

        # Per-frame table (compact)
        print(f"    {'frame':>6}  {'pred_x':>7}  {'pred_y':>7}  {'gt_cx':>7}  {'gt_cy':>7}  {'dist':>6}  {'interp':>6}")
        for fp in run:
            interp_mark = "Y" if fp["is_interpolated"] else "N"
            print(
                f"    {fp['frame_idx']:>6}  "
                f"{fp['pred_x']:>7.1f}  {fp['pred_y']:>7.1f}  "
                f"{fp['gt_cx']:>7.1f}  {fp['gt_cy']:>7.1f}  "
                f"{fp['dist_px']:>6.0f}  {interp_mark:>6}"
            )
        print()


def write_csv(runs, point_outputs, gt_stationary_px, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "classification",
        "frame_idx",
        "pred_x",
        "pred_y",
        "gt_cx",
        "gt_cy",
        "dist_px",
        "is_interpolated",
        "final_source",
        "run_start",
        "run_end",
        "run_n_frames",
        "run_gt_movement_px",
        "speed_at_gap_start_px_frame",
        "vx_at_gap_start",
        "vy_at_gap_start",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, run in enumerate(runs):
            gt_range = gt_range_during_run(run)
            vx, vy, speed = estimate_velocity_at_gap_start(
                run[0]["frame_idx"], point_outputs
            )
            classification = classify_run(
                run, vx, vy, speed, gt_range, gt_stationary_px
            )
            for fp in run:
                writer.writerow(
                    {
                        "run_id": i + 1,
                        "classification": classification,
                        "frame_idx": fp["frame_idx"],
                        "pred_x": round(fp["pred_x"], 2),
                        "pred_y": round(fp["pred_y"], 2),
                        "gt_cx": round(fp["gt_cx"], 2),
                        "gt_cy": round(fp["gt_cy"], 2),
                        "dist_px": round(fp["dist_px"], 1),
                        "is_interpolated": fp["is_interpolated"],
                        "final_source": fp["final_source"],
                        "run_start": run[0]["frame_idx"],
                        "run_end": run[-1]["frame_idx"],
                        "run_n_frames": len(run),
                        "run_gt_movement_px": round(gt_range, 1),
                        "speed_at_gap_start_px_frame": (
                            round(speed, 2) if speed is not None else ""
                        ),
                        "vx_at_gap_start": (
                            round(vx, 2) if vx is not None else ""
                        ),
                        "vy_at_gap_start": (
                            round(vy, 2) if vy is not None else ""
                        ),
                    }
                )
    print(f"CSV written → {out_path}")


# ──────────────────────────────────────────────────────────────────
# Summary stats
# ──────────────────────────────────────────────────────────────────

def print_summary(runs, gt_stationary_px, point_outputs):
    possession_runs = []
    drift_runs = []
    for run in runs:
        gt_range = gt_range_during_run(run)
        vx, vy, speed = estimate_velocity_at_gap_start(run[0]["frame_idx"], point_outputs)
        cls = classify_run(run, vx, vy, speed, gt_range, gt_stationary_px)
        if cls == POSSESSION_FLAG:
            possession_runs.append(run)
        else:
            drift_runs.append(run)

    possession_frames = sum(len(r) for r in possession_runs)
    drift_frames = sum(len(r) for r in drift_runs)
    total_fp_frames = possession_frames + drift_frames

    print(f"\n{'='*70}")
    print(f"  SUMMARY  (GT stationary threshold = {gt_stationary_px:.0f} px)")
    print(f"{'='*70}")
    print(f"  Possession-candidate runs  : {len(possession_runs):3d}  ({possession_frames} frames)")
    print(f"  Free-drift runs            : {len(drift_runs):3d}  ({drift_frames} frames)")
    print(f"  Total FP frames            : {total_fp_frames}")
    print()

    if possession_runs:
        speeds = []
        for run in possession_runs:
            _, _, speed = estimate_velocity_at_gap_start(run[0]["frame_idx"], point_outputs)
            if speed is not None:
                speeds.append(speed)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        print(f"  Avg speed at gap-start (possession runs): {avg_speed:.1f} px/frame")

    if drift_runs:
        speeds = []
        for run in drift_runs:
            _, _, speed = estimate_velocity_at_gap_start(run[0]["frame_idx"], point_outputs)
            if speed is not None:
                speeds.append(speed)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        print(f"  Avg speed at gap-start (drift runs):      {avg_speed:.1f} px/frame")
    print()


# ──────────────────────────────────────────────────────────────────
# Image saving
# ──────────────────────────────────────────────────────────────────

# Colors (BGR)
_GREEN = (50, 205, 50)
_RED   = (0, 60, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_YELLOW = (0, 200, 255)


def _put_text(img, text, x, y, color, scale=0.6, thickness=1):
    import cv2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_box(img, cx, cy, w, h, color, label, thickness=2):
    import cv2
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # Label just above the box
    lx = max(x1, 4)
    ly = max(y1 - 6, 14)
    _put_text(img, label, lx, ly, color, scale=0.55, thickness=1)


def _annotate_fp_frame(frame, fp, run_id, classification, box_size):
    import cv2

    img = frame.copy()
    h_img, w_img = img.shape[:2]

    pred_cx, pred_cy = fp["pred_x"], fp["pred_y"]
    gt_cx, gt_cy = fp["gt_cx"], fp["gt_cy"]
    gt_bbox = fp["gt_bbox"]
    gx, gy, gw, gh = (float(v) for v in gt_bbox)

    # Draw GT box (green)
    _draw_box(img, gt_cx, gt_cy, gw, gh, _GREEN, "GT", thickness=2)

    # Draw predicted box (red) — fixed 20x20
    _draw_box(img, pred_cx, pred_cy, box_size, box_size, _RED, "PRED", thickness=2)

    # Arrow from pred to GT center
    cv2.arrowedLine(
        img,
        (int(pred_cx), int(pred_cy)),
        (int(gt_cx), int(gt_cy)),
        _YELLOW, 1, tipLength=0.15,
    )

    # Distance label at midpoint of arrow
    mid_x = int((pred_cx + gt_cx) / 2)
    mid_y = int((pred_cy + gt_cy) / 2)
    _put_text(img, f"{fp['dist_px']:.0f}px", mid_x + 4, mid_y - 4, _YELLOW, scale=0.5)

    # Header bar at top-left
    interp_str = "INTERP" if fp["is_interpolated"] else "DETECT"
    cls_color = _RED if classification == DRIFT_FLAG else _YELLOW
    lines = [
        (f"Frame {fp['frame_idx']:04d}  Run #{run_id}  {classification}", cls_color),
        (f"{interp_str}  dist={fp['dist_px']:.0f}px", _WHITE),
    ]
    for i, (text, color) in enumerate(lines):
        _put_text(img, text, 10, 28 + i * 26, color, scale=0.65, thickness=1)

    return img


def save_fp_images(runs, video_path, images_dir, box_size, point_outputs, gt_stationary_px):
    import cv2

    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup: frame_idx -> (fp dict, run_id, classification)
    fp_lookup = {}
    for i, run in enumerate(runs):
        gt_range = gt_range_during_run(run)
        vx, vy, speed = estimate_velocity_at_gap_start(run[0]["frame_idx"], point_outputs)
        cls = classify_run(run, vx, vy, speed, gt_range, gt_stationary_px)
        for fp in run:
            fp_lookup[fp["frame_idx"]] = (fp, i + 1, cls)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved = 0
    # Iterate video sequentially (frame_idx is 1-indexed in data, 0-indexed in OpenCV)
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in fp_lookup:
            fp, run_id, cls = fp_lookup[frame_idx]
            annotated = _annotate_fp_frame(frame, fp, run_id, cls, box_size)
            fname = f"frame_{frame_idx:04d}_run{run_id:02d}_{cls.rstrip('?')}.jpg"
            out_path = images_dir / fname
            cv2.imwrite(str(out_path), annotated)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} FP images → {images_dir}/")


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diagnose ground-tracking FPs")
    parser.add_argument("--run-dir", default=str(DEFAULTS["run_dir"]))
    parser.add_argument("--tracker-json", default=str(DEFAULTS["tracker_json"]))
    parser.add_argument("--gt-coco", default=str(DEFAULTS["gt_coco"]))
    parser.add_argument("--output-csv", default=str(DEFAULTS["output_csv"]))
    parser.add_argument("--box-size", type=float, default=DEFAULTS["box_size"])
    parser.add_argument("--match-iou", type=float, default=DEFAULTS["match_iou"])
    parser.add_argument(
        "--ball-category-id", type=int, default=DEFAULTS["ball_category_id"]
    )
    parser.add_argument(
        "--gt-stationary-px",
        type=float,
        default=DEFAULTS["gt_stationary_px"],
        help="Max GT centroid movement (px) across a run to flag as POSSESSION?",
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="Skip writing the CSV output"
    )
    parser.add_argument(
        "--save-images", action="store_true", help="Save annotated FP images"
    )
    parser.add_argument("--video", default=str(DEFAULTS["video"]))
    parser.add_argument("--images-dir", default=str(DEFAULTS["images_dir"]))
    args = parser.parse_args()

    print(f"Loading frame trace from: {args.run_dir}")
    frame_trace = load_frame_trace(args.run_dir)

    print(f"Loading tracker outputs from: {args.tracker_json}")
    point_outputs = load_point_outputs(args.tracker_json)

    print(f"Loading GT annotations from: {args.gt_coco}")
    gt_by_imgid = load_gt(args.gt_coco, args.ball_category_id)

    fp_frames = find_fp_frames(
        frame_trace, gt_by_imgid, args.box_size, args.match_iou
    )
    print(f"Found {len(fp_frames)} FP frames")

    runs = group_into_runs(fp_frames)
    print(f"Grouped into {len(runs)} consecutive runs")

    print_report(runs, point_outputs, args.gt_stationary_px)
    print_summary(runs, args.gt_stationary_px, point_outputs)

    if not args.no_csv:
        write_csv(runs, point_outputs, args.gt_stationary_px, args.output_csv)

    if args.save_images:
        save_fp_images(
            runs, args.video, args.images_dir,
            args.box_size, point_outputs, args.gt_stationary_px,
        )


if __name__ == "__main__":
    main()
