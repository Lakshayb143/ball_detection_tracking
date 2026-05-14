"""
Calibration script — reads ground truth from a COCO annotations JSON.
Optionally also reads detector predictions (for R / measurement noise estimate).
"""

import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from collections import defaultdict
import matplotlib.pyplot as plt


# ============================================================
# Config — EDIT THESE
# ============================================================
COCO_PATH = "/home/lakshay/lx/ball_detection_tracking/train/_annotations.coco.json"
DETECTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output.json"        # Optional: path to detector predictions JSON (see format below)
                              # Set to None if you only want to calibrate gravity/Q/gaps
FPS = 30                      # Your video's fps
BALL_CATEGORY_NAME = "ball"   # Adjust if your category is named differently


# ============================================================
# 1. Load COCO ground truth
# ============================================================
def load_coco_ground_truth(coco_path, ball_category_name=BALL_CATEGORY_NAME):
    """
    Returns:
        gt: dict mapping frame_idx -> (x, y) center of ball bbox
        n_frames: total number of frames inferred from image set
    """
    with open(coco_path, "r") as f:
        coco = json.load(f)

    # Find the ball category id
    ball_cat_id = None
    for cat in coco["categories"]:
        if cat["name"].lower() == ball_category_name.lower():
            ball_cat_id = cat["id"]
            break
    if ball_cat_id is None:
        for cat in coco["categories"]:
            if "ball" in cat["name"].lower():
                ball_cat_id = cat["id"]
                print(f"[coco] Using category '{cat['name']}' (id={ball_cat_id}) as ball")
                break
    if ball_cat_id is None:
        raise ValueError(f"No ball category found. Available: "
                         f"{[c['name'] for c in coco['categories']]}")

    # Use sorted position as frame index — robust against any filename scheme.
    images_sorted = sorted(coco["images"], key=lambda im: im["file_name"])
    image_to_frame = {img["id"]: idx for idx, img in enumerate(images_sorted)}
    n_frames = len(images_sorted)

    print(f"[coco] {n_frames} images in dataset")
    print(f"[coco] first 3 filenames: {[im['file_name'] for im in images_sorted[:3]]}")
    print(f"[coco] last 3 filenames:  {[im['file_name'] for im in images_sorted[-3:]]}")

    # Extract ball annotations
    gt = {}
    skipped = 0
    for ann in coco["annotations"]:
        try:
            cat_id = int(ann["category_id"])
            if cat_id != int(ball_cat_id):
                continue
            img_id = ann["image_id"]
            if img_id not in image_to_frame:
                continue
            bbox = ann["bbox"]
            x = float(bbox[0])
            y = float(bbox[1])
            w = float(bbox[2])
            h = float(bbox[3])
        except (ValueError, TypeError, KeyError, IndexError):
            skipped += 1
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        frame_idx = image_to_frame[img_id]
        new_area = w * h
        if frame_idx in gt:
            existing_area = gt[frame_idx][2]
            if new_area > existing_area:
                gt[frame_idx] = (cx, cy, new_area)
        else:
            gt[frame_idx] = (cx, cy, new_area)

    gt = {f: (x, y) for f, (x, y, _) in gt.items()}
    if skipped > 0:
        print(f"[coco] skipped {skipped} malformed annotations")
    print(f"[coco] loaded {len(gt)} ball annotations")
    return gt, n_frames


# ============================================================
# 2. Optional — load detector predictions
# ============================================================
def load_detections(detections_path):
    """
    Expected format (JSON):
        [
          {"frame_idx": 0, "x": 512.3, "y": 244.1, "confidence": 0.92},
          {"frame_idx": 1, "x": null,  "y": null,  "confidence": null},
          ...
        ]
    Or a COCO-style detection results file.
    """
    if detections_path is None:
        return {}

    with open(detections_path, "r") as f:
        data = json.load(f)

    det = {}
    if isinstance(data, list) and len(data) > 0 and "frame_idx" in data[0]:
        # Simple format
        for entry in data:
            f_idx = entry["frame_idx"]
            x, y = entry.get("x"), entry.get("y")
            if x is not None and y is not None:
                det[f_idx] = (float(x), float(y))
    elif isinstance(data, list) and len(data) > 0 and "image_id" in data[0]:
        # COCO results format
        # You'd need an image_id -> frame_idx mapping; skipping for brevity.
        # If you have COCO-format detections, tell me and I'll extend this.
        raise NotImplementedError("COCO detection results format — tell me and I'll add this.")

    print(f"[det] loaded {len(det)} detections")
    return det


# ============================================================
# 3. Convert dicts to dense arrays for analysis
# ============================================================
def dicts_to_arrays(gt_dict, det_dict, n_frames):
    gt_arr = np.full((n_frames, 2), np.nan)
    det_arr = np.full((n_frames, 2), np.nan)
    for f_idx, (x, y) in gt_dict.items():
        if 0 <= f_idx < n_frames:
            gt_arr[f_idx] = (x, y)
    for f_idx, (x, y) in det_dict.items():
        if 0 <= f_idx < n_frames:
            det_arr[f_idx] = (x, y)
    return gt_arr, det_arr


# ============================================================
# 4. Find airborne segments from ground truth
# ============================================================
def find_airborne_segments(gt, dt, min_length=8, peak_height_px=20):
    """
    Identify segments where the ball is in the air via parabolic y(t).
    Skips frames where gt is NaN by working only on contiguous valid runs.
    """
    segments = []
    valid = ~np.isnan(gt[:, 0])

    # Find contiguous runs of valid frames
    runs = []
    i = 0
    while i < len(valid):
        if valid[i]:
            j = i
            while j < len(valid) and valid[j]:
                j += 1
            if j - i >= min_length:
                runs.append((i, j))
            i = j
        else:
            i += 1

    for run_start, run_end in runs:
        y = gt[run_start:run_end, 1]
        if len(y) < min_length:
            continue

        # Smooth and compute discrete second derivative
        window = 3
        kernel = np.ones(window) / window
        y_smooth = np.convolve(y, kernel, mode="same")
        a_y = np.gradient(np.gradient(y_smooth, dt), dt)

        # Sustained downward acceleration (positive in image coords)
        airborne_mask = a_y > 100

        in_seg = False
        seg_start = None
        for k, is_air in enumerate(airborne_mask):
            if is_air and not in_seg:
                seg_start = k
                in_seg = True
            elif not is_air and in_seg:
                if k - seg_start >= min_length:
                    seg_y = y[seg_start:k]
                    if seg_y.max() - seg_y.min() >= peak_height_px:
                        segments.append((run_start + seg_start, run_start + k))
                in_seg = False
        if in_seg and len(y) - seg_start >= min_length:
            seg_y = y[seg_start:]
            if seg_y.max() - seg_y.min() >= peak_height_px:
                segments.append((run_start + seg_start, run_end))

    return segments


# ============================================================
# 5. Estimate gravity in pixels per second^2
# ============================================================
def estimate_gravity(gt, dt, segments):
    if not segments:
        print("[gravity] No airborne segments found. Returning default 600 px/s².")
        return 600.0, []

    g_estimates = []
    for start, end in segments:
        y = gt[start:end, 1]
        t = np.arange(len(y)) * dt
        try:
            popt, _ = curve_fit(lambda t, y0, vy, g: y0 + vy * t + 0.5 * g * t**2,
                                t, y, p0=[y[0], 0, 600])
            g_fit = popt[2]
            if 100 < g_fit < 3000:
                g_estimates.append(g_fit)
                print(f"[gravity] segment frames {start}-{end} ({end-start} frames): g = {g_fit:.1f} px/s²")
        except Exception as e:
            print(f"[gravity] segment {start}-{end} fit failed: {e}")

    if not g_estimates:
        print("[gravity] No valid fits. Returning default 600.")
        return 600.0, []

    g_median = float(np.median(g_estimates))
    g_mad = float(np.median(np.abs(np.array(g_estimates) - g_median)))
    print(f"\n[gravity] {len(g_estimates)} segments fitted")
    print(f"[gravity] median g = {g_median:.1f} px/s²,  MAD = {g_mad:.1f}")
    return g_median, g_estimates


# ============================================================
# 6. Estimate measurement noise R from detector vs ground truth
# ============================================================
def estimate_measurement_noise(gt, det):
    valid = ~np.isnan(det[:, 0]) & ~np.isnan(gt[:, 0])
    if valid.sum() < 10:
        print("\n[R] Too few overlapping detection+gt frames. Using default R = 9.0 * I.")
        return np.eye(2) * 9.0, None
    residuals = det[valid] - gt[valid]
    R = np.cov(residuals.T)
    rms = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
    print(f"\n[R] {valid.sum()} frames with both gt and detection")
    print(f"[R] residual covariance:\n{R}")
    print(f"[R] RMS detection error: {rms:.2f} px")
    print(f"[R] suggested R = {np.diag(R).mean():.1f} * I")
    return R, rms


# ============================================================
# 7. Estimate process noise Q from ground-truth dynamics
# ============================================================
def estimate_process_noise(gt, dt):
    valid = ~np.isnan(gt[:, 0])
    gt_valid = gt[valid]

    v = np.diff(gt_valid, axis=0) / dt
    a = np.diff(v, axis=0) / dt

    pos_var = 1.0
    vel_var = float(np.var(v))
    accel_var = float(np.var(a))
    accel_var_capped = min(accel_var, 500.0)

    print(f"\n[Q] velocity variance from GT: {vel_var:.1f} (px/s)²")
    print(f"[Q] acceleration variance from GT: {accel_var:.1f} (px/s²)²")
    print(f"[Q] capped accel variance: {accel_var_capped:.1f}")
    print(f"[Q] suggested Q = diag([{pos_var}, {pos_var}, "
          f"{vel_var/100:.1f}, {vel_var/100:.1f}, "
          f"{accel_var_capped/100:.1f}, {accel_var_capped/100:.1f}])")
    return pos_var, vel_var, accel_var_capped


# ============================================================
# 8. Detection gap analysis
# ============================================================
def analyze_gaps(det, n_frames):
    if not np.any(~np.isnan(det[:, 0])):
        print("\n[gaps] No detections provided — cannot estimate MAX_GAP_FRAMES from data. "
              "Defaulting to 30.")
        return 30

    missing = np.isnan(det[:, 0])
    gaps = []
    current = 0
    for m in missing:
        if m:
            current += 1
        else:
            if current > 0:
                gaps.append(current)
            current = 0
    if current > 0:
        gaps.append(current)

    if not gaps:
        print("\n[gaps] No detection gaps in this clip.")
        return 30

    gaps = np.array(gaps)
    print(f"\n[gaps] {len(gaps)} gaps total")
    print(f"[gaps] mean: {gaps.mean():.1f}, median: {np.median(gaps):.0f}, "
          f"95th pct: {np.percentile(gaps, 95):.0f}, max: {gaps.max()}")
    return int(np.percentile(gaps, 95))


# ============================================================
# 9. Ground-truth annotation gap analysis (separate signal)
# ============================================================
def analyze_gt_gaps(gt):
    """
    Even without detector outputs, gaps in the ground truth tell you the
    longest stretches the ball is invisible (occluded) in your data.
    """
    missing = np.isnan(gt[:, 0])
    gaps = []
    current = 0
    for m in missing:
        if m:
            current += 1
        else:
            if current > 0:
                gaps.append(current)
            current = 0
    if current > 0:
        gaps.append(current)

    if not gaps:
        print("\n[gt-gaps] Ground truth covers every frame.")
        return None

    gaps = np.array(gaps)
    print(f"\n[gt-gaps] gaps in ground truth annotations:")
    print(f"[gt-gaps] count: {len(gaps)}, mean: {gaps.mean():.1f}, "
          f"median: {np.median(gaps):.0f}, 95th pct: {np.percentile(gaps, 95):.0f}, "
          f"max: {gaps.max()}")
    return int(np.percentile(gaps, 95))


# ============================================================
# 10. Diagnostics
# ============================================================
def plot_diagnostics(gt, det, segments, g_estimates):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    valid_gt = ~np.isnan(gt[:, 0])
    ax.plot(gt[valid_gt, 0], gt[valid_gt, 1], "b.", markersize=2, alpha=0.5, label="Ground truth")
    if det is not None:
        valid_det = ~np.isnan(det[:, 0])
        ax.scatter(det[valid_det, 0], det[valid_det, 1], c="r", s=5, alpha=0.5, label="Detections")
    for s, e in segments:
        seg = gt[s:e]
        valid = ~np.isnan(seg[:, 0])
        ax.plot(seg[valid, 0], seg[valid, 1], "g-", linewidth=2, alpha=0.7)
    ax.set_title("Trajectories — green = airborne segments used for gravity fit")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.invert_yaxis(); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(gt[:, 1], "b-", alpha=0.5, label="Ground truth y")
    for s, e in segments:
        ax.axvspan(s, e, color="green", alpha=0.2)
    ax.set_title("y(t) — green bands are airborne")
    ax.set_xlabel("frame"); ax.set_ylabel("y (px)")
    ax.invert_yaxis(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if g_estimates:
        ax.hist(g_estimates, bins=20, edgecolor="black")
        ax.axvline(np.median(g_estimates), color="r", linestyle="--",
                   label=f"median = {np.median(g_estimates):.0f}")
        ax.set_title("Per-segment gravity estimates (px/s²)")
        ax.set_xlabel("g (px/s²)"); ax.legend()
    else:
        ax.text(0.5, 0.5, "No gravity fits", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if det is not None:
        valid = ~np.isnan(det[:, 0]) & ~np.isnan(gt[:, 0])
        if valid.sum() > 0:
            res = det[valid] - gt[valid]
            ax.scatter(res[:, 0], res[:, 1], s=3, alpha=0.4)
            ax.set_title(f"Detector residuals (det - gt), {valid.sum()} frames")
            ax.set_xlabel("dx (px)"); ax.set_ylabel("dy (px)")
            ax.axhline(0, color="k", linewidth=0.5); ax.axvline(0, color="k", linewidth=0.5)
            ax.set_aspect("equal")
        else:
            ax.text(0.5, 0.5, "No overlapping det/gt frames", ha="center", va="center",
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No detections provided", ha="center", va="center",
                transform=ax.transAxes)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("calibration_diagnostics.png", dpi=120)
    print("\n[plot] saved calibration_diagnostics.png")


# ============================================================
# Main
# ============================================================
def main():
    dt = 1.0 / FPS

    gt_dict, n_frames = load_coco_ground_truth(COCO_PATH)
    det_dict = load_detections(DETECTIONS_PATH)
    gt, det = dicts_to_arrays(gt_dict, det_dict, n_frames)

    print(f"\nClip: {n_frames} frames at {FPS} fps (dt = {dt:.4f} s)")
    print(f"Ground truth available on {(~np.isnan(gt[:, 0])).sum()} / {n_frames} frames")
    if det is not None and np.any(~np.isnan(det[:, 0])):
        print(f"Detections available on {(~np.isnan(det[:, 0])).sum()} / {n_frames} frames")

    segments = find_airborne_segments(gt, dt)
    print(f"\nFound {len(segments)} airborne segments")

    gravity, g_estimates = estimate_gravity(gt, dt, segments)
    R, rms = estimate_measurement_noise(gt, det)
    pos_var, vel_var, accel_var = estimate_process_noise(gt, dt)
    max_gap = analyze_gaps(det, n_frames)
    gt_max_gap = analyze_gt_gaps(gt)

    plot_diagnostics(gt, det, segments, g_estimates)

    print("\n" + "=" * 60)
    print("CALIBRATED PARAMETERS — paste into your tracker config")
    print("=" * 60)
    print(f"GRAVITY_PX_PER_S2 = {gravity:.1f}")
    print(f"MAX_GAP_FRAMES = {max_gap}")
    print(f"MAHALANOBIS_GATE = 9.21   # chi-squared 99% for 2 DOF")
    print(f"R = np.eye(2) * {np.diag(R).mean():.1f}")
    print(f"Q = np.diag([1.0, 1.0, "
          f"{vel_var/100:.1f}, {vel_var/100:.1f}, "
          f"{accel_var/100:.1f}, {accel_var/100:.1f}])")
    print("=" * 60)


if __name__ == "__main__":
    main()