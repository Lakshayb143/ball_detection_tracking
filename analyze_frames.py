"""
Detection diagnostic script.

Looks at three things:
  1. Frame alignment between COCO ground truth and detector outputs.
  2. Distribution of detection errors (residuals) — magnitude, location, direction.
  3. The K worst frames, with image overlays so you can see what's going wrong.

Run after the calibration script. Edit the CONFIG block at the top.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


# ============================================================
# CONFIG — edit these
# ============================================================
COCO_PATH = "/home/lakshay/lx/ball_detection_tracking/train/_annotations.coco.json"
DETECTIONS_PATH = "/home/lakshay/lx/ball_detection_tracking/clip1_output_v2.json"
IMAGES_DIR = "/home/lakshay/lx/ball_detection_tracking/train"   # where COCO image files live
OUTPUT_DIR = "diagnostics_output"
BALL_CATEGORY_NAME = "ball"
N_WORST_FRAMES = 30        # how many bad frames to render
FPS = 30


# ============================================================
# Loaders (reused from calibration script, lightly trimmed)
# ============================================================
def load_coco(coco_path, ball_cat_name=BALL_CATEGORY_NAME):
    with open(coco_path) as f:
        coco = json.load(f)

    ball_cat_id = None
    for cat in coco["categories"]:
        if cat["name"].lower() == ball_cat_name.lower():
            ball_cat_id = cat["id"]
            break
    if ball_cat_id is None:
        for cat in coco["categories"]:
            if "ball" in cat["name"].lower():
                ball_cat_id = cat["id"]
                break
    if ball_cat_id is None:
        raise ValueError("No ball category found")

    images_sorted = sorted(coco["images"], key=lambda im: im["file_name"])
    image_to_frame = {img["id"]: idx for idx, img in enumerate(images_sorted)}
    frame_to_filename = {idx: img["file_name"] for idx, img in enumerate(images_sorted)}

    gt = {}
    for ann in coco["annotations"]:
        try:
            if int(ann["category_id"]) != int(ball_cat_id):
                continue
            img_id = ann["image_id"]
            if img_id not in image_to_frame:
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
        except (ValueError, TypeError, KeyError, IndexError):
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        f = image_to_frame[img_id]
        new_area = w * h
        if f in gt and new_area <= gt[f][4]:
            continue
        gt[f] = (cx, cy, w, h, new_area)

    # keep (cx, cy, w, h)
    gt = {f: (cx, cy, w, h) for f, (cx, cy, w, h, _) in gt.items()}
    return gt, frame_to_filename


def load_detections(det_path):
    with open(det_path) as f:
        data = json.load(f)
    det = {}
    if isinstance(data, list) and len(data) > 0 and "frame_idx" in data[0]:
        for entry in data:
            f_idx = entry["frame_idx"]
            x, y = entry.get("x"), entry.get("y")
            conf = entry.get("confidence")
            if x is not None and y is not None:
                det[f_idx] = (float(x), float(y), float(conf) if conf is not None else None)
    return det


# ============================================================
# Diagnostic 1: frame alignment
# ============================================================
def check_alignment(gt_dict, det_dict):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1: FRAME ALIGNMENT")
    print("=" * 60)

    gt_frames = sorted(gt_dict.keys())
    det_frames = sorted(det_dict.keys())

    print(f"\nGround truth frames: {len(gt_frames)} total")
    print(f"  range: {min(gt_frames)} to {max(gt_frames)}")
    print(f"  first 15: {gt_frames[:15]}")
    print(f"  consecutive? {all(gt_frames[i+1] - gt_frames[i] == 1 for i in range(min(50, len(gt_frames)-1)))}")

    print(f"\nDetector frames: {len(det_frames)} total")
    print(f"  range: {min(det_frames)} to {max(det_frames)}")
    print(f"  first 15: {det_frames[:15]}")
    print(f"  consecutive? {all(det_frames[i+1] - det_frames[i] == 1 for i in range(min(50, len(det_frames)-1)))}")

    overlap = sorted(set(gt_frames) & set(det_frames))
    print(f"\nFrames with both GT and detection: {len(overlap)}")
    if overlap:
        print(f"  first 15: {overlap[:15]}")

    gt_only = sorted(set(gt_frames) - set(det_frames))
    det_only = sorted(set(det_frames) - set(gt_frames))
    print(f"  GT only (detector missed): {len(gt_only)} frames")
    print(f"  Detector only (no GT label): {len(det_only)} frames")

    # Sanity: do detection coordinates fall on top of GT coordinates ON AVERAGE?
    # If alignment is broken, the mean offset will be dominated by the misalignment.
    if overlap:
        residuals = []
        for f in overlap:
            gx, gy, gw, gh = gt_dict[f]
            dx, dy, _ = det_dict[f]
            residuals.append((dx - gx, dy - gy))
        residuals = np.array(residuals)
        print(f"\nMean (det - gt) on overlapping frames: dx={residuals[:, 0].mean():.1f}, dy={residuals[:, 1].mean():.1f}")
        print(f"Median (det - gt): dx={np.median(residuals[:, 0]):.1f}, dy={np.median(residuals[:, 1]):.1f}")
        print(f"  (a large mean offset >> median typically indicates a few outliers, ")
        print(f"   not a systematic alignment issue. Median should be near zero if aligned.)")

    return overlap


# ============================================================
# Diagnostic 2: residual distribution
# ============================================================
def analyze_residuals(gt_dict, det_dict, overlap, output_dir):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2: RESIDUAL DISTRIBUTION")
    print("=" * 60)

    if not overlap:
        print("No overlapping frames; skipping.")
        return None

    rows = []
    for f in overlap:
        gx, gy, gw, gh = gt_dict[f]
        dx, dy, conf = det_dict[f]
        err = np.hypot(dx - gx, dy - gy)
        # ball "size" — geometric mean of bbox dimensions
        ball_size = (gw * gh) ** 0.5 if gw > 0 and gh > 0 else None
        rows.append({
            "frame": f, "gx": gx, "gy": gy, "gw": gw, "gh": gh,
            "dx": dx, "dy": dy, "conf": conf,
            "err_x": dx - gx, "err_y": dy - gy, "err": err,
            "ball_size": ball_size,
            "err_in_ball_widths": err / ball_size if ball_size else None,
        })

    errs = np.array([r["err"] for r in rows])
    err_xs = np.array([r["err_x"] for r in rows])
    err_ys = np.array([r["err_y"] for r in rows])

    print(f"\nError magnitude statistics ({len(errs)} frames):")
    print(f"  mean:   {errs.mean():.1f} px")
    print(f"  median: {np.median(errs):.1f} px")
    print(f"  rms:    {np.sqrt(np.mean(errs**2)):.1f} px")
    print(f"  std:    {errs.std():.1f} px")
    print(f"  25th pct: {np.percentile(errs, 25):.1f} px")
    print(f"  75th pct: {np.percentile(errs, 75):.1f} px")
    print(f"  90th pct: {np.percentile(errs, 90):.1f} px")
    print(f"  95th pct: {np.percentile(errs, 95):.1f} px")
    print(f"  max:      {errs.max():.1f} px")

    # Bimodality check: how many detections are 'good' vs 'bad'?
    # Define a 'good' detection as within 1 ball-width of GT.
    sized = [r for r in rows if r["ball_size"] is not None]
    if sized:
        bw_errors = np.array([r["err_in_ball_widths"] for r in sized])
        print(f"\nError relative to ball size ({len(sized)} frames with size info):")
        print(f"  median ball size: {np.median([r['ball_size'] for r in sized]):.1f} px")
        print(f"  median err in ball-widths: {np.median(bw_errors):.2f}")
        print(f"  fraction within 1 ball-width: {(bw_errors < 1.0).mean()*100:.1f}%")
        print(f"  fraction within 2 ball-widths: {(bw_errors < 2.0).mean()*100:.1f}%")
        print(f"  fraction beyond 5 ball-widths (likely wrong-object): {(bw_errors > 5.0).mean()*100:.1f}%")

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 2a. error histogram (log y)
    ax = axes[0, 0]
    ax.hist(errs, bins=50, edgecolor="black")
    ax.axvline(np.median(errs), color="r", linestyle="--", label=f"median {np.median(errs):.1f}")
    ax.axvline(np.sqrt(np.mean(errs**2)), color="orange", linestyle="--", label=f"rms {np.sqrt(np.mean(errs**2)):.1f}")
    ax.set_yscale("log")
    ax.set_xlabel("error (px)")
    ax.set_ylabel("count (log)")
    ax.set_title("Detection error magnitude (log y)\nBimodal? Heavy-tailed?")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2b. error in ball-widths
    ax = axes[0, 1]
    if sized:
        ax.hist(bw_errors, bins=50, edgecolor="black")
        ax.axvline(1.0, color="green", linestyle="--", label="1 ball-width")
        ax.axvline(5.0, color="red", linestyle="--", label="5 ball-widths (likely wrong object)")
        ax.set_yscale("log")
        ax.set_xlabel("error / ball width")
        ax.set_ylabel("count (log)")
        ax.set_title("Error in units of ball width")
        ax.legend()
        ax.grid(alpha=0.3)

    # 2c. spatial map of errors
    ax = axes[1, 0]
    gxs = np.array([r["gx"] for r in rows])
    gys = np.array([r["gy"] for r in rows])
    sc = ax.scatter(gxs, gys, c=errs, cmap="hot_r", s=20, vmin=0, vmax=np.percentile(errs, 90))
    plt.colorbar(sc, ax=ax, label="error (px)")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Where in the frame are errors largest?")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    # 2d. (err_x, err_y) scatter
    ax = axes[1, 1]
    ax.scatter(err_xs, err_ys, s=10, alpha=0.4)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("error_x (px)")
    ax.set_ylabel("error_y (px)")
    ax.set_title("Direction of error\n(systematic bias appears as offset cluster)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    # add a circle at 1 ball-width if we have sizing
    if sized:
        med_bs = np.median([r["ball_size"] for r in sized])
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(med_bs*np.cos(theta), med_bs*np.sin(theta), "g--", label=f"1 ball-width ({med_bs:.0f}px)")
        ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "residual_analysis.png")
    plt.savefig(plot_path, dpi=120)
    print(f"\n[plot] saved {plot_path}")
    plt.close()

    return rows


# ============================================================
# Diagnostic 3: render the worst frames
# ============================================================
def render_worst_frames(rows, frame_to_filename, images_dir, output_dir, n_worst=N_WORST_FRAMES):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 3: RENDERING WORST FRAMES")
    print("=" * 60)

    if not rows:
        print("No rows to render.")
        return

    sorted_rows = sorted(rows, key=lambda r: -r["err"])
    worst = sorted_rows[:n_worst]

    print(f"\nTop {n_worst} worst detection frames:")
    print(f"{'frame':>8} {'err_px':>8} {'ball_w':>8} {'gx,gy':>20} {'dx,dy':>20} {'conf':>6}")
    for r in worst:
        ball_w = r["ball_size"] if r["ball_size"] else 0
        conf = r["conf"] if r["conf"] is not None else 0
        print(f"{r['frame']:>8} {r['err']:>8.1f} {ball_w:>8.1f} "
              f"{r['gx']:>7.1f},{r['gy']:>6.1f} "
              f"{r['dx']:>7.1f},{r['dy']:>6.1f} "
              f"{conf:>6.2f}")

    # Render images
    bad_frames_dir = os.path.join(output_dir, "worst_frames")
    os.makedirs(bad_frames_dir, exist_ok=True)
    rendered = 0
    skipped = 0

    for rank, r in enumerate(worst, start=1):
        f = r["frame"]
        fname = frame_to_filename.get(f)
        if fname is None:
            skipped += 1
            continue
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            print(f"  [skip] frame {f}: image not found at {img_path}")
            skipped += 1
            continue
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        gx, gy, gw, gh = r["gx"], r["gy"], r["gw"], r["gh"]
        dx, dy = r["dx"], r["dy"]

        # GT box (green)
        x1, y1 = int(gx - gw/2), int(gy - gh/2)
        x2, y2 = int(gx + gw/2), int(gy + gh/2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "GT", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detection point (red)
        cv2.circle(img, (int(dx), int(dy)), 8, (0, 0, 255), 2)
        cv2.circle(img, (int(dx), int(dy)), 1, (0, 0, 255), -1)
        cv2.putText(img, "DET", (int(dx) + 10, int(dy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Line connecting them
        cv2.line(img, (int(gx), int(gy)), (int(dx), int(dy)), (0, 255, 255), 1)

        # Header
        header = f"frame {f}  err={r['err']:.1f}px  ball_w={r['ball_size']:.1f}px  conf={r.get('conf') or 0:.2f}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, header, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = os.path.join(bad_frames_dir, f"rank{rank:02d}_frame{f:05d}_err{int(r['err'])}.jpg")
        cv2.imwrite(out_path, img)
        rendered += 1

    print(f"\nRendered {rendered} frames to {bad_frames_dir}/")
    if skipped > 0:
        print(f"Skipped {skipped} frames (image not found or unreadable)")


# ============================================================
# Diagnostic 4: confidence vs error scatter
# ============================================================
def confidence_analysis(rows, output_dir):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 4: DETECTOR CONFIDENCE vs ERROR")
    print("=" * 60)

    rows_with_conf = [r for r in rows if r.get("conf") is not None]
    if not rows_with_conf:
        print("No confidence values present in detection JSON; skipping.")
        return

    confs = np.array([r["conf"] for r in rows_with_conf])
    errs = np.array([r["err"] for r in rows_with_conf])

    print(f"\nConfidence-stratified error stats ({len(rows_with_conf)} detections):")
    bins = [0.8, 0.85, 0.9, 0.95, 1.01]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        n = mask.sum()
        if n > 0:
            print(f"  confidence in [{lo:.2f}, {hi:.2f}): {n:>3} frames, "
                  f"median_err={np.median(errs[mask]):.1f}px, "
                  f"rms_err={np.sqrt(np.mean(errs[mask]**2)):.1f}px")

    plt.figure(figsize=(10, 6))
    plt.scatter(confs, errs, s=15, alpha=0.5)
    plt.xlabel("detector confidence")
    plt.ylabel("error from GT (px)")
    plt.title("Does higher confidence mean lower error?")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    out = os.path.join(output_dir, "confidence_vs_error.png")
    plt.savefig(out, dpi=120)
    print(f"\n[plot] saved {out}")
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading COCO from {COCO_PATH}")
    gt_dict, frame_to_filename = load_coco(COCO_PATH)
    print(f"  {len(gt_dict)} ball annotations")

    print(f"\nLoading detections from {DETECTIONS_PATH}")
    det_dict = load_detections(DETECTIONS_PATH)
    print(f"  {len(det_dict)} detections")

    overlap = check_alignment(gt_dict, det_dict)
    rows = analyze_residuals(gt_dict, det_dict, overlap, OUTPUT_DIR)
    if rows is not None:
        confidence_analysis(rows, OUTPUT_DIR)
        render_worst_frames(rows, frame_to_filename, IMAGES_DIR, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs in: {OUTPUT_DIR}/")
    print("  - residual_analysis.png       (4 plots: error distribution and bias)")
    print("  - confidence_vs_error.png     (does confidence predict accuracy?)")
    print(f"  - worst_frames/               ({N_WORST_FRAMES} worst-error frames as JPEGs)")
    print("\nLook at the worst-frame images first. They tell you what's actually wrong.")


if __name__ == "__main__":
    main()