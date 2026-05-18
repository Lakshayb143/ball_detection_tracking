"""
Visualize FPs and missed detections for the v6_3 RANSAC run.

Saves two folders inside v6_3-visuals/:
  fps/    — 18 FP frames: red=pred, green=GT, arrow + distance label
  misses/ — 89 missed frames: green=GT only, "MISSED" label

Usage:
  uv run python scripts/visualize_v6_3.py
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
RUN_DIR     = REPO_ROOT / "clip1_fresh_runs/v6_3_then_ransac_v2_online10__clip1"
GT_COCO     = REPO_ROOT / "train/_annotations.coco.json"
VIDEO       = REPO_ROOT / "clips/clip1.mp4"
OUT_ROOT    = REPO_ROOT / "v6_3-visuals"
FPS_DIR     = OUT_ROOT / "fps"
MISSES_DIR  = OUT_ROOT / "misses"

BOX_SIZE     = 20.0
MATCH_IOU    = 0.01
BALL_CAT_ID  = 1

# BGR
_GREEN  = (50, 205, 50)
_RED    = (0, 60, 220)
_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)
_YELLOW = (0, 200, 255)
_ORANGE = (0, 165, 255)


# ──────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────

def load_frame_trace(run_dir):
    with open(Path(run_dir) / "frame_trace.csv") as f:
        return list(csv.DictReader(f))


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


# ──────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────

def _iou(pred_cx, pred_cy, gt_bbox, box_size):
    px1 = pred_cx - box_size / 2;  py1 = pred_cy - box_size / 2
    px2 = pred_cx + box_size / 2;  py2 = pred_cy + box_size / 2
    gx, gy, gw, gh = (float(v) for v in gt_bbox)
    ix1 = max(px1, gx);            iy1 = max(py1, gy)
    ix2 = min(px2, gx + gw);       iy2 = min(py2, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_size * box_size + gw * gh - inter
    return inter / union if union > 0 else 0.0


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


# ──────────────────────────────────────────────────────────────────
# Frame classification
# ──────────────────────────────────────────────────────────────────

def classify_frames(frame_trace, gt_by_imgid):
    fps, misses = [], []

    for row in frame_trace:
        img_id = int(row["image_id"]) if row["image_id"] else None
        if img_id is None:
            continue
        gts = gt_by_imgid.get(img_id)
        if not gts:
            continue  # no GT ball in this frame — skip

        kept = row["final_kept"] == "True"
        fx = float(row["final_x"]) if row["final_x"] else None
        fy = float(row["final_y"]) if row["final_y"] else None

        if not kept or fx is None or fy is None:
            # Missed: GT exists but no prediction kept
            best_gt = gts[0]
            misses.append({
                "frame_idx": int(row["frame_idx"]),
                "gt_cx": best_gt[0],
                "gt_cy": best_gt[1],
                "gt_bbox": best_gt[2],
            })
        else:
            # Check if TP or FP
            max_iou = max(_iou(fx, fy, bbox, BOX_SIZE) for _, _, bbox in gts)
            if max_iou < MATCH_IOU:
                best_gt = min(gts, key=lambda g: _dist(fx, fy, g[0], g[1]))
                fps.append({
                    "frame_idx": int(row["frame_idx"]),
                    "pred_x": fx,
                    "pred_y": fy,
                    "gt_cx": best_gt[0],
                    "gt_cy": best_gt[1],
                    "gt_bbox": best_gt[2],
                    "dist_px": _dist(fx, fy, best_gt[0], best_gt[1]),
                    "is_interpolated": row["raw_interpolated"] == "True",
                    "final_source": row["final_source"],
                })

    return fps, misses


# ──────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────

def _put_text(img, text, x, y, color, scale=0.6, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _draw_box(img, cx, cy, w, h, color, label, thickness=2):
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    _put_text(img, label, max(x1, 4), max(y1 - 6, 14), color, scale=0.55)


def annotate_fp(frame, fp):
    img = frame.copy()
    gx, gy, gw, gh = (float(v) for v in fp["gt_bbox"])

    _draw_box(img, fp["gt_cx"], fp["gt_cy"], gw, gh, _GREEN, "GT")
    _draw_box(img, fp["pred_x"], fp["pred_y"], BOX_SIZE, BOX_SIZE, _RED, "PRED")

    cv2.arrowedLine(img,
        (int(fp["pred_x"]), int(fp["pred_y"])),
        (int(fp["gt_cx"]),  int(fp["gt_cy"])),
        _YELLOW, 1, tipLength=0.15)

    mid_x = int((fp["pred_x"] + fp["gt_cx"]) / 2)
    mid_y = int((fp["pred_y"] + fp["gt_cy"]) / 2)
    _put_text(img, f"{fp['dist_px']:.0f}px", mid_x + 4, mid_y - 4, _YELLOW, scale=0.5)

    src = "INTERP" if fp["is_interpolated"] else fp["final_source"].upper()
    _put_text(img, f"Frame {fp['frame_idx']:04d}  FP  {src}", 10, 28, _RED, scale=0.65)
    _put_text(img, f"dist={fp['dist_px']:.0f}px", 10, 54, _WHITE, scale=0.55)
    return img


def annotate_miss(frame, miss):
    img = frame.copy()
    gx, gy, gw, gh = (float(v) for v in miss["gt_bbox"])

    _draw_box(img, miss["gt_cx"], miss["gt_cy"], gw, gh, _GREEN, "GT")

    _put_text(img, f"Frame {miss['frame_idx']:04d}  MISSED", 10, 28, _ORANGE, scale=0.65)
    _put_text(img, f"gt=({miss['gt_cx']:.0f},{miss['gt_cy']:.0f})", 10, 54, _WHITE, scale=0.55)
    return img


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    FPS_DIR.mkdir(parents=True, exist_ok=True)
    MISSES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading frame trace…")
    frame_trace = load_frame_trace(RUN_DIR)

    print("Loading GT annotations…")
    gt_by_imgid = load_gt(GT_COCO, BALL_CAT_ID)

    fps, misses = classify_frames(frame_trace, gt_by_imgid)
    print(f"Found {len(fps)} FP frames, {len(misses)} missed frames")

    # Build lookup sets for sequential video read
    fp_lookup    = {f["frame_idx"]: f for f in fps}
    miss_lookup  = {m["frame_idx"]: m for m in misses}
    target_frames = set(fp_lookup) | set(miss_lookup)

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")

    saved_fps = saved_misses = 0
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_frames:
            if frame_idx in fp_lookup:
                fp = fp_lookup[frame_idx]
                src = "interp" if fp["is_interpolated"] else fp["final_source"]
                fname = f"frame_{frame_idx:04d}_{src}.jpg"
                cv2.imwrite(str(FPS_DIR / fname), annotate_fp(frame, fp))
                saved_fps += 1

            if frame_idx in miss_lookup:
                miss = miss_lookup[frame_idx]
                fname = f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(MISSES_DIR / fname), annotate_miss(frame, miss))
                saved_misses += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved_fps} FP images  → {FPS_DIR}/")
    print(f"Saved {saved_misses} miss images → {MISSES_DIR}/")


if __name__ == "__main__":
    main()
