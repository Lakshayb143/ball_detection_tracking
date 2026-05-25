"""
Visualize missed detections with GT ball position.

For each frame where GT ball exists but the tracker has no output,
saves an annotated image showing the GT box and a header with frame info.

Usage:
  uv run python scripts/visualize_missed_detections.py \
    --run-dir clip1_fresh_runs/v6_11 \
    --gt-coco train/_annotations.coco.json \
    --video clips/clip1.mp4 \
    --images-dir outputs/missed_frames_v6_11_clip1
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = dict(
    run_dir=REPO_ROOT / "clip1_fresh_runs/v6_11",
    gt_coco=REPO_ROOT / "train/_annotations.coco.json",
    video=REPO_ROOT / "clips/clip1.mp4",
    images_dir=REPO_ROOT / "outputs/missed_frames_v6_11_clip1",
    ball_category_id=1,
)

# BGR colors
_GREEN  = (50, 205, 50)
_ORANGE = (0, 165, 255)
_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)


def load_frame_trace(run_dir):
    path = Path(run_dir) / "frame_trace.csv"
    with open(path) as f:
        return list(csv.DictReader(f))


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


def find_missed_frames(frame_trace, gt_by_imgid):
    """Returns list of dicts for frames with GT ball but no tracker output."""
    missed = []
    for row in frame_trace:
        if row["final_source"] != "no_output":
            continue
        img_id = int(row["image_id"])
        gts = gt_by_imgid.get(img_id)
        if not gts:
            continue  # no GT ball on this frame — not a missed detection
        missed.append({
            "frame_idx": int(row["frame_idx"]),
            "image_id": img_id,
            "raw_predicted": row["raw_predicted"] == "True",
            "gts": gts,
        })
    return missed


def _put_text(img, text, x, y, color, scale=0.6, thickness=1):
    import cv2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _annotate_missed_frame(frame, missed):
    import cv2

    img = frame.copy()
    frame_idx = missed["frame_idx"]
    raw_predicted = missed["raw_predicted"]

    for gt_cx, gt_cy, bbox in missed["gts"]:
        gx, gy, gw, gh = (float(v) for v in bbox)
        x1, y1 = int(gx), int(gy)
        x2, y2 = int(gx + gw), int(gy + gh)

        # Draw GT box in green
        cv2.rectangle(img, (x1, y1), (x2, y2), _GREEN, 2)
        _put_text(img, "GT", max(x1, 4), max(y1 - 6, 14), _GREEN, scale=0.55, thickness=1)

        # Draw crosshair at GT centroid
        cx, cy = int(gt_cx), int(gt_cy)
        cv2.drawMarker(img, (cx, cy), _GREEN, cv2.MARKER_CROSS, 12, 2)

    # Header
    det_str = "detector fired (gated/rejected)" if raw_predicted else "no detection"
    lines = [
        (f"Frame {frame_idx:04d}  MISSED", _ORANGE),
        (f"{det_str}", _WHITE),
    ]
    for i, (text, color) in enumerate(lines):
        _put_text(img, text, 10, 28 + i * 26, color, scale=0.65, thickness=1)

    return img


def save_missed_images(missed_frames, video_path, images_dir):
    import cv2

    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    lookup = {m["frame_idx"]: m for m in missed_frames}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved = 0
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in lookup:
            annotated = _annotate_missed_frame(frame, lookup[frame_idx])
            det_tag = "rejected" if lookup[frame_idx]["raw_predicted"] else "nodet"
            fname = f"frame_{frame_idx:04d}_{det_tag}.jpg"
            cv2.imwrite(str(images_dir / fname), annotated)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} missed-detection images → {images_dir}/")
    return saved


def print_report(missed_frames):
    n_no_det = sum(1 for m in missed_frames if not m["raw_predicted"])
    n_rejected = sum(1 for m in missed_frames if m["raw_predicted"])
    print(f"\n{'='*60}")
    print(f"  MISSED DETECTION REPORT")
    print(f"  Total missed frames : {len(missed_frames)}")
    print(f"  No detection at all : {n_no_det}")
    print(f"  Detector fired, gated/rejected : {n_rejected}")
    print(f"{'='*60}\n")
    print(f"  {'frame':>6}  {'raw_det':>8}  GT balls")
    for m in missed_frames:
        det_str = "yes" if m["raw_predicted"] else "no"
        gt_str = "  ".join(f"({cx:.0f},{cy:.0f})" for cx, cy, _ in m["gts"])
        print(f"  {m['frame_idx']:>6}  {det_str:>8}  {gt_str}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Visualize missed detections with GT")
    parser.add_argument("--run-dir", default=str(DEFAULTS["run_dir"]))
    parser.add_argument("--gt-coco", default=str(DEFAULTS["gt_coco"]))
    parser.add_argument("--video", default=str(DEFAULTS["video"]))
    parser.add_argument("--images-dir", default=str(DEFAULTS["images_dir"]))
    parser.add_argument("--ball-category-id", type=int, default=DEFAULTS["ball_category_id"])
    args = parser.parse_args()

    print(f"Loading frame trace from: {args.run_dir}")
    frame_trace = load_frame_trace(args.run_dir)

    print(f"Loading GT annotations from: {args.gt_coco}")
    gt_by_imgid = load_gt(args.gt_coco, args.ball_category_id)

    missed_frames = find_missed_frames(frame_trace, gt_by_imgid)
    print(f"Found {len(missed_frames)} missed frames (GT exists, no tracker output)")

    print_report(missed_frames)
    save_missed_images(missed_frames, args.video, args.images_dir)


if __name__ == "__main__":
    main()
