import os
import json
import cv2
import csv
import numpy as np
import contextlib
import io
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRMedium
from tqdm import tqdm

# =========================
# CONFIG
# =========================

DATASET_ROOT          = "/home/lakshay/folder_to_zip/Soccer-Tracking-6"
MERGED_ANN_JSON       = os.path.join(DATASET_ROOT, "_annotations_merged.coco.json")
PLAYER_MODEL_CHECKPOINT = "/home/lakshay/folder_to_zip/models/player.pth"
BALL_MODEL_CHECKPOINT   = "/home/lakshay/folder_to_zip/models/ball_1120.pth"

# FIX 1: Must be extremely low for COCO evaluation to draw the PR Curve
CONF_THRESHOLD        = 0.01 
OUTPUT_JSON           = "results_dual_model_new_ball.json"
CSV_OUTPUT            = "per_class_metrics_dual_new_ball.csv"

# =========================
# MODEL → GT category mappings
# =========================

PLAYER_MODEL_TO_GT = {
    1: 2,   # goalkeeper → Goalkeeper
    2: 4,   # player     → Soccer Player
    3: 3,   # referee    → Referee
}

BALL_MODEL_TO_GT = {
    0: 1,   # ball → Ball (ID 0 from model -> ID 1 in GT)
}

# =========================
# LOAD MODELS
# =========================

print("Loading player model...")
player_model = RFDETRMedium(pretrain_weights=PLAYER_MODEL_CHECKPOINT)
player_model.optimize_for_inference() # FIX 2: Ensure dropout layers are off
print("Player model loaded.")

print("Loading ball model...")
ball_model = RFDETRMedium(pretrain_weights=BALL_MODEL_CHECKPOINT, resolution=1120)
ball_model.optimize_for_inference() # FIX 2
print("Ball model loaded.\n")

# =========================
# LOAD GT
# =========================

coco_gt = COCO(MERGED_ANN_JSON)

with open(MERGED_ANN_JSON) as f:
    gt_data = json.load(f)

img_id_to_fname = {img["id"]: img["file_name"] for img in gt_data["images"]}

# =========================
# RUN INFERENCE
# =========================

results = []

def collect_detections(detections, class_map, img_id):
    """Convert sv.Detections output to COCO result dicts using class_map."""
    out = []
    if detections is None or len(detections) == 0:
        return out
    
    # Safe indexing to avoid Supervision library bug
    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = detections.xyxy[i]
        score    = float(detections.confidence[i])
        class_id = int(detections.class_id[i])
        
        gt_cat_id = class_map.get(class_id)
        if gt_cat_id is None:
            continue
            
        out.append({
            "image_id":    img_id,
            "category_id": gt_cat_id,
            "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score":       score,
        })
    return out

print("Running inference (both models)...")

# Wrap in inference mode for speed and memory
with torch.inference_mode():
    for img_id, fname in tqdm(img_id_to_fname.items()):
        img_path = os.path.join(DATASET_ROOT, fname)
        if not os.path.exists(img_path):
            print(f"  WARNING: image not found: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"  WARNING: could not read: {img_path}")
            continue

        # FIX 3: Convert BGR to RGB so the model isn't colorblind
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # FIX 4: Use 'confidence=' API argument
        player_dets = player_model.predict(rgb_image, confidence=CONF_THRESHOLD)
        ball_dets   = ball_model.predict(rgb_image, confidence=CONF_THRESHOLD)

        results.extend(collect_detections(player_dets, PLAYER_MODEL_TO_GT, img_id))
        results.extend(collect_detections(ball_dets,   BALL_MODEL_TO_GT,   img_id))

print(f"\nTotal detections: {len(results)}")

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f)

# =========================
# COCO EVAL
# =========================

if len(results) == 0:
    print("No detections — cannot evaluate. Lower CONF_THRESHOLD.")
    exit()

coco_dt = coco_gt.loadRes(OUTPUT_JSON)

# ---- Overall metrics ----
print("\n" + "="*85)
print("OVERALL METRICS — Dual Model (Player + Ball)")
print("="*85)

eval_all = COCOeval(coco_gt, coco_dt, "bbox")
eval_all.evaluate()
eval_all.accumulate()
eval_all.summarize()

# =========================
# PER-CLASS METRICS
# =========================

cat_names = {c["id"]: c["name"] for c in gt_data["categories"] if c["id"] != 0}

print("\n" + "="*85)
print("PER-CLASS METRICS")
print("="*85)
print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'mAP@.50':>10} {'mAP@.50:.95':>12} {'mAR@.50':>10} {'mAR@.50:.95':>12}")
print("-" * 85)

csv_rows = []

for cat_id, cat_name in sorted(cat_names.items()):
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.params.catIds = [cat_id]
    e.params.imgIds = coco_gt.getImgIds()


    e.params.iouThrs = np.array([0.10, 0.25, 0.50])    
    e.evaluate()
    e.accumulate()
    
    with contextlib.redirect_stdout(io.StringIO()):
        e.summarize()

    stats = e.stats

    if stats is None or len(stats) == 0:
        map_50_95, map_50, mar_50_95 = 0.0, 0.0, 0.0
        precision, recall, mar_50 = 0.0, 0.0, 0.0
    else:
        map_50_95 = stats[0]
        map_50    = stats[1]
        mar_50_95 = stats[8]

        precision = 0.0
        recall    = 0.0
        mar_50    = 0.0

        if e.eval is not None:
            if "precision" in e.eval:
                prec = e.eval["precision"]
                iou_idx = np.where(e.params.iouThrs == 0.5)[0]
                if len(iou_idx) > 0:
                    iou_idx = iou_idx[0]
                    prec_slice = prec[iou_idx, :, 0, 0, -1]
                    prec_slice = prec_slice[prec_slice > -1]
                    if len(prec_slice) > 0:
                        precision = float(np.mean(prec_slice))

            if "recall" in e.eval:
                rec = e.eval["recall"]
                iou_idx = np.where(e.params.iouThrs == 0.5)[0]
                if len(iou_idx) > 0:
                    iou_idx = iou_idx[0]
                    rec_val = rec[iou_idx, 0, 0, -1]
                    if rec_val > -1:
                        recall = float(rec_val)
                        mar_50 = float(rec_val)

    row = [cat_name, precision, recall, map_50, map_50_95, mar_50, mar_50_95]
    csv_rows.append(row)

    print(f"{cat_name:<15} {precision:>10.4f} {recall:>10.4f} {map_50:>10.4f} {map_50_95:>12.4f} {mar_50:>10.4f} {mar_50_95:>12.4f}")

print("="*85)

# =========================
# SAVE CSV
# =========================

with open(CSV_OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Class", "Precision", "Recall", "map.50", "map.50:95", "mar.50", "mar.50:95"
    ])
    writer.writerows(csv_rows)

print(f"\n✅ CSV saved to: {CSV_OUTPUT}")

# =========================
# OPTIONAL: run each model solo for diagnostics
# =========================

def eval_single(name, solo_results, gt):
    if not solo_results:
        print(f"\n{name}: no detections")
        return
    tmp = f"_tmp_{name}.json"
    with open(tmp, "w") as f:
        json.dump(solo_results, f)
    dt = gt.loadRes(tmp)
    e  = COCOeval(gt, dt, "bbox")
    e.evaluate()
    e.accumulate()
    print(f"\n{'='*85}\nOVERALL — {name}\n{'='*85}")
    e.summarize()
    os.remove(tmp)

player_only = [r for r in results if r["category_id"] in PLAYER_MODEL_TO_GT.values()]
ball_only   = [r for r in results if r["category_id"] in BALL_MODEL_TO_GT.values()]

eval_single("Player model only", player_only, coco_gt)
eval_single("Ball model only",   ball_only,   coco_gt)
