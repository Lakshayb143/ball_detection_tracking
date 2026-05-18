# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a soccer ball detection and tracking system for real-time commentary. The system breaks down into two subproblems:
1. **Airborne event detection** — identifying when the ball is in the air (outputs start/end frame windows)
2. **On-ground ball tracking** — accurate per-frame ball position when ball is on the ground

## Running Scripts

**Always use `uv run python` to execute scripts**, not `python` directly:
```bash
uv run python scripts/script_name.py [args]
uv run python ball_outlier_interpolator_v5.py --video clips/clip1.mp4 --output-json output.json
```

## Key Directories & Files

### Core Tracking
- `ball_outlier_interpolator_v5.py` — Current main on-ground tracker (KF + Mahalanobis gate, 1-indexed frames, RESET_MAX_DISTANCE cap)
  - Accepts CLI args: `--video`, `--output-video`, `--output-json`, `--actions-json`
  - Model: `checkpoints/ball_samy_1120.pth` (RF-DETR Medium @ 1120 resolution)
  - Baseline: TP=378, Missed=44, FP=48, F1=0.8915 (+ RANSAC v2 pipeline)

### Airborne Detection
- `event_detection/` — airborne event detection pipeline
  - `extract_trajectory_features.py` — rolling-buffer trajectory features (L=30 frames)
  - `airborne_rule.py` — weighted-score fire rule with velocity/density gates
  - `airborne_state_machine.py` — event state tracking with landing detection
  - `run_airborne_eval.py` — batch evaluation pipeline (features → state machine → metrics)
  - Current performance: Macro recall 0.86, precision 0.67 across 8 clips

### Evaluation & Benchmarking
- `scripts/ball_detection_metrics.py` — evaluation framework (TP, Missed, FP, distance metrics)
- `scripts/benchmark_physics_ransac_v2_clip1.py` — physics-based outlier removal post-processing
  - Use with: `--online_lookahead_frames 10` for best clip1 results
- `scripts/eval_footandball_clip1.py` — FootAndBall model evaluation (currently poor: 6% recall)

### Detections & Ground Truth
- `detections_v5/` — per-frame ball positions for all clips
- `ground_truths/<clip_name>_actions.json` — hand-labeled airborne events
  - Format: `{start_frame, end_frame, action: "airborne"}` (both inclusive, 1-indexed)
- `clips/clip1.mp4` … `clips/clip7.mp4` — video files for development
- `train/_annotations.coco.json` — COCO GT annotations (1-indexed frames)

## Common Commands

### Run tracker on a single clip
```bash
uv run python ball_outlier_interpolator_v5.py \
  --video clips/clip1.mp4 \
  --output-json clip1_output.json \
  --output-video clip1_output.mp4
```

### Evaluate clip1 with RANSAC v2 physics post-processing
```bash
uv run python scripts/benchmark_physics_ransac_v2_clip1.py \
  --tracker_json clip1_output.json \
  --airborne_json ground_truths/clip1_actions.json \
  --run_name my_run_name \
  --online_lookahead_frames 10
```

### Run airborne detection evaluation on clip1
```bash
cd event_detection
uv run python run_airborne_eval.py --clips 1 --no-plots
```

### Compute metrics for tracker output
```bash
uv run python scripts/ball_detection_metrics.py \
  --predictions clip1_output.json \
  --ground_truth train/_annotations.coco.json
```

## Architecture & Key Concepts

### Ball Tracking Pipeline

**RF-DETR ball detector** (1120 resolution) → **Kalman filter** (constant-velocity model) → **Mahalanobis gating** (outlier rejection) → **OutlierConfirmer** (4-frame consensus check) → **Physics-based RANSAC** (optional post-processing)

Key parameters:
- `CONFIDENCE = 0.01` — detection confidence threshold
- `MAHALANOBIS_GATE = 20` — gating threshold
- `BASELINE_MAX_GAP_FRAMES = 13` — KF interpolation window
- `RESET_MAX_DISTANCE = 300px` — max distance for outlier-triggered reset

### Frame Indexing
- **COCO GT**: 1-indexed (frame 1 is first frame)
- **Tracker output**: 1-indexed (fixed in v5 to match COCO)
- **Video frames** (cv2): 0-indexed internally

### Metrics Contract

Use event-style per-frame metrics from `scripts/ball_detection_metrics.py`:

1. **TP** (`event_tp`): GT exists, prediction exists, top prediction matches GT (IoU ≥ 0.01)
2. **Missed**: GT exists, no prediction
3. **FP (all)** = `false_positive_count + no_gt_predicted`
4. **Distance ≤ threshold**: per-GT-frame FP where `distance ≤ 2*(width+height)` of predicted box

Always report on clip1 AND clip2 minimum before committing to any change.

## Git Commit Policy

**Commit after every change**, regardless of size. Message format: **5–8 words max**, describing the change.

Examples:
```
Fix frame indexing off-by-one error
Add KF interpolation position saving
Increase Mahalanobis gate to 25
Update airborne config min density
```

## Working Style & Constraints

From `CONTEXT.md` — expected patterns:

- **Diagnose before fixing** — ablate independently, don't layer changes blind
- **Test on multiple clips** — minimum 2-clip regression check before accepting a change
- **Real numbers > guesses** — "it should work" doesn't count; run the eval
- **Ask before assuming** — when a decision point arises (metrics, formats), ask the user instead of picking silently

## Active Work

See `TODO.md` for current task list. As of latest:
- **Step 1 (v5)**: ✅ Done. v5 + RANSAC v2 online10 achieves TP=378, FP=48, F1=0.8915
- **Step 2 (v6)**: ⬜ Pending. Diagnose remaining FPs for next improvement direction

## Test Data

- **Clips**: clip1–7 (train set) + testing_clip_1080 (9-clip eval set)
- **Detections**: `detections_v5/` for all clips (pre-computed RF-DETR @ 1120)
- **Airborne GT**: `ground_truths/*_actions.json` (new format: start_frame, end_frame, action="airborne")
- **COCO GT**: `train/_annotations.coco.json` for ball boxes (1-indexed frames)

## Model Checkpoints

- `checkpoints/ball_samy_1120.pth` — RF-DETR Medium ball detector (primary)
- `checkpoints/player.pth` — RF-DETR Medium player detector
  - Class map: {0: parent_class, 1: goalkeeper, 2: player, 3: referee}
