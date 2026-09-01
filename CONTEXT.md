# Soccer Ball Tracking — Project Context

## Project goal
Real-time soccer commentary system. Needs to know which player has possession
of the ball. Decomposes into two subproblems:

1. **Airborne event detection** — when ball is in the air, output (start, end)
   windows. Downstream commentary suspends possession assignment during these.
2. **On-ground ball tracking** — accurate per-frame ball position when ball is
   on the ground or in low-height play. Drives the closest-player computation
   for possession.

## Repo layout

- `/home/lakshay/lx/ball_detection_tracking/` — pre-airborne work
  - `ball_outlier_interpolator_v4.py` — current main tracker (v4 baseline,
    both ablation phases off). On clip 1: TP=350, Miss=50, FP=70.
    Accepts CLI args: `--video`, `--output-video`, `--output-json`, `--actions-json`.
  - `scripts/run_detections.py` — batch runner; generates `detections/<clip>.json`
    for all clips missing one. Flags: `--clips N [N…]`, `--force`.
  - `ball_detection_metrics.py`, `benchmark_*.py` — eval infra
  - `checkpoints/ball_samy_1120.pth` — current ball model (RF-DETR Medium @ 1120)
  - `checkpoints/player.pth` — player model (RF-DETR Medium);
    class map: {0: parent_class, 1: goalkeeper, 2: player, 3: referee}

- `/home/lakshay/lx/ball_detection_tracking/event_detection/` — airborne work
  - `extract_trajectory_features.py` — per-frame trajectory features over a
    rolling buffer L=30 frames
  - `airborne_rule.py` — weighted-score fire rule
  - `airborne_state_machine.py` — wraps rule with measured-duration landing
    detection (rolling ground baseline + ascent gate + landing streak)
  - `evaluate_airborne_detector.py` — eval against hand-labeled events
  - `run_airborne_eval.py` — full batch eval pipeline: features → state machine
    → evaluate → save. Per-clip to `outputs/airborne_eval/<clip>/`, consolidated
    to `outputs/airborne_eval/summary.{json,csv}`. Flags: `--clips N`, `--no-plots`.
  - `precompute_pose_for_clip.py` + `pose_features.py` — ViTPose-Huge pose
    features (precomputed but NOT yet wired into the rule)
  - `extract_trajectory_features_without_pose.py` — pose-free variant of the
    extractor
  - `visualize_air_only.py` — overlays "AIRBORNE EVENT" label on video
  - `visualize_with_airborne.py` — fuller overlay variant

## Status as of now

**Airborne detection: significantly improved across 8 clips.**

Latest eval run: `outputs/airborne_eval_v5/` (detections from `detections_v5/`)

| clip | n_gt | n_pred | recall | precision | frame_f1 |
|---|---|---|---|---|---|
| clip1 | 3 | 4 | 1.0 | 0.75 | 0.91 |
| clip2 | 3 | 4 | 1.0 | 0.75 | 0.70 |
| clip3 | 1 | 2 | 1.0 | 0.50 | 0.36 |
| clip4 | 4 | 0 | 0.0 | — | 0.0 |
| clip5 | 2 | 3 | 1.0 | 0.67 | 0.76 |
| clip6 | 1 | 1 | 1.0 | 1.0 | 0.93 |
| clip7 | 1 | 1 | 1.0 | 1.0 | 0.93 |
| testing_clip_1080 | 8 | 10 | 0.875 | 0.70 | 0.58 |
| **MACRO** | — | — | **0.86** | **0.67** | **0.64** |

Pose features precomputed; trajectory-only in use (pose not wired into feature extraction).

**On-ground tracking: v6 green-filter experiments (May 2026).**

Baseline: `ball_outlier_interpolator_v5.py` (KF + Mahalanobis gate).
Best: TP=378, Missed=44, FP=48, No-GT=10, F1=0.8915.
Run: `clip1_fresh_runs/v5_then_ransac_v2_online10__clip1/`.

**v6 experiments: green-ground rejection for interpolated positions.**

Rationale: v5 FPs showed many KF-interpolated positions landing on empty grass.
Approach: suppress interpolated positions on grass-colored pixels (HSV h∈35–85, s≥40, v∈40–200).

| Variant | Coverage | Rejections | TP | Missed | FP | F1 |
|---|---|---:|---:|---:|---:|---:|
| v6 | 80% full 20×20 bbox | 57 | 362 | 94 | 18 | 0.8715 |
| v6_1 | 80% center 10×10 | 59 | 362 | 96 | 12 | 0.8712 |
| v6_2 | 95% full 20×20 | 57 | 362 | 94 | 14 | 0.8715 |
| v6_3 | 95% adaptive crop (from last accepted detection) | 51 | 363 | 89 | 18 | 0.8715 |

Best: **v6_3**. Still 0.02 F1 below v5 — FP reduction (48→18) cannot offset miss regression (44→89).

**Diagnostic findings (v6_3):**
- 51 rejections: ~40 genuine FPs on grass, ~11 valid interpolations where ball is
  too small relative to search window.
- 89 misses (vs 44 in v5): mostly cascades from rejected interpolations disrupting KF track.
- Visuals: `v6_3-visuals/fps/` (18 frames), `v6_3-visuals/misses/` (89 frames).

**Conclusion:** Green filter is a dead end at current ball detection scale. Color-based
discrimination at 20–26px bbox is too weak — removes valid interpolations alongside FPs.
Next: diagnose the 48 v5 FPs directly.

**Version history note:** `ball_outlier_interpolator_v4_no_interpolation.py` (previously
named v5) stripped KF from v4 to isolate its contribution — dead end, reaches only
TP=352, Missed=118. v6/v7 built on that dead-end branch have been deleted.

v5 fixes over v4: (1) frame_count 0→1 to align with COCO GT, (2) RESET_MAX_DISTANCE
300px cap to prevent teleports, (3) KF-interpolated positions saved to JSON so
RANSAC v2 can repair bad physics fits.

---

## Ball tracking metrics contract

Per-frame metrics from `scripts/ball_detection_metrics.py` + distance bucket from
`scripts/distance_rule_adjustment.py`.

**Four metrics to report by default:**

1. `TP`: `event_tp`
   - GT ball exists; at least one prediction exists; top-scored prediction matches
     GT with `IoU >= match_iou` (current runs use `match_iou = 0.01`).

2. `Missed`: `missed_detection_count`
   - GT ball exists, no prediction for that frame.

3. `FP (all)`: `false_positive_count + no_gt_predicted`
   - `false_positive_count`: GT exists and prediction exists but top prediction
     doesn't match by IoU.
   - `no_gt_predicted`: no GT exists but tracker still outputs a prediction.
   - Label `false_positive_count` alone as `FP (GT frames)` to avoid ambiguity.

4. `Detection distance <= threshold`: `distance_le_threshold_px_count`
   - For each GT-frame FP: `distance = center_distance(gt_box, predicted_box)`,
     `threshold = 2 * (predicted_bbox_width + predicted_bbox_height)`.
   - Threshold computed from predicted bbox for that frame. Do not average over GT boxes.

Current best clip1 RANSAC/v4 v2 run:
`clip1_fresh_runs/v4_then_ransac_v2_online10__clip1/`

**Pipeline:**
1. `ball_outlier_interpolator_v4.py` (v4 baseline tracker)
2. `scripts/benchmark_physics_ransac_v2_clip1.py --online_lookahead_frames 10`

**To re-run:**
```bash
python scripts/benchmark_physics_ransac_v2_clip1.py \
  --run_name v4_then_ransac_v2_online10__clip1 \
  --online_lookahead_frames 10
```

| Metric | Value |
|---|---:|
| TP | 369 |
| Missed | 32 |
| FP (GT frames) | 69 |
| No-GT predicted | 10 |
| FP (all) | 79 |
| Detection distance <= predicted-box threshold | 27 |

Predicted boxes are `20x20`, so per-frame distance threshold = `2 * (20 + 20) = 80 px`.

---

## Airborne detection — session findings & changes (May 2026)

### Pipeline
```
detections_v5/<clip>.json
  → extract_trajectory_features_without_pose.py   (rolling-buffer features)
    → airborne_rule.py                             (per-frame weighted score)
      → airborne_state_machine.py                  (event start/end tracking)
        → run_airborne_eval.py                     (batch eval entry point)
```

### Active config (as of latest run)

**`airborne_rule.py` — `AirborneRuleConfig`:**
- `min_density_for_ground = 0.3` ← lowered from 0.5 (Config A fix)
- `upward_dy_per_frame_threshold = -8.0`
- `weight_upward_velocity = 2.0`, `weight_sudden_gap = 0.5`
- `fire_threshold = 1.2`
- `upward_velocity_consistency_frames = 3`, `upward_velocity_consistency_threshold = -3.0`

**`airborne_state_machine.py` — `StateMachineConfig`:**
- `landing_consecutive_detections = 2` ← lowered from 3
- `post_event_cooldown_frames = 15` ← new, was absent
- `min_ascent_before_landing_px = 80.0`
- `min_event_ascent_px = 80.0` ← new, precision filter
- `max_airborne_frames = 90`

### What was tried and why

**Config A (`min_density_for_ground = 0.3`):**
Clip 6 had 6 null frames before launch (279–284). With gate=0.5, the two
strongest-velocity frames (286: dy=-12.35, 288: dy=-13.36) were blocked (density=0.4).
By frame 290 when density hit 0.5, velocity had dropped below -8.0.
Lowering to 0.3 lets those frames through → clip 6: 0→1.0 recall.
Macro recall: 0.73→0.86. Precision: 0.55→0.67.

**Configs B–E:** No meaningful improvement over A.
- B (ground_window=6): slight recall drop on some clips.
- C (threshold=-6.0): same as A; no new events unlocked.
- D (fire_threshold=1.4): precision slightly better, recall dropped. 1.2 is sweet spot.
- E (weight_sudden_gap=1.0): same as A; clip4 still unrecoverable.
- fire_threshold tested at 1.0, 1.2, 1.4, 1.6: 1.2 is best.

**Velocity-consistency check moved into state machine:**
Added `_passes_consistency_check()` to `AirborneStateMachine.step()` (≥2 of last 3
frames with `dy ≤ -3.0`). Zero effect on current FPs — they are sustained multi-frame
motions, not single-frame spikes.

**Post-event cooldown (15 frames):**
Two FPs (clip1 pred 181–218, clip2 pred 90–180) re-fire immediately after a prior
event ends. Cooldown after every completed event suppresses these.

**`landing_consecutive_detections` lowered 3→2:**
Clip3 overshoots GT by 51 frames (pred ends at 291, GT at 240) — ball briefly touches
ground but only 1–2 consecutive near-ground frames, never reaching streak=3. Lowering
to 2 allows landing on the brief touch.

### Policy shift: precision-first

After recall=0.86, precision=0.67: shift to precision-first. Remaining FPs include
small ball bounces (ball briefly leaves ground 20–60 px) that aren't real events.

**Precision filter `min_event_ascent_px = 80`:**
After event completes, check `ascent = baseline_y - min_y_during_airborne`. If
`< min_event_ascent_px`, drop silently without cooldown (so a real launch immediately
after a fake bounce is still allowed). Raise to 100–120 for stricter precision.

### FP analysis (current 7 FPs)

| FP | Clip | Frame | Root cause |
|---|---|---|---|
| Unlabeled real event | clip3 | 30 | Ball genuinely airborne; GT label starts at 199 |
| Config-A-introduced | testing_clip_1080 | 1287 | density=0.367, passes 0.3 gate; likely unlabeled event |
| Re-fire (bounce) | clip2 | 90 | Ball bounces at ~85; fixed by cooldown |
| Re-fire (long event) | clip1 | 181 | max_duration hit at 175, re-fires at 181; fixed by cooldown |
| GT boundary | clip5 | 537 | Ball still ascending 2 frames after GT end |
| GT boundary | testing_clip_1080 | 903 | Fires 1 frame after GT event ends |
| GT boundary | testing_clip_1080 | 2947 | Fires 3 frames after GT event ends |

**Precision ceiling with current GT labels: ~0.82** (5/7 FPs are unlabeled events
or GT boundary issues, not system errors).

### Clip 4 — unrecoverable with current detector

All 4 GT events have zero or near-zero detection density. Events 3 and 4 have 20–200
consecutive null frames covering the entire window. Needs a better detector or different
signal source.

## Ground truth format change (incoming)

Hand-labeled action JSONs are being upgraded:
- Old format: `{"frame": N, "action": "<class>"}` — only start frame
- New format: `{"start_frame": N, "end_frame": M, "action": "airborne"}` —
  measured spans, single class

When loading actions, support both formats. `event_type` is no longer used for behavior.

## Test data
- Videos: `clips/clip1.mp4` … `clips/clip7.mp4`, `clips/testing_clip_1080.mp4`
- Ground-truth action labels: `ground_truths/<clip_name>_actions.json`
  (new format: `{start_frame, end_frame, action: "airborne"}`, both inclusive)
- Detection JSONs: `detections_v5/<clip_name>.json` (all clips: 1–7, testing_clip_1080)
- Annotated output videos: `outputs/videos/<clip_name>_v4.mp4`
- COCO GT: `train/_annotations.coco.json`
- Pose features cache: `clip1_pose_features.json`

## Hardware
Remote: Tesla T4 GPU, CUDA available.

## Airborne detection v3 — Player-feet ground baseline (May 2026, WIP)

### Goal

Fix stale-baseline problem in clips with camera pans/zooms (e.g., testing_clip_1080),
where the rolling deque averages y across multiple distinct ground levels. Alternative:
use per-frame player-feet (y2 bbox coordinate) as direct ground estimate.

### Implementation

- `precompute_player_detections.py` → `player_detections_v5/<clip>.json`
- `extract_trajectory_features_with_players_v3.py` — adds `player_ground_baseline_y`
  (90th percentile of player bbox y2 per frame)
- `airborne_state_machine_v3.py` — uses player baseline in `_update_ground_baseline()`
  with deque-median fallback

### Results (v3 on clip1, clip2)

| Metric | v2 | v3 | Change |
|---|---:|---:|---|
| clip1 recall | 1.0 | 0.667 | ↓ REGRESSION |
| clip1 start_error | 35.67 | 83.5 | ↑ worse |
| clip1 end_error | 27.0 | 37.0 | ↑ worse |
| clip2 recall | 1.0 | 1.0 | — |
| clip2 end_error | 18.67 | 95.3 | ↑↑ much worse |

**Root cause:** Per-frame player baseline (even smoothed with 90th percentile) is too
noisy. When baseline shifts between frames, landing-distance check
(`|ball_y - baseline_y| < 50px`) fails to trigger → ball exits AIRBORNE via `max_duration`.

### V3_1 hybrid baseline — Deque primary + player fallback (May 2026)

Keep v2's rolling deque as primary. Use player-feet (90th percentile y2) only as
fallback initialization when deque is empty (first ~5 frames). Once deque builds,
proceed as v2.

- `airborne_state_machine_v3_1.py` — adds `_baseline_source` tracking (deque vs player)
- `run_airborne_eval_v3_1.py` — batch eval runner

| Metric | clip1 | clip2 | testing_1080 |
|---|---|---|---|
| v2 recall | 1.0 | 1.0 | 0.6 |
| v3_1 recall | 1.0 | 1.0 | 0.6 |
| v2 mean_end_err | 27.0 | 18.67 | 106.0 |
| v3_1 mean_end_err | 27.0 | 18.67 | 106.0 |

**v3_1 preserves v2 exactly** — safe hybrid foundation. Does not fix testing_clip_1080
stale-baseline (106f end error persists): deque initializes from early frames and becomes
fixed; player fallback only initializes in first ~5 frames and doesn't reset mid-clip.

### V3_2: Velocity-stationarity landing (May 2026, FAILED)

**Approach:** Landing triggers only when BOTH ball is near baseline (distance < 50px)
AND velocity is stationary (`|dy| < threshold` for last N frames).

Two parameter sets tested:
- **Strict**: 5-frame window, 2.0 px/frame threshold
- **Relaxed**: 3-frame window, 4.0 px/frame threshold

**Results (Relaxed params):**

| Metric | v3_1 | v3_2 | Change |
|---|---:|---:|---|
| clip1 recall | 1.0 | 0.667 | ↓ lost 1 event |
| clip1 end_error | 27.0 | 31.0 | ↑ worse |
| clip2 recall | 1.0 | 1.0 | — |
| clip2 end_error | 18.67 | 49.0 | ↑↑ +30f |
| clip4 recall | 0.0 | 0.0 | — |
| MACRO recall | 1.0 | 0.656 | ↓ -0.344 |
| MACRO end_error | 22.8 | 57.4 | ↑ +34.6f |

**REJECTED.** Velocity-stationarity too strict. Real falling balls have velocity noise
and micro-bounces — check fails on legitimate landings → state machine falls back to
max_duration → huge end errors.

### Path forward

- **Option B (camera-pan baseline reset):** Detect when player-baseline diverges >100px
  from deque-median; reset deque to player baseline. Solves testing_clip_1080 stale-baseline
  but complex.
- **Option C (adaptive landing gate):** Use clip properties (detection density, ascent ratio)
  to set landing distance dynamically. May not be worth complexity given v3_1 stability.

---

## Working style

- **Brainstorm before coding.** Don't write code from a one-line request.
- **Diagnose before fixing.** Always ablate independently.
- **Test on multiple clips.** Minimum 2-clip regression check before accepting a change.
- **Concise responses.** No over-explaining. No restating what I already know.
- **Real numbers > guesses.** "It should work" doesn't count. Run the eval.
- **Ask before assuming.** When a decision point arises, ask the user. Don't pick silently.
- **Commit every change.** 5–10 word message per edit.

## Key references

- Airborne detector: macro recall 0.86, precision 0.67 across 8 clips. State machine
  fixes: cooldown (15f) + landing streak=2. Remaining FPs mostly GT labeling gaps.
- Ground tracking: v5 baseline TP=378, Missed=44, FP=48, F1=0.8915. v6 family (green
  filter) drops to F1=0.8715. Next: analyze the 48 v5 FPs.
- Ground tracker benchmark command:
  ```bash
  uv run python ball_outlier_interpolator_v5.py --video clips/clip1.mp4 \
      --output-json clip1_v5.json --output-video clip1_v5.mp4
  uv run scripts/benchmark_physics_ransac_v2_clip1.py \
      --tracker_json clip1_v5.json \
      --airborne_json ground_truths/clip1_actions.json \
      --run_name my_run_name \
      --online_lookahead_frames 10
  ```
- Metrics logged to: `ball_detection_metrics.csv`
- Visualizations: `v6_3-visuals/fps/` (18 FP frames), `v6_3-visuals/misses/` (89 miss frames)
- Active task tracker: `TODO.md`
