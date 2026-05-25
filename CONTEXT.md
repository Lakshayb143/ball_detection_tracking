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
  - `checkpoints/player.pth` — player model (RF-DETR Medium)
    - class map: {0: parent_class, 1: goalkeeper, 2: player, 3: referee}

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

Pose features precomputed, weighted-score rule supports them, but currently
using trajectory-only (pose not wired into feature extraction).

**On-ground tracking: v6 green-filter experiments (May 2026).**

Baseline: `ball_outlier_interpolator_v5.py` (KF + Mahalanobis gate).
Best: TP=378, Missed=44, FP=48, No-GT=10, F1=0.8915. 
Run: `clip1_fresh_runs/v5_then_ransac_v2_online10__clip1/`.

**v6 experiments: green-ground rejection for interpolated positions.**

Rationale: analysis of v5 FPs showed many come from KF-interpolated positions
landing on empty grass (ball ocluded by field geometry or simply not detected).
Approach: when interpolated position lands on grass-colored pixels (HSV h∈35–85,
s≥40, v∈40–200), suppress it.

Variants tested:
- **v6** (80% full 20×20 bbox): 57 rejections → TP=362, Missed=94, FP=18, F1=0.8715
- **v6_1** (80% center 10×10): 59 rejections → TP=362, Missed=96, FP=12, F1=0.8712
- **v6_2** (95% full 20×20): 57 rejections → TP=362, Missed=94, FP=14, F1=0.8715
- **v6_3** (95% adaptive crop, sized from last accepted detection): 51 rejections 
  → TP=363, Missed=89, FP=18, F1=0.8715

Best: **v6_3** (adaptive crop radius). Still 0.02 F1 below v5 baseline — FP 
reduction (48→18) cannot offset the miss regression (44→89). Root cause: ball 
is ~10px diameter; even a 20–26px adaptive crop is mostly grass when ball is 
present. Color-based discrimination at this bbox scale is too weak. The filter 
is removing valid interpolations alongside the FPs.

**Diagnostic findings:**
- v5 has 48 FP (GT frames). v6_3 rejects 51 interpolated positions, of which ~40 
  are genuine FPs on grass but ~11 are valid interpolations where the ball is 
  too small relative to the search window.
- 89 misses in v6_3 (vs 44 in v5): mostly cascades from rejected interpolations 
  disrupting the KF track in the subsequent frames.
- Visual inspection: FP and miss frames saved in `v6_3-visuals/fps/` (18 frames) 
  and `v6_3-visuals/misses/` (89 frames) for manual review.

**Conclusion:** green filter approach is a dead end at current ball detection 
scale. Next direction: diagnose the 48 v5 FPs directly rather than trying to 
filter them blindly.

**Version history note:** `ball_outlier_interpolator_v4_no_interpolation.py` was
an experiment (previously named v5) that stripped KF from v4 to isolate its
contribution. It was a dead end — no-interp + RANSAC only reaches TP=352,
Missed=118. v6 and v7 were built on that dead-end branch and have been deleted.

v5 fixes over v4: (1) frame_count 0→1 to align with COCO GT, (2) RESET_MAX_DISTANCE
300px cap to prevent teleports, (3) KF-interpolated positions saved to JSON so
RANSAC v2 can repair bad physics fits.

---

## Ball tracking metrics contract

For ball tracking benchmarks, use the event-style per-frame metrics from
`scripts/ball_detection_metrics.py`, then add the distance bucket from
`scripts/distance_rule_adjustment.py`.

**Four metrics to report by default:**

1. `TP`: `event_tp`
   - A GT ball exists in the frame.
   - At least one prediction exists.
   - The top-scored prediction matches the GT ball with `IoU >= match_iou`.
   - Current RANSAC/v4 runs use `match_iou = 0.01`.

2. `Missed`: `missed_detection_count`
   - A GT ball exists in the frame.
   - No prediction exists for that frame.

3. `FP (all)`: `false_positive_count + no_gt_predicted`
   - `false_positive_count`: GT ball exists and a prediction exists, but the
     top prediction does not match by IoU.
   - `no_gt_predicted`: no GT ball exists, but the tracker still outputs a
     prediction.
   - Older score tables sometimes used `FP = false_positive_count` only. If
     reporting that number, label it as `FP (GT frames)` to avoid ambiguity.

4. `Detection distance <= threshold`: `distance_le_threshold_px_count`
   - This is a bucket for GT-frame FPs, not a fixed 50 px rule and not an
     averaged GT-box rule.
   - For each GT-frame FP:
     - `distance = center_distance(gt_box, predicted_box)`
     - `threshold = 2 * (predicted_bbox_width + predicted_bbox_height)`
     - Count the frame if `distance <= threshold`.
   - The threshold is computed from the predicted bbox for that same frame.
     Do not average over GT boxes.

Current best clip1 RANSAC/v4 v2 run:
`clip1_fresh_runs/v4_then_ransac_v2_online10__clip1/`

**Pipeline:** 
1. `ball_outlier_interpolator_v4.py` (v4 baseline tracker)
2. `scripts/benchmark_physics_ransac_v2_clip1.py` with `--online_lookahead_frames 10` (RANSAC v2 physics-based outlier removal)

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

For this run, predicted boxes are `20x20`, so the per-frame distance threshold is
`2 * (20 + 20) = 80 px`.

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
- Clip 6 had 6 null frames before launch (frames 279–284). With gate=0.5, the two
  strongest-velocity frames (286: dy=-12.35, 288: dy=-13.36) were blocked (density=0.4).
  By frame 290 when density hit 0.5, velocity had dropped below the -8.0 threshold.
  Lowering gate to 0.3 lets those frames through. Clip 6 went from 0 to 1.0 recall.
- Macro recall: 0.73 → 0.86. Precision: 0.55 → 0.67.

**Configs B, C, D, E:** Tested. No meaningful improvement over A.
- B (shorter ground_window=6): slight recall drop on some clips.
- C (threshold=-6.0): same result as A; no new events unlocked.
- D (fire_threshold=1.4): precision slightly better, but recall dropped. 1.2 is the sweet spot.
- E (weight_sudden_gap=1.0): same result as A; clip4 still unrecoverable.
- fire_threshold tested at 1.0, 1.2, 1.4, 1.6: 1.2 is best.

**Velocity-consistency check moved into state machine:**
- Check existed in `detect_airborne_events_offline` but was absent from
  `AirborneStateMachine.step()`. Added `_passes_consistency_check()` method.
- Requires ≥2 of last 3 frames to have `dy_per_frame ≤ -3.0`.
- Result: zero effect on current FPs — the FPs are sustained multi-frame motions,
  not single-frame spikes.

**Post-event cooldown (15 frames) added to state machine:**
- Two FPs (clip1 pred 181–218, clip2 pred 90–180) are re-fires immediately after
  a prior event ends while the ball is still in flight or bouncing. A 15-frame
  cooldown after every completed event suppresses these.

**`landing_consecutive_detections` lowered 3 → 2:**
- Clip3 predicted event overshoots GT by 51 frames (pred ends at 291, GT at 240)
  because the ball briefly touches ground and bounces — only 1–2 consecutive
  near-ground frames, never reaching the streak=3 requirement. Event hits max_duration.
  Lowering to 2 allows landing to be detected on the brief touch.

### Policy shift: precision-first

After getting recall to 0.86 and precision to 0.67, the strategy was changed to
"precision-first" — fix precision first, then push recall back up. The remaining
FPs include "small ball bounces" during passes: ball briefly leaves ground at low
height (20–60 px) but isn't a real airborne event in the soccer-commentary sense.

**Precision filter `min_event_ascent_px = 80`:**
- Velocity peaks momentarily even on small bounces, so the rule fires.
- Discriminator is **peak ascent above ground baseline**, not velocity.
- After every event completes (landing or max_duration), check
  `ascent = baseline_y - min_y_during_airborne`. If `< min_event_ascent_px`,
  drop the event silently; do NOT apply cooldown (so a real launch immediately
  after a fake bounce is still allowed).
- Default 80 matches `min_ascent_before_landing_px` for consistency. Raise to
  100–120 for stricter precision.

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

All 4 GT events have zero or near-zero detection density during the event window.
Events 3 and 4 have 20–200 consecutive null frames covering the entire window.
No rule-level change can fix this — the ball detector simply doesn't see the ball
in flight on this clip. Needs either a better detector or a different signal source.

## Ground truth format change (incoming)

Hand-labeled action JSONs are being upgraded:
- Old format: `{"frame": N, "action": "<class>"}` — only start frame
- New format: `{"start_frame": N, "end_frame": M, "action": "airborne"}` —
  measured spans, single class

When loading actions from now on, support both formats. `event_type` is no
longer used for behavior — everything is just "airborne yes/no."

## Test data
- Videos: `clips/clip1.mp4` … `clips/clip7.mp4`, `clips/testing_clip_1080.mp4`
- Ground-truth action labels: `ground_truths/<clip_name>_actions.json`
  (new format: `{start_frame, end_frame, action: "airborne"}`, both inclusive)
- Detection JSONs (per-frame ball positions): `detections_v5/<clip_name>.json`
  - All clips (1–7, testing_clip_1080) exist in detections_v5/
- Annotated output videos: `outputs/videos/<clip_name>_v4.mp4`
- COCO GT at `train/_annotations.coco.json`
- Pose features cache: `clip1_pose_features.json`

## Hardware
Remote: Tesla T4 GPU, CUDA available.

## Airborne detection v3 — Player-feet ground baseline (May 2026, WIP)

### Goal

Fix stale-baseline problem in clips with camera pans/zooms (e.g., testing_clip_1080),
where the rolling deque of ground detections averages y across multiple distinct ground
levels, producing incorrect landing detection. Alternative: use per-frame player-feet
(y2 bbox coordinate) as direct ground estimate.

### Implementation

- **New component**: `precompute_player_detections.py` → `player_detections_v5/<clip>.json`
- **New extractor**: `extract_trajectory_features_with_players_v3.py` adds column `player_ground_baseline_y`
  (90th percentile of player bbox y2 coordinates per frame)
- **New state machine**: `airborne_state_machine_v3.py` uses player baseline in `_update_ground_baseline()`
  with fallback to deque median if no players detected.

### Results (v3 on clip1, clip2)

| Metric            | v2    | v3    | Change           |
| ---               | ---:  | ---:  | ---              |
| clip1 recall      | 1.0   | 0.667 | ↓ (REGRESSION)   |
| clip1 start_error | 35.67 | 83.5  | ↑ (worse)        |
| clip1 end_error   | 27.0  | 37.0  | ↑ (worse)        |
| clip2 recall      | 1.0   | 1.0   | —                |
| clip2 start_error | 10.0  | 10.0  | —                |
| clip2 end_error   | 18.67 | 95.3  | ↑ (much worse)   |

### Root cause of regression

Per-frame player baseline (even smoothed with 90th percentile) is too noisy and
unstable relative to rolling deque median. When baseline shifts between frames,
landing-distance check (requires `|ball_y - baseline_y| < 50px`) fails to trigger
at the right time, causing ball to linger in AIRBORNE state and exit via `max_duration`.

### V3_1 hybrid baseline — Deque primary + player fallback (May 2026, completed)

### Implementation

Keep v2's rolling deque as primary baseline. Use player-feet detection (90th percentile y2)
only as fallback initialization when deque is empty (first ~5 frames of clip). Once deque
builds, proceed as v2.

- `airborne_state_machine_v3_1.py` — adds `_baseline_source` tracking (deque vs player)
- Feature extractor reuses v3's `extract_trajectory_features_with_players_v3.py`
- `run_airborne_eval_v3_1.py` — batch eval runner

### Results: v3_1 vs v2

| Metric            | clip1 | clip2  | testing_1080 |
| ---               | ---   | ---    | ---          |
| v2 recall         | 1.0   | 1.0    | 0.6          |
| v3_1 recall       | 1.0   | 1.0    | 0.6          |
| v2 mean_end_err   | 27.0  | 18.67  | 106.0        |
| v3_1 mean_end_err | 27.0  | 18.67  | 106.0        |

**v3_1 preserves v2 exactly** — no regression risk. Safe hybrid foundation.
However, does not fix testing_clip_1080's stale-baseline (106f end error persists).

### Why v3_1 didn't solve testing_clip_1080

Deque initializes from first ground detections (early frames) and that median becomes
fixed for the entire clip. When camera pans to different ground level for later event,
deque baseline no longer applies. Player baseline fallback only initializes deque in
first ~5 frames; it doesn't reset/adjust mid-clip. Fixing stale baseline requires
per-frame reset logic (detect camera pan, reset deque) — larger change.

### Path forward

- **Option B (velocity-stationarity landing)**: Require `|dy| < 2px/frame` for N frames
  PLUS ball near baseline. Fully causal, should help with premature landings (clip10 case).
- **Deeper baseline fix**: Implement per-frame baseline reset on large divergence between
  deque and player baseline (indicates camera pan). More complex but solves testing_clip_1080.

---

## Working style

- **Brainstorm before coding.** Don't write code from a one-line request.
- **Diagnose before fixing.** Past pattern: we kept layering changes before
  understanding failure. Always ablate independently.
- **Test on multiple clips.** Anything that improves clip 1 must be checked
  on clip 2. Two-clip regression check is the minimum bar.
- **Concise responses.** No over-explaining. No restating what I already know.
- **Real numbers > guesses.** "It should work" doesn't count. Run the eval.
- **Ask before assuming.** When a decision point arises (matching criterion,
  format ambiguity, metric definition), ask the user. Don't pick silently.
- **Commit every change.** After each file edit or meaningful step, commit with
  a message of 5–10 words describing the change.

## Key references
- Airborne detector: tuned across 8 clips. Macro recall 0.86, precision 0.67.
  Two state machine fixes applied (cooldown + landing streak=2). Remaining FPs
  are mostly GT labeling gaps, not system errors.

- Ground tracking: v5 baseline TP=378, Missed=44, FP=48, F1=0.8915 is current best.
  v6 family (green filter) does not improve; F1 drops to 0.8715. Next: analyze 
  the 48 v5 FPs to identify alternate improvement direction.

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

- Metrics logged to: `ball_detection_metrics.csv` (all variants)
- Visualizations: `v6_3-visuals/fps/` (18 FP frames), `v6_3-visuals/misses/` 
  (89 miss frames) show ground-truth (green) and predictions (red) for manual review.
- Active task tracker: `TODO.md`
