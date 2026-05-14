# Ball Tracking — Active TODO

## Current task: Ground tracking improvement
Improving on-ground ball tracking. Working on top of **v4** (KF + Mahalanobis gate),
which is the correct base. Benchmarked with RANSAC v2 pipeline
(`scripts/benchmark_physics_ransac_v2_clip1.py --online_lookahead_frames 10`).

### Version history clarification
- `ball_outlier_interpolator_v4.py` — KF + Mahalanobis gate. The real baseline.
- `ball_outlier_interpolator_v4_no_interpolation.py` — v4 with KF stripped out.
  Was an experiment to isolate the KF's contribution; named v5 at the time.
  Not an improvement. Dead end.
- v6, v7 — built on top of the no-interpolation branch. Discarded.

---

## Baseline

| variant | TP | Missed | FP (GT) | No-GT |
|---|---|---|---|---|
| v4 + RANSAC v2 online10 (best ever) | 369 | 32 | 69 | 10 |
| v4 raw (no RANSAC) | 350 | ~130 | ~20 | ~5 |

The 369 TP row is the target to beat or match with fewer FPs.

---

## Steps

### 🔄 Step 1 — v5: Fix v4's two bugs
**File:** `ball_outlier_interpolator_v5.py` (to be created)
**Changes:**
1. `frame_count = 0` → start at 1. v4 is 0-indexed; COCO GT is 1-indexed.
   Causes a 1-frame misalignment across all clips.
2. Reset spatial cap: when `OutlierConfirmer` fires, only accept the reset if
   the candidate is within `RESET_MAX_DISTANCE = 300px` of last known position.
   Prevents the 715px teleport (run #8, 14 FP frames) seen in v4 diagnostics.

**What to measure:**
- TP, Missed, FP (GT), No-GT on clip1 + RANSAC v2 online10
- Compare directly against v4+RANSAC (369/32/69/10)
- Expect: FP should drop (teleport run removed), TP should stay near 369

**Status:** Not started.

---

### ⬜ Step 2 — v6: Diagnose remaining FPs
After v5 baseline is established, dig into the remaining FP runs to decide
what to fix next. Options:
- Player-bbox suppression during KF interpolation (old v8 idea)
- Gate tuning
- RANSAC parameter tuning

**Status:** Blocked on Step 1.

---

## Notes / constraints
- Always test on clip1 AND clip2 minimum before committing to a change.
- Use `uv run script.py` (not `python`).
- Benchmark command template:
  ```
  uv run scripts/benchmark_physics_ransac_v2_clip1.py \
      --tracker_json <output.json> \
      --airborne_json ground_truths/clip1_actions.json \
      --run_name <name> \
      --online_lookahead_frames 10
  ```
- Update `ball_detection_metrics.csv` after each benchmark run.
- Don't overfit to clip1 — all logic changes must be explainable from first principles.
