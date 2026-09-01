# Airborne Detection — State, Assumptions, Failure Modes, Production Gaps

## 1. Current State (where we are)

**Pipeline:** Ball detector → 30-frame rolling features → weighted-score rule (direct launch + reappearance launch) → state machine (GROUND → CANDIDATE → AIRBORNE) → events.

**Best stable version**: v3_1 (hybrid baseline). Frozen because v3 (player baseline replacement) and v3_2 (velocity-stationarity) both regressed.

**Measured performance (6 clips with player detections):**

|           | clip1 | clip2 | clip3 | clip4 | clip5 | testing_1080 |
| ---       | ---   | ---   | ---   | ---   | ---   | ---          |
| recall    | 1.0   | 1.0   | 1.0   | **0.0** | 1.0 | 0.6          |
| precision | 0.75  | 1.0   | ?     | —     | ?     | 0.857        |
| end_err   | 27f   | 19f   | ?     | —     | ?     | **106f**     |

Macro recall ~0.86 reported in CONTEXT.md (8 clips), but only 17 clips ever evaluated. **No held-out test set.**

## 2. How It Works (signals → decision)

**Launch fires when ANY of:**
- Direct: `dy ≤ -8 px/f` AND recent detection density ≥ 0.3, weighted sum ≥ 1.2
- Reappearance: ≥2 missing frames, ball reappears ≥45 px higher, weighted sum ≥ 1.0
- Plus consistency check: ≥2 of last 3 frames upward

**Landing fires when:**
- Ball near baseline (|y − baseline_y| < 50 px) for 2 consecutive frames
- AND ascent so far ≥ 60 px above baseline

**Baseline = rolling median of last 120 ground detections.**

## 3. Implicit Assumptions (the cracks)

All in fixed pixel units and image-space — none scale-invariant:

| Assumption                          | What breaks it in production                                                              |
| ---                                 | ---                                                                                       |
| `dy ≤ -8 px/f` is "upward"          | Different frame rate (60fps slow-mo halves dy), different zoom level, different resolution |
| `|y − baseline| < 50 px` is "landed"| Tighter zoom (smaller pixel ground = real ball lands far in pixels), wider shot (ball never gets close enough to baseline in px) |
| `ascent ≥ 60 px` is meaningful      | Same — distance from camera changes pixel ascent for identical real-world height          |
| Image y-axis ≈ vertical             | Tilted broadcast angles, behind-goal cameras, drone shots break this                     |
| Camera is stable                    | Pans/zooms shift baseline; testing_1080 already shows the failure                        |
| Single ball                         | Training-drill scenes, ball-out-of-bounds with replacement ball will confuse detector    |
| Player feet visible                 | Tight ball-only zooms, overhead shots → no player baseline fallback                       |
| Detector reliably sees ball ≥30% of time | clip4 detector density is near-zero during airborne; no rule fix possible            |
| Frame stream is contiguous          | Scene cuts, replay inserts, broadcast graphics overlays corrupt state machine            |
| Events <7 seconds                   | Slow-mo replays exceed `max_airborne_frames=180`                                          |

## 4. Concrete Failure Modes (what we've observed)

1. **Stale baseline on camera pan** (testing_1080): deque locks early; events at different ground levels never satisfy landing. End errors 100+ frames.
2. **Detector blackout** (clip4): 0% recall, no airborne ever fires because ball is undetected throughout flight. **Not a rule problem.**
3. **Premature landing** (some clip5 events): rule fires landing when ball passes near baseline mid-arc.
4. **Late/missing landing** (clip2, clip3): landing never satisfies; max_duration fallback kicks in at 180f → huge end errors.
5. **GT boundary disagreement** (per CONTEXT.md): ~5 of 7 historical FPs are unlabeled real events or boundary mismatches, not actual system errors.

**The real picture**: failure modes (1), (3), (4) are interconnected. They are all "is the ball near the ground right now" failing because the ground reference is wrong, too rigid, or in the wrong pixel range.

## 5. What's Missing for Production Robustness

In rough priority order:

### A. Diagnostic instrumentation (do this first)
We don't currently separate **"detector failed"** from **"rule failed"** in evaluation output. Without this, every failure looks the same and we'll keep tweaking rules when the problem is detection coverage. Need a per-event diagnostic: detection density during GT window, ascent-in-pixels, detector confidence distribution.

### B. Scale invariance
All thresholds (8 px/f, 50 px, 60 px ascent) should be normalized — to **player bbox height** (already computed), to ball diameter (from detector), or to camera-frame motion statistics. Today's thresholds were tuned for ~1080p broadcast with ~10-px ball; they will silently fail on different framing.

### C. Camera-motion signal
Use the player bboxes already loaded to estimate per-frame camera translation/zoom. When camera is panning fast, (a) baseline must be reset/recomputed, (b) "upward velocity" thresholds must account for global y-motion. Right now camera motion contaminates every signal silently.

### D. Held-out evaluation
17 clips total, all used for tuning + reporting. There's **no honest measure** of how the system generalizes. Need to split: e.g., 12 dev clips, 5 truly-held-out test clips that are never looked at until acceptance. Without this, every "improvement" risks overfitting.

### E. Confidence calibration
Current "score" is an unbounded sum of weights. Not interpretable; cannot threshold for precision/recall trade-off downstream. Two events scoring 1.3 and 2.0 are not 1.5× different in confidence.

### F. Scene-cut / context handling
Broadcast video has cuts, replays, graphic overlays. State machine has no notion of these. Need at minimum a "detection blackout = reset to GROUND" rule, and ideally an explicit cut detector.

### G. Detector quality is the ceiling
clip4 is the canonical reminder: when the detector misses the ball mid-flight, no rule fixes it. Production robustness on unseen matches will be **bounded by detector recall in airborne conditions** (motion blur, small ball, sky/crowd backgrounds). Worth measuring detector-side recall on airborne frames separately.

## 6. Honest Recommendation

The current path of iterating on landing logic (v3, v3_1, v3_2) is **micro-tuning a system that has macro-level structural gaps**. We will keep getting mixed results because:
- We're tuning on the same 6 clips we evaluate on
- We have no signal for whether failures are detector or rule
- All thresholds are pixel-absolute on broadcast 1080p — they will not transfer

**Before more rule changes, do these in order:**
1. **Build per-event diagnostic output** (detection density, max ascent, baseline divergence per GT span). 1-day task. Will redirect future work.
2. **Hold out 3–5 clips**. Never look at them while iterating. Use only for accept/reject decisions.
3. **Normalize thresholds by player bbox height**. Single change; immediately gives a degree of scale invariance.
4. **Add camera-motion estimator** (already free with player bboxes loaded). Use it to (a) reset baseline on pan, (b) compensate for global y-shift in dy thresholds.

After 1–3, *then* revisit landing logic. The right landing rule will probably emerge naturally once baseline is reliable and thresholds are normalized.

**Items to defer until detector improves:** clip4-style total-blackout failures. Worth measuring detector-airborne recall as a separate metric so we don't conflate it with rule failures.
