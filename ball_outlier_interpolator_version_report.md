# Ball Outlier Interpolator Version Report

## Scope

This report covers every repo file matching `ball_outlier_interpolator*.py`: the original script, the baseline script, v2 through v6_11, and the `v4_no_interpolation` ablation. The summaries are based on the current source code plus available run artifacts such as `ball_detection_metrics.csv`, `baseline_metrics.txt`, `v6_11_metrics.txt`, and the v6 run logs. Metrics labeled `with_ransac` include the downstream RANSAC repair stage, so they should be read as pipeline results rather than raw interpolator-only results.

## Evolution At A Glance

The project evolved from a simple RF-DETR plus threshold tracker into a physics-aware Kalman tracker with a series of targeted false-positive suppressors. The most stable core appears in v2/v5 onward: low detector confidence, Mahalanobis gating, 4-frame outlier confirmation, gravity-aware Kalman prediction, and JSON export for evaluation. Later v6 versions are mostly ablations around what to do with interpolated positions when the detector misses the ball.

| Version family | Main idea | Main tradeoff |
| --- | --- | --- |
| Original and baseline | Position/velocity thresholds plus short interpolation | Simple but brittle and detector-confidence sensitive |
| v2 | Gravity-aware Kalman filter with Mahalanobis gating | Strong core, but interpolation can hallucinate during gaps |
| v3 | Action-aware rejection using event regimes | Useful concept, depends heavily on action labels and calibration |
| v4 | Player detector and action phases, with phase toggles | Good ablation framework, phases are disabled by default |
| v4_no_interpolation | Remove interpolation entirely | Eliminates false positives but misses many ball frames |
| v5 | v4/v2 core on clip1 with reset-distance cap and interpolated JSON records | Best F1 with RANSAC, but still many false positives |
| v6.x | Iterative filters on interpolated positions | Improves precision, often sacrifices recall |

## Benchmark Snapshot

Available CSV results show that v5 with RANSAC has the highest F1 among the listed rows, while v6_11 is the most precision-focused later variant.

| Variant | TP | Missed | FP | No-GT pred | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5_with_ransac | 378 | 44 | 48 | 10 | 0.8915 |
| v6_6_with_ransac | 370 | 68 | 32 | 7 | 0.8810 |
| v6_5_with_ransac | 370 | 69 | 31 | 3 | 0.8804 |
| v4_with_ransac | 369 | 32 | 69 | 10 | 0.8796 |
| v6_8_with_ransac | 368 | 84 | 18 | 3 | 0.8783 |
| v6_10_with_ransac | 368 | 68 | 34 | 3 | 0.8783 |
| v6_11_with_ransac | 367 | 91 | 12 | 6 | 0.8770 |
| v6_7_with_ransac | 364 | 89 | 17 | 3 | 0.8729 |
| v6_3_with_ransac | 363 | 89 | 18 | 3 | 0.8715 |
| v6_9_with_ransac | 356 | 92 | 22 | 3 | 0.8620 |
| v4_with_interpolation | 350 | 32 | 88 | 10 | 0.8537 |
| v4_no_interpolation | 349 | 121 | 0 | 0 | 0.8523 |
| old_version_no_ransac_fair_clip1 | 314 | 84 | 72 | 10 | 0.8010 |
| v6_11_no_ransac_fair_clip1 | 353 | 94 | 23 | 4 | 0.8578 |

## Version Notes

### `ball_outlier_interpolator.py`

Approach:
- Uses `RFDETRMedium` with a high default confidence threshold of `0.8`.
- Tracks detections with an adaptive constant-acceleration Kalman filter and rolling position/velocity buffers.
- Marks outliers by fixed Euclidean position and velocity thresholds, then waits for several bad frames before resetting.
- Interpolates only short gaps through a separate interpolation tracker capped at four frames.

Finding: This is a useful prototype, but the high detector threshold and fixed pixel gates make it fragile for broadcast soccer, where the ball can be small, blurred, or move very quickly.

### `ball_outlier_interpolator_baseline.py`

Approach:
- Ports the original prototype to the `ball_samy_1120.pth` model, `clips/clip1.mp4`, and a low `0.01` confidence threshold.
- Keeps the original adaptive Kalman filter, rolling outlier detector, and short interpolation tracker.
- Adds JSON output so the older approach can be evaluated against later versions.
- Acts as the fairer baseline for comparing the old threshold-based idea against the newer Kalman/Mahalanobis family.

Finding: It preserves the original design while controlling for model and confidence differences, which is important because earlier apparent failures could come from detector setup rather than tracker logic alone.

### `ball_outlier_interpolator_v2.py`

Approach:
- Replaces fixed position thresholds with a gravity-aware `BallKalmanFilter` using state `[x, y, vx, vy, ax, ay]`.
- Uses Mahalanobis distance against the filter covariance to select in-gate detections.
- Interpolates from the Kalman prediction for up to `13` missing/outlier frames.
- Requires four consecutive outlier frames before resetting the track.

Finding: v2 is the first strong core design. Its low detector threshold improves recall, but the same choice makes false-positive control much more important.

### `ball_outlier_interpolator_v3.py`

Approach:
- Adds an action timeline and `ActionConditionalFilter` driven by event labels such as High Pass, Shot, Pass, Drive, and Out.
- Applies regime-specific plausibility rules, including airborne ground rejection and large-displacement penalties.
- Keeps the v2 Kalman/Mahalanobis tracker underneath the action filter.
- Introduces per-action max-gap ideas, especially longer gaps for airborne events and zero gap for Out.

Finding: The idea is strong, but it depends on accurate action annotations and carefully calibrated camera assumptions. Without reliable player/foot context, some action rules are only partial.

### `ball_outlier_interpolator_v4.py`

Approach:
- Moves to the `ball_samy_1120.pth` ball model and adds a separate `player.pth` detector.
- Introduces Phase 1: reject ball detections inside player boxes during airborne regimes.
- Introduces Phase 2: change Kalman process noise and max-gap by action regime.
- Both phases are present but disabled by default, so the default run mostly tests the baseline tracker on new data paths.

Finding: v4 is mainly an ablation framework. The metrics show interpolation recovers many frames but adds false positives: `v4_with_interpolation` has 32 missed detections and 88 FP, while `v4_no_interpolation` has 121 missed detections and 0 FP.

### `ball_outlier_interpolator_v4_no_interpolation.py`

Approach:
- Removes Kalman interpolation from output and only records accepted detector observations.
- Uses a simple `100` px Euclidean gate from the last accepted position instead of Mahalanobis covariance gating.
- Keeps four-frame outlier confirmation and optional action/player phase hooks.
- Outputs `None` during gaps instead of creating predicted ball positions.

Finding: This ablation proves interpolation is the main source of false positives. It reaches 0 FP in the available table, but missed detections jump to 121.

### `ball_outlier_interpolator_v5.py`

Approach:
- Returns to the gravity-aware Kalman/Mahalanobis tracker on `clip1`.
- Adds `RESET_MAX_DISTANCE = 300` so four consecutive outliers cannot reset onto an implausibly distant object.
- Saves interpolated positions in JSON with `source: "interpolation"` for downstream RANSAC repair.
- Switches frame counting to 1-indexing to align with COCO-style ground-truth annotations.

Finding: v5 is the best listed pipeline by F1 when paired with RANSAC: `0.8915`. It has strong recall, but its 48 false positives show the need for later v6 suppression work.

### `ball_outlier_interpolator_v6.py`

Approach:
- Starts the v6 line by adding a green-ground rejection filter for interpolated positions.
- Suppresses a predicted position when the local HSV crop is at least `80%` grass-green.
- Leaves accepted detector boxes untouched and only filters Kalman-generated interpolation outputs.
- Keeps v5 reset-distance protection and the disabled action phase framework.

Finding: This directly targets empty-grass hallucinations. The weakness is that soccer balls often sit on grass, so green rejection can remove true ball positions if the crop is not sized well.

### `ball_outlier_interpolator_v6_1.py`

Approach:
- Tightens the green threshold to `95%` so only very pure grass crops are rejected.
- Uses a smaller `10x10` center crop around the predicted point.
- Keeps the same Kalman, reset, phase, and JSON logic as v6.
- Tests whether local color at the exact predicted center is enough to distinguish ball from grass.

Finding: This is less aggressive than v6, but it is more sensitive to small localization errors because a tiny crop can miss the visible ball edge.

### `ball_outlier_interpolator_v6_2.py`

Approach:
- Keeps the stricter `95%` green threshold from v6_1.
- Returns to a larger `20x20` crop around the interpolated position.
- Maintains the same v5/v6 tracker and post-interpolation filter structure.
- Serves as a fixed-window crop-size ablation against v6_1.

Finding: v6_2 shows that crop support size is a major hidden parameter. Larger crops are more stable but can mix ball pixels and background pixels in ways that weaken a simple green fraction test.

### `ball_outlier_interpolator_v6_3.py`

Approach:
- Replaces fixed crop size with a dynamic crop radius based on the most recent accepted ball bbox.
- Uses `1.5 * max(width, height) / 2` as the crop radius, with a fallback radius of `10`.
- Keeps the `95%` grass-green rejection threshold.
- Logs green rejections separately for diagnostics.

Finding: Dynamic crop sizing is a better design than fixed windows. With RANSAC, v6_3 has low FP count at 18 but misses 89 detections, showing that green-only suppression is still too recall-expensive.

### `ball_outlier_interpolator_v6_4.py`

Approach:
- Replaces green rejection with a white/light-patch acceptance test for interpolated positions.
- Uses HSV low-saturation, high-value pixels to decide whether a crop is ball-like.
- Rejects interpolation if the crop is not white/light enough.
- Tracks crop radius from the last accepted bbox, but the current threshold/comment/log wording is inconsistent.

Finding: The white filter is too brittle for real broadcast frames because lighting, blur, compression, shadows, and ball texture change the visible white fraction. Available logs show heavy suppression of interpolations.

### `ball_outlier_interpolator_v6_5.py`

Approach:
- Switches from green/white masks to average saturation in HSV space.
- Rejects interpolated crops with high saturation, targeting blue jerseys, saturated grass, and colored false positives.
- Keeps dynamic crop sizing from the last accepted detection bbox.
- Leaves accepted detections unchanged and only suppresses interpolated outputs.

Finding: Saturation is a better general cue than explicit green or white. With RANSAC, v6_5 reaches F1 `0.8804` with 31 FP and 69 missed detections, a more balanced result than green-only variants.

### `ball_outlier_interpolator_v6_6.py`

Approach:
- Adds stationary-ball detection before Kalman prediction.
- If the last tracked position moved less than `5` px, it zeros the Kalman velocity components.
- Keeps v6_5 saturation rejection for interpolated positions.
- Targets cases where a stopped ball causes the Kalman filter to keep drifting forward during a detector gap.

Finding: This is one of the cleanest additions. It is behaviorally grounded and gives the best v6-family F1 in the CSV at `0.8810`, with 29 stationary events reported in the available run log.

### `ball_outlier_interpolator_v6_7.py`

Approach:
- Combines stationary-ball velocity zeroing with the green-ground filter instead of the saturation filter.
- Uses a more permissive green threshold of `85%` with dynamic crop sizing.
- Keeps the same reset cap, Mahalanobis gate, and disabled action phases.
- Tests whether stationary handling can make the earlier green approach competitive.

Finding: It achieves a low FP count of 17 with RANSAC, but missed detections remain high at 89. The green cue still tends to over-suppress useful interpolations.

### `ball_outlier_interpolator_v6_8.py`

Approach:
- Returns to saturation filtering plus stationary-ball handling.
- Adds an extrapolation-distance cap: reject interpolation if the predicted point is more than `100` px from the last known position.
- Applies the cap during detector gaps before writing an interpolated point.
- Targets long Kalman drift when velocity or direction becomes stale.

Finding: This is a good targeted guard. It keeps FP low at 18, but missed detections rise to 84; the run log reports 21 extrapolation-cap rejections.

### `ball_outlier_interpolator_v6_9.py`

Approach:
- Replaces the v6_8 extrapolation cap with a global height rejection rule.
- Rejects interpolations in the top portion of the frame using `HEIGHT_REJECTION_THRESHOLD = 0.4`.
- Keeps saturation and stationary-ball handling.
- Targets upper-frame player-head/jersey false positives using a simple spatial prior.

Finding: The rule is too coarse for broadcast soccer because valid balls can appear high in the image during aerial play or camera perspective changes. Its F1 drops to `0.8620`, below v6_8 and v6_6.

### `ball_outlier_interpolator_v6_10.py`

Approach:
- Removes global height rejection and adds a color-variance check.
- Rejects an interpolated crop if it has high saturation or very low color standard deviation.
- Keeps stationary-ball handling, but does not include the v6_8 extrapolation cap.
- Targets uniform empty patches where no ball texture is visible.

Finding: The idea improves recall compared with v6_8 but raises FP count. The CSV shows F1 `0.8783`, 68 missed detections, and 34 FP, so low variance alone is not discriminative enough.

### `ball_outlier_interpolator_v6_11.py`

Approach:
- Combines saturation rejection, stationary-ball velocity zeroing, and the v6_8 extrapolation-distance cap.
- Adds player-top rejection: suppress interpolation if it lands inside the top `40%` of a detected player bbox.
- Runs the player detector when player-top rejection is enabled, even though Phase 1 remains disabled.
- Drops v6_10 color variance and avoids the v6_9 global height rule.

Finding: v6_11 is precision-focused. In the fair no-RANSAC comparison it improves F1 from `0.8010` to `0.8578` and reduces FP-all from 82 to 27 versus the old version, but with RANSAC it misses 91 detections and lands at F1 `0.8770`.

## Cross-Version Findings

1. Interpolation is the central recall/precision lever. The `v4_no_interpolation` ablation has 0 FP but 121 missed detections, while interpolation-heavy versions recover many frames but add hallucinated ball positions.

2. v2 established the most important tracker foundation. Gravity-aware Kalman prediction, Mahalanobis gating, and four-frame outlier confirmation are reused by nearly every strong later variant.

3. v5 is the best listed RANSAC-assisted pipeline by F1. It has the strongest balance in `ball_detection_metrics.csv`, but its false positives motivate the v6 suppression experiments.

4. v6_6 is the cleanest v6 improvement. Stationary-ball velocity zeroing is physically meaningful, low-risk, and slightly improves the v6 family while preserving the saturation filter.

5. Color heuristics are useful but fragile. Green, white, saturation, and variance filters all depend on crop size, lighting, jersey colors, shadows, and camera exposure; dynamic crop sizing is preferable to fixed windows.

6. Targeted spatial rejection beats global spatial rejection. The v6_8 extrapolation cap and v6_11 player-top rejection are more defensible than v6_9's global top-frame rule.

7. Phase 1 and Phase 2 remain under-tested in the current default configurations. Because `ENABLE_PHASE_1` and `ENABLE_PHASE_2` are usually `False`, most reported gains come from interpolation filtering rather than action-regime dynamics.

8. The codebase should standardize metric naming and configs. Some comments/logs drift from current constants, and some rows represent raw scripts while others include RANSAC, so future reports should separate raw tracker, tracker-plus-filter, and tracker-plus-RANSAC results.

## Practical Recommendation

Use v5 as the high-recall baseline and v6_11 as the high-precision baseline. For the next iteration, start from v6_6 or v6_11, then tune interpolation suppression with a validation split that reports raw tracker metrics and RANSAC-assisted metrics separately. The next most valuable experiment is not another standalone color rule; it is a combined confidence policy that decides when to interpolate based on Kalman uncertainty, last accepted bbox size, saturation, distance from last position, and nearby player overlap.
