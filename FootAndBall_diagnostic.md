# FootAndBall Diagnostic: Why Is It Missing Frames?

## Key Finding: NO SCALE BIAS

The model is NOT failing due to resolution/scale issues:

| Metric | TP (Detected) | Missed | Difference |
|---|---|---|---|
| **Ball area - Mean** | 259 px² | 375 px² | Detected balls are **0.69x SMALLER** |
| **Ball size - Min** | 13.1×13.1 | 9.0×9.0 | Detected: larger minimum |
| **Ball size - Max** | 22.4×22.4 | 33.3×33.3 | Detected: smaller maximum |

**What this means:**
- Not an input resolution problem (would see consistent size bias)
- Not an architecture limitation (would see all small balls missed)
- **Failure is due to appearance/domain mismatch, not scale**

## Detection Patterns

**Detected frame distribution:**
- Frames: 110, 116, 124 (early spikes), then 217, 349-479 (main cluster)
- Sparse across video (24 frames = 5% coverage)
- Frames 0-109 and 218-348: complete miss

**Detection confidence:**
- Range: 0.618 - 0.976
- Mean: 0.847 (high and stable)
- **Model is confident** → suggests it has learned something valid, not random firing

## Interpretation

✅ **Fine-tuning COULD help** because:
1. No systematic scale bias (model isn't blind to certain sizes)
2. High confidence (0.847 mean) suggests valid learned features
3. 80% precision on detections (only 6 FPs) confirms it learned something real
4. Failure pattern suggests appearance adaptation issue, not fundamental limitation

⚠️ **But consider:**
- 94% miss rate is severe (even if you improve to 50% with fine-tune, still < v5's 87.5%)
- RF-DETR v5 already at F1=0.89; diminishing returns on time spent
- Would need ~100-200 labeled examples to fine-tune properly

## Recommendation

**Visual inspection first:**
1. Look at the 24 frames in `footandball_tp_frames/` (GREEN=GT, RED=prediction)
2. Compare: do they have different lighting, ball color/contrast, or camera angle?
3. Look at a few missed frames manually - do they look similar to the ones detected?

If domain mismatch is obvious → **fine-tuning is worth trying**
If you can't spot a pattern → **architecture/resolution issue remains → skip fine-tune**
