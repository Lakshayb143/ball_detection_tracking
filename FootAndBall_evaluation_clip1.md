# FootAndBall Model Evaluation on clip1

## Summary

Evaluated the pre-trained FootAndBall model (2020 paper) on clip1. **Result: Poor performance.** The model significantly underperforms compared to your current v4 baseline tracker.

## Model Details

- **Source:** [FootAndBall: Integrated Player and Ball Detector](https://www.scitepress.org/Link.aspx?doi=10.5220/0008916000470056) (VISAPP 2020)
- **Training data:** ISSIA-CNR Soccer dataset + SoccerPlayerDetection_bmvc17
- **Architecture:** Efficient fully-convolutional network with Feature Pyramid design
- **Strengths claimed:** Real-time processing, small parameter count, handles arbitrary resolution

## Evaluation Results (clip1)

### FootAndBall Performance
| Metric | Value |
|---|---:|
| Total detections | 30 |
| TP (True Positives) | 29 |
| Missed | 441 |
| FP (False Positives) | 1 |
| **Recall** | **0.0617** |
| **Precision** | 0.9667 |
| **F1** | 0.1160 |

### v4 Baseline (for comparison)
| Metric | Value |
|---|---:|
| TP | 350 |
| Missed | 50 |
| FP | 80 |
| **Recall** | **0.8750** |
| **Precision** | 0.8140 |
| **F1** | 0.8439 |

## Key Findings

1. **Severe underfitting:** FootAndBall detects the ball in only **6.2%** of frames where it exists (recall: 0.0617)
   - Found 29 true positives out of 470 ground-truth frames
   - Missed 441 frames

2. **When it detects, it's accurate:** 96.7% precision on detections
   - Only 1 false positive on frames with no GT ball
   - But this precision is not useful given the low recall

3. **Domain mismatch:** The training domain (ISSIA-CNR 2020) appears significantly different from your clip1:
   - Different field type, lighting, camera angle?
   - Different ball size/contrast in footage?
   - Model trained on lower-resolution or different match conditions

4. **Your v4 baseline is 14x better:** 
   - v4 recall (87.5%) >> FootAndBall recall (6.2%)
   - Your current system is far superior for this application

## Recommendation

**Do not use FootAndBall.** Your RF-DETR-based v4/v5 system vastly outperforms it:
- v4 achieves 87.5% recall; FootAndBall only 6.2%
- v4 achieves 81.4% precision; FootAndBall 96.7% (but useless without recall)
- v4 F1 = 0.8439; FootAndBall F1 = 0.1160

### If you still want to explore external models:

Consider instead:
1. **Newer papers (2022+):** FootAndBall is from 2020. More recent ball trackers may have better generalization.
2. **Fine-tuning:** If you had time, fine-tuning FootAndBall on your domain might help, but v4/v5 are already excellent.
3. **Ensemble:** Could ensemble v4+other model if needed for redundancy, but v4 alone is strong.

## Detection Command

For reference, the evaluation was run with:
```bash
uv run python scripts/run_footandball_clip1.py
uv run python scripts/eval_footandball_clip1.py
```

Results saved to:
- Detections: `detections_footandball/clip1.json`
- Evaluation: `FootAndBall_evaluation_clip1.md`
