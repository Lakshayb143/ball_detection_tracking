# FootAndBall Model Evaluation on clip1

## Standalone Metrics

**Model:** Pre-trained FootAndBall (2020 paper)
**Dataset:** ISSIA-CNR Soccer + SoccerPlayerDetection_bmvc17
**Test clip:** clip1 (521 frames, 470 GT frames with ball)
**Matching criterion:** IoU >= 0.01

### Results

| Metric | Value |
|---|---:|
| **TP** | 24 |
| **Missed** | 441 |
| **FP (GT frames)** | 5 |
| **No-GT predicted** | 1 |
| **FP (all)** | 6 |
| **Recall** | 0.0511 |
| **Precision** | 0.8000 |
| **F1** | 0.0960 |

Total predictions: 30 detections across 30 frames

## Analysis

- **Very low recall:** Only detected 24 of 470 ground-truth ball frames (5.1%)
- **High precision when detected:** 80% of predictions match GT with IoU >= 0.01
- **Severely underfitting:** Missed 441 frames (~94% miss rate)
- **Few false positives:** 6 FP total (5 on GT frames, 1 on frames without GT)

## Evaluation Command

```bash
uv run python scripts/eval_footandball_clip1.py
```

Detections saved to: `detections_footandball/clip1.json`
