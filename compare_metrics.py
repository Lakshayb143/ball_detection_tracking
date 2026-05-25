import json

baseline = json.load(open('baseline_metrics.txt'))
v6_11 = json.load(open('v6_11_metrics.txt'))

print("\n" + "="*70)
print("NO-RANSAC FAIR CLIP1 COMPARISON")
print("Shared: clip1, ball_samy_1120.pth, confidence=0.01, IoU=0.01")
print("="*70)
print(f"\n{'Metric':<30} {'Baseline':<15} {'V6_11':<15} {'Difference':<15}")
print("-"*70)

metrics = [
    ('TP (True Positives)', 'event_tp'),
    ('Missed (FN)', 'missed_detection_count'),
    ('FP (GT frames)', 'false_positive_count'),
    ('No-GT Predicted', 'no_gt_predicted'),
    ('FP (all)', 'fp_all'),
    ('Distance <= Threshold', 'distance_le_threshold_px_count'),
    ('Mapped Predictions', 'mapped_predictions'),
    ('Precision', 'precision'),
    ('Recall', 'recall'),
    ('F1 Score', 'f1_score'),
    ('Precision (FP all)', 'precision_fp_all'),
    ('F1 Score (FP all)', 'f1_score_fp_all'),
]

for name, key in metrics:
    b_val = baseline[key]
    v_val = v6_11[key]

    if isinstance(b_val, float):
        print(f"{name:<30} {b_val:<15.4f} {v_val:<15.4f} {v_val - b_val:<15.4f}")
    else:
        print(f"{name:<30} {int(b_val):<15} {int(v_val):<15} {int(v_val - b_val):<15}")

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)
print(f"\nBaseline (older ball_outlier_interpolator.py):")
print(f"  - Mapped Predictions: {int(baseline['mapped_predictions'])}")
print(f"  - TP: {int(baseline['event_tp'])}, Missed: {int(baseline['missed_detection_count'])}, FP-all: {int(baseline['fp_all'])}")
print(f"  - Recall: {baseline['recall']:.4f}, Precision: {baseline['precision']:.4f}")

print(f"\nV6_11 (newer ball_outlier_interpolator_v6_11.py):")
print(f"  - Mapped Predictions: {int(v6_11['mapped_predictions'])}")
print(f"  - TP: {int(v6_11['event_tp'])}, Missed: {int(v6_11['missed_detection_count'])}, FP-all: {int(v6_11['fp_all'])}")
print(f"  - Recall: {v6_11['recall']:.4f}, Precision: {v6_11['precision']:.4f}")

print(f"\nWinner by F1: V6_11 ({v6_11['f1_score']:.4f} vs {baseline['f1_score']:.4f})")
print("="*70 + "\n")
