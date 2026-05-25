# Ball Detection RANSAC Comparison Report

Precision is computed as `tp_iou_ge_0_01 / (tp_iou_ge_0_01 + false_positive_count)`, so `no_gt_count` is shown separately but not included in precision. Recall uses all `470` GT ball boxes.

## Without RANSAC

| Variant | tp_iou_ge_0_01 | missed_detections | false_positive_count | no_gt_count | distance_le_pred_bbox_threshold_count | precision | recall | f1_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_no_ransac | 314 | 84 | 72 | 10 | 30 | 0.8135 | 0.6681 | 0.7336 |
| v5_no_ransac | 360 | 44 | 66 | 10 | 22 | 0.8451 | 0.7660 | 0.8036 |
| v6_11_no_ransac | 353 | 94 | 23 | 4 | 8 | 0.9388 | 0.7511 | 0.8345 |

## With RANSAC

| Variant | tp_iou_ge_0_01 | missed_detections | false_positive_count | no_gt_count | distance_le_pred_bbox_threshold_count | precision | recall | f1_score | f1_change_vs_no_ransac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_with_ransac | 333 | 70 | 67 | 10 | 35 | 0.8325 | 0.7085 | 0.7655 | +0.0319 |
| v5_with_ransac | 378 | 44 | 48 | 10 | 26 | 0.8873 | 0.8043 | 0.8438 | +0.0402 |
| v6_11_with_ransac | 367 | 91 | 12 | 4 | 7 | 0.9683 | 0.7809 | 0.8645 | +0.0300 |
