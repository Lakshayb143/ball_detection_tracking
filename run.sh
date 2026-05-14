# 2) Main ablation matrix: 20 runs
declare -A OUTLIER_MODES=(
  [off]="--position_threshold 1000000 --velocity_threshold 1000000"
  [position_only]="--position_threshold 60 --velocity_threshold 1000000"
  [velocity_only]="--position_threshold 1000000 --velocity_threshold 40"
  [both]="--position_threshold 60 --velocity_threshold 40"
)

GDINO_CROP_ARGS="--gdino_crop_expansion 12 --gdino_min_crop_size 192 --gdino_max_crop_size 640 --gdino_full_frame_on_reset true"
SST_CROP_ARGS="--sst_crop_expansion 12 --sst_min_crop_size 192 --sst_max_crop_size 640 --sst_full_frame_on_reset true"

for mode in off position_only velocity_only both; do
  args="${OUTLIER_MODES[$mode]}"

  # A. RF-DETR + outlier only
  # one script is enough here; fallback is disabled, so GDINO/SST choice does not matter
  python scripts/benchmark_rfdetr_outlier_gdino_fullframe_fallback_tracking.py \
    --run_name "rfdetr_outlier_only_${mode}" \
    --enable_gdino_fallback false \
    $args

  # B. RF-DETR + outlier + GDINO full-frame fallback
  python scripts/benchmark_rfdetr_outlier_gdino_fullframe_fallback_tracking.py \
    --run_name "rfdetr_gdino_fullframe_${mode}" \
    --enable_gdino_fallback true \
    $args

  # C. RF-DETR + outlier + SST full-frame fallback
  python scripts/benchmark_rfdetr_outlier_sst_fullframe_fallback_tracking.py \
    --run_name "rfdetr_sst_fullframe_${mode}" \
    --enable_sst_fallback true \
    $args

  # D. RF-DETR + crop/local GDINO fallback
  python scripts/benchmark_rfdetr_groundingdino_fallback_tracking.py \
    --run_name "rfdetr_gdino_crop_${mode}" \
    $GDINO_CROP_ARGS \
    $args

  # E. RF-DETR + crop/local SST fallback
  python scripts/benchmark_rfdetr_sst_fallback_tracking.py \
    --run_name "rfdetr_sst_crop_${mode}" \
    $SST_CROP_ARGS \
    $args
done
