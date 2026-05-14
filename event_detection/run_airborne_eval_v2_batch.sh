#!/usr/bin/env bash
set -euo pipefail

# Batch wrapper for the v2 airborne event detector.
#
# Defaults:
#   outputs -> outputs/airborne_eval_v5_v2
#   python  -> repo .venv if present, otherwise python3
#
# Examples:
#   ./event_detection/run_airborne_eval_v2_batch.sh
#   ./event_detection/run_airborne_eval_v2_batch.sh --clips 16 --no-plots
#   AIRBORNE_V2_OUT_DIR=/tmp/airborne_v2 ./event_detection/run_airborne_eval_v2_batch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

OUT_DIR="${AIRBORNE_V2_OUT_DIR:-${ROOT_DIR}/outputs/airborne_eval_v5_v2}"

# Let an explicit --out-dir argument decide where artifacts/logs go.
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --out-dir)
      if ((i + 1 < ${#args[@]})); then
        OUT_DIR="${args[$((i + 1))]}"
      fi
      ;;
    --out-dir=*)
      OUT_DIR="${args[$i]#--out-dir=}"
      ;;
  esac
done

mkdir -p "${OUT_DIR}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${OUT_DIR}/run_${RUN_ID}.log"

echo "[airborne-v2] root:   ${ROOT_DIR}"
echo "[airborne-v2] python: ${PYTHON_BIN}"
echo "[airborne-v2] out:    ${OUT_DIR}"
echo "[airborne-v2] log:    ${LOG_PATH}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_airborne_eval_v2.py" \
  --out-dir "${OUT_DIR}" \
  "$@" 2>&1 | tee "${LOG_PATH}"

echo "[airborne-v2] done"
