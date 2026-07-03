#!/usr/bin/env bash
# recompute_wu CPU dual benchmark for GPU-unsupported cases.json entries.
#
# Usage:
#   TEST_DEVICE_ID=2 ./run_recompute_wu_cpu_dual_casesjson.sh --smoke
#   TEST_DEVICE_ID=2 ./run_recompute_wu_cpu_dual_casesjson.sh --cases gva_fix_3,gva_var_2
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${RECOMPUTE_WU_CPU_DUAL_OUT:-${REPO_ROOT}/fla/ops/ascendc/gdn/dual_benchmark_logs/recompute_wu/cpu_dual_casesjson/${TS}}"
MASTER_LOG="${OUT_DIR}/cpu_dual_batch.log"

mkdir -p "${OUT_DIR}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate wnc
source /data/zs/run/8.5/ascend-toolkit/set_env.sh
source /data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer/bin/set_env.bash

export TEST_DEVICE_ID="${TEST_DEVICE_ID:-2}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}"

echo "[recompute_wu cpu dual] device=${TEST_DEVICE_ID} out=${OUT_DIR}" | tee "${MASTER_LOG}"
echo "[recompute_wu cpu dual] start $(date -Is)" | tee -a "${MASTER_LOG}"

python "${SCRIPT_DIR}/run_recompute_wu_cpu_dual_casesjson.py" \
  --out-dir "${OUT_DIR}" \
  --device "${TEST_DEVICE_ID}" \
  "$@" 2>&1 | tee -a "${MASTER_LOG}"

RC=${PIPESTATUS[0]}
echo "[recompute_wu cpu dual] end $(date -Is) exit=${RC}" | tee -a "${MASTER_LOG}"
exit "${RC}"
