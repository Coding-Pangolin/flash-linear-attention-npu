#!/usr/bin/env bash
# bwd_dhu CPU dual benchmark for GPU-unsupported cases.json entries.
#
# Usage:
#   TEST_DEVICE_ID=2 ./run_bwd_dhu_cpu_dual_casesjson.sh --smoke
#   TEST_DEVICE_ID=2 ./run_bwd_dhu_cpu_dual_casesjson.sh --cases gva_fix_3,gva_var_2
#
# 环境：见 BWD_DHU_TEST.md / GDN_DUAL_TEST_GUIDE.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BWD_DHU_CPU_DUAL_OUT:-${REPO_ROOT}/fla/ops/ascendc/gdn/dual_benchmark_logs/bwd_dhu/cpu_dual_casesjson/${TS}}"
MASTER_LOG="${OUT_DIR}/cpu_dual_batch.log"

mkdir -p "${OUT_DIR}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate wnc
# shellcheck disable=SC1091
source "${REPO_ROOT}/torch_custom/fla_npu/test/setup_cann_env.sh"

export TEST_DEVICE_ID="${TEST_DEVICE_ID:-2}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}"

echo "[bwd_dhu cpu dual] device=${TEST_DEVICE_ID} out=${OUT_DIR}" | tee "${MASTER_LOG}"
echo "[bwd_dhu cpu dual] start $(date -Is)" | tee -a "${MASTER_LOG}"

python "${SCRIPT_DIR}/run_bwd_dhu_cpu_dual_casesjson.py" \
  --out-dir "${OUT_DIR}" \
  --device "${TEST_DEVICE_ID}" \
  "$@" 2>&1 | tee -a "${MASTER_LOG}"

RC=${PIPESTATUS[0]}
echo "[bwd_dhu cpu dual] end $(date -Is) exit=${RC}" | tee -a "${MASTER_LOG}"
exit "${RC}"
