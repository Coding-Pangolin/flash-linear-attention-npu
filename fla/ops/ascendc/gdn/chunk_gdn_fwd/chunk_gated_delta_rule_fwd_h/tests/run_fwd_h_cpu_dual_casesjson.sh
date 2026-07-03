#!/usr/bin/env bash
# fwd_h CPU dual for GPU-unsupported cases.json entries.
#
# 默认走 example dump 模型分布 + 单算子 dual（同 feat/fwd-h-gva-test）:
#   TEST_DEVICE_ID=2 ./run_fwd_h_cpu_dual_casesjson.sh
#   TEST_DEVICE_ID=2 ./run_fwd_h_cpu_dual_casesjson.sh --cases gva_fix_3
#
# 随机输入快速路径（不 dump）:
#   FWD_H_CPU_DUAL_RANDOM=1 TEST_DEVICE_ID=2 ./run_fwd_h_cpu_dual_casesjson.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
FLA_NPU_TEST="${REPO_ROOT}/torch_custom/fla_npu/test"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${FWD_H_CPU_DUAL_OUT:-${REPO_ROOT}/fla/ops/ascendc/gdn/dual_benchmark_logs/fwd_h/cpu_dual_casesjson/${TS}}"
MASTER_LOG="${OUT_DIR}/cpu_dual_batch.log"
DUMP_DIR="${GDN_FWD_H_DUMP_DIR:-${REPO_ROOT}/examples/fast_kernel_launch_example/tests/chunk_gated_delta_rule_fwd_h/data/casesjson}"

mkdir -p "${OUT_DIR}" "${DUMP_DIR}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate wnc
source /data/zs/run/8.5/ascend-toolkit/set_env.sh
source /data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer/bin/set_env.bash

export TEST_DEVICE_ID="${TEST_DEVICE_ID:-2}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}"

echo "[fwd_h cpu dual] device=${TEST_DEVICE_ID} out=${OUT_DIR}" | tee "${MASTER_LOG}"
echo "[fwd_h cpu dual] start $(date -Is)" | tee -a "${MASTER_LOG}"

if [[ "${FWD_H_CPU_DUAL_RANDOM:-0}" == "1" ]]; then
  echo "[fwd_h cpu dual] mode=random (run_fwd_h_cpu_dual_casesjson.py)" | tee -a "${MASTER_LOG}"
  python "${SCRIPT_DIR}/run_fwd_h_cpu_dual_casesjson.py" \
    --out-dir "${OUT_DIR}" \
    --device "${TEST_DEVICE_ID}" \
    "$@" 2>&1 | tee -a "${MASTER_LOG}"
else
  CASES_ARG=""
  SMOKE=0
  EXTRA=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --cases)
        CASES_ARG="$2"
        shift 2
        ;;
      --smoke)
        SMOKE=1
        EXTRA+=("$1")
        shift
        ;;
      *)
        EXTRA+=("$1")
        shift
        ;;
    esac
  done
  export FWD_H_SUITE=unsupported
  export GDN_FWD_H_DUMP_DIR="${DUMP_DIR}"
  export FWD_H_OUT_DIR="${OUT_DIR}"
  export FWD_H_VIZ="${FWD_H_VIZ:-1}"
  if [[ -n "${CASES_ARG}" ]]; then
    export FWD_H_CASE="${CASES_ARG}"
  fi
  if [[ "${SMOKE}" == "1" ]]; then
    echo "[fwd_h cpu dual] mode=example_dump smoke via random runner" | tee -a "${MASTER_LOG}"
    python "${SCRIPT_DIR}/run_fwd_h_cpu_dual_casesjson.py" \
      --out-dir "${OUT_DIR}" --device "${TEST_DEVICE_ID}" --smoke \
      "${EXTRA[@]}" 2>&1 | tee -a "${MASTER_LOG}"
  else
    echo "[fwd_h cpu dual] mode=example_dump suite=unsupported dump_dir=${DUMP_DIR}" | tee -a "${MASTER_LOG}"
    cd "${FLA_NPU_TEST}"
    python test_npu_fwd_h_gva.py 2>&1 | tee -a "${MASTER_LOG}"
  fi
fi

RC=${PIPESTATUS[0]}
echo "[fwd_h cpu dual] end $(date -Is) exit=${RC}" | tee -a "${MASTER_LOG}"
exit "${RC}"
