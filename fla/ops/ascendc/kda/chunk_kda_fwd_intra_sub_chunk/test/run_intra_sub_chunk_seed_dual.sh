#!/usr/bin/env bash
# One-click seed dual: CPU-RNG inputs (BTHD→BNSD), NPU vs local CPU. No dump copy.
#
# Usage:
#   ./run_intra_sub_chunk_seed_dual.sh --phase smoke
#   ./run_intra_sub_chunk_seed_dual.sh --phase all --seed 0
#   ./run_intra_sub_chunk_seed_dual.sh --names smoke_mha_fix,BSND_GVA_V256_28
#
# Must use same --seed / --phase / --names as GPU dump for input parity.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
PY="${SCRIPT_DIR}/test_intra_sub_chunk_seed_dual.py"
SETUP="${REPO_ROOT}/torch_custom/fla_npu/test/setup_cann_env.sh"

export TEST_DEVICE_ID="${TEST_DEVICE_ID:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_isub_seed_dual}"
mkdir -p "$MPLCONFIGDIR"

if [[ -f "$SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$SETUP" || true
elif [[ -n "${CANN_SET_ENV:-}" && -f "${CANN_SET_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV}"
fi

if ! python3 -c "import torch, torch_npu" 2>/dev/null; then
  if [[ -f /data/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate wnc 2>/dev/null || true
  fi
fi

if ! python3 -c "import torch, torch_npu, ct" 2>/dev/null; then
  echo "ERROR: need torch + torch_npu + ct" >&2
  exit 2
fi

OUT_DIR="${ISUB_SEED_DUAL_OUT:-./isub_seed_dual_out}"
mkdir -p "$OUT_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_DIR/logs/seed_dual_${TS}.log"

echo "REPO_ROOT=$REPO_ROOT TEST_DEVICE_ID=$TEST_DEVICE_ID"
echo "args: $*"
set -o pipefail
python3 "$PY" --out-dir "$OUT_DIR" "$@" 2>&1 | tee "$LOG"
ec=${PIPESTATUS[0]}
echo "exit=$ec log=$LOG"
exit "$ec"
