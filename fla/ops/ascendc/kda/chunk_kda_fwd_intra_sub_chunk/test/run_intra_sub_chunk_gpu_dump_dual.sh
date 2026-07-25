#!/usr/bin/env bash
# One-click: NPU vs GPU(+CPU dump) dual for chunk_kda_fwd_intra_sub_chunk.
#
# Usage:
#   ./run_intra_sub_chunk_gpu_dump_dual.sh /path/to/isub_gpu_dump
#   ./run_intra_sub_chunk_gpu_dump_dual.sh /path/to/isub_gpu_dump --names smoke_mha_fix,BSND_noGVA_V128_14
#   ./run_intra_sub_chunk_gpu_dump_dual.sh --pt /path/to/001_chunk_kda_fwd_intra_sub_chunk.pt
#
# Env:
#   TEST_DEVICE_ID   NPU id (default 0)
#   CANN_SET_ENV     optional; sourced via torch_custom setup if present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# test -> op -> kda -> ascendc -> ops -> fla -> repo
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

PY="${SCRIPT_DIR}/test_intra_sub_chunk_gpu_dump_dual.py"
SETUP="${REPO_ROOT}/torch_custom/fla_npu/test/setup_cann_env.sh"

export TEST_DEVICE_ID="${TEST_DEVICE_ID:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_isub_dual}"
mkdir -p "$MPLCONFIGDIR"

if [[ -f "$SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$SETUP" || true
elif [[ -n "${CANN_SET_ENV:-}" && -f "${CANN_SET_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV}"
fi

# Prefer conda env if already active; otherwise try wnc.
if ! python3 -c "import torch, torch_npu" 2>/dev/null; then
  if [[ -f /data/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate wnc 2>/dev/null || true
  fi
fi

if ! python3 -c "import torch, torch_npu, ct" 2>/dev/null; then
  echo "ERROR: need python with torch + torch_npu + ct (pip install ct)" >&2
  exit 2
fi

DUMP_ROOT=""
declare -a PY_ARGS=()

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <DUMP_ROOT> [extra python args...]" >&2
  echo "   or: $0 --pt <001_*.pt>" >&2
  exit 1
fi

if [[ "${1:-}" == *.pt || "${1:-}" == --pt ]]; then
  PY_ARGS=("$@")
  if [[ "${1:-}" == *.pt ]]; then
    DUMP_ROOT="$(cd "$(dirname "$1")" && pwd)"
    PY_ARGS=(--pt "$1" "${@:2}")
  elif [[ -n "${2:-}" ]]; then
    DUMP_ROOT="$(cd "$(dirname "$2")" && pwd)"
  fi
elif [[ -d "${1:-}" ]]; then
  DUMP_ROOT="$(cd "$1" && pwd)"
  shift
  PY_ARGS=(--dump-root "$DUMP_ROOT" "$@")
else
  PY_ARGS=("$@")
fi

LOG_DIR="${DUMP_ROOT:-.}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/intra_sub_chunk_gpu_dump_dual_${TS}.log"

echo "REPO_ROOT=$REPO_ROOT"
echo "TEST_DEVICE_ID=$TEST_DEVICE_ID"
echo "PY_ARGS=${PY_ARGS[*]}"
echo "LOG=$LOG"

set -o pipefail
python3 "$PY" "${PY_ARGS[@]}" 2>&1 | tee "$LOG"
ec=${PIPESTATUS[0]}
echo "exit=$ec log=$LOG"
exit "$ec"
