#!/usr/bin/env bash
# KDA chunk_kda_fwd NPU vs GPU dual benchmark from GPU-collected dumps.
#
# Usage:
#   ./run_kda_gpu_dump_dual.sh [DUMP_ROOT] [extra args...]
#   ./run_kda_gpu_dump_dual.sh /path/to/001_chunk_kda_fwd.pt
#   ./run_kda_gpu_dump_dual.sh --pt /path/to/001_chunk_kda_fwd.pt
#   TEST_DEVICE_ID=6 ./run_kda_gpu_dump_dual.sh /data/kda_dump/all --phase smoke --no-viz
#
# Logs: <DUMP_ROOT>/logs/kda_gpu_dump_dual_<timestamp>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KDA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../gpu_dump_dual_log.sh
source "${KDA_DIR}/gpu_dump_dual_log.sh"

export TEST_DEVICE_ID="${TEST_DEVICE_ID:-0}"
OP_TAG="kda"
PY="${SCRIPT_DIR}/test_kda_gpu_dump_dual.py"

DUMP_ROOT=""
declare -a PY_ARGS=()

if [[ $# -eq 0 ]]; then
  DUMP_ROOT="./GPU_DUMP"
  PY_ARGS=(--dump-root "$DUMP_ROOT" --isolated)
elif [[ "${1:-}" == *.pt ]]; then
  DUMP_ROOT="$(cd "$(dirname "$1")" && pwd)"
  PY_ARGS=(--pt "$1" "${@:2}")
elif [[ "${1:-}" == --pt || "${1:-}" == --pts ]]; then
  if [[ "${1:-}" == --pt && -n "${2:-}" ]]; then
    DUMP_ROOT="$(cd "$(dirname "$2")" && pwd)"
  fi
  PY_ARGS=("$@")
elif [[ -d "${1:-}" ]]; then
  DUMP_ROOT="$(cd "$1" && pwd)"
  shift
  PY_ARGS=(--dump-root "$DUMP_ROOT" --isolated "$@")
else
  PY_ARGS=("$@")
fi

LOG_DIR="$(gpu_dump_dual_log_dir "$DUMP_ROOT" "$SCRIPT_DIR")"
gpu_dump_dual_run_python "$PY" "$LOG_DIR" "$OP_TAG" "${PY_ARGS[@]}"
