#!/usr/bin/env bash
# recompute_wu NPU vs GPU dual benchmark from GPU-collected dumps.
#
# Usage:
#   ./run_recompute_wu_gpu_dump_dual.sh [DUMP_ROOT] [extra args...]
#   ./run_recompute_wu_gpu_dump_dual.sh /path/to/004_recompute_wu.pt
#   ./run_recompute_wu_gpu_dump_dual.sh --pt /path/to/004_recompute_wu.pt
#
# Logs: <DUMP_ROOT>/logs/recompute_wu_gpu_dump_dual_<timestamp>.log
#       (or ./logs/ when using --pt only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GDN_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=../../gpu_dump_dual_log.sh
source "${GDN_DIR}/gpu_dump_dual_log.sh"

export TEST_DEVICE_ID="${TEST_DEVICE_ID:-0}"
OP_TAG="recompute_wu"
PY="${SCRIPT_DIR}/test_recompute_wu_gpu_dump_dual.py"

DUMP_ROOT=""
declare -a PY_ARGS=()

if [[ $# -eq 0 ]]; then
  DUMP_ROOT="./GPU_DUMP"
  PY_ARGS=(--dump-root "$DUMP_ROOT")
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
  PY_ARGS=(--dump-root "$DUMP_ROOT" "$@")
else
  PY_ARGS=("$@")
fi

LOG_DIR="$(gpu_dump_dual_log_dir "$DUMP_ROOT" "$SCRIPT_DIR")"
gpu_dump_dual_run_python "$PY" "$LOG_DIR" "$OP_TAG" "${PY_ARGS[@]}"
