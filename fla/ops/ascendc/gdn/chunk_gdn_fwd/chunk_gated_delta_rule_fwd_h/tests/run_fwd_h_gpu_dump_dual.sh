#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST_DEVICE_ID="${TEST_DEVICE_ID:-0}"
if [[ $# -eq 0 ]]; then
  exec python3 "${SCRIPT_DIR}/test_fwd_h_gpu_dump_dual.py" --dump-root ./GPU_DUMP
fi
if [[ "${1:-}" == *.pt ]]; then
  exec python3 "${SCRIPT_DIR}/test_fwd_h_gpu_dump_dual.py" --pt "$1" "${@:2}"
fi
if [[ "${1:-}" == --pt || "${1:-}" == --pts ]]; then
  exec python3 "${SCRIPT_DIR}/test_fwd_h_gpu_dump_dual.py" "$@"
fi
DUMP_ROOT="${1:-./GPU_DUMP}"
shift || true
exec python3 "${SCRIPT_DIR}/test_fwd_h_gpu_dump_dual.py" --dump-root "${DUMP_ROOT}" "$@"
