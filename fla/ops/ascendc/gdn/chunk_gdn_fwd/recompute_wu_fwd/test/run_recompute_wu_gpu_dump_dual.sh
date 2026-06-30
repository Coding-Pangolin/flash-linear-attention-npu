#!/usr/bin/env bash
# recompute_wu NPU vs GPU dual benchmark from GPU-collected dumps.
#
# Usage:
#   ./run_recompute_wu_gpu_dump_dual.sh [DUMP_ROOT] [extra args...]
#
# Examples:
#   ./run_recompute_wu_gpu_dump_dual.sh /path/to/GPU_DUMP
#   ./run_recompute_wu_gpu_dump_dual.sh ./GPU_DUMP --case phase_1_fix_1
#   ./run_recompute_wu_gpu_dump_dual.sh ./GPU_DUMP --phase prefix:phase_1_

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUMP_ROOT="${1:-./GPU_DUMP}"
shift || true

export TEST_DEVICE_ID="${TEST_DEVICE_ID:-0}"

exec python3 "${SCRIPT_DIR}/test_recompute_wu_gpu_dump_dual.py" \
  --dump-root "${DUMP_ROOT}" \
  "$@"
