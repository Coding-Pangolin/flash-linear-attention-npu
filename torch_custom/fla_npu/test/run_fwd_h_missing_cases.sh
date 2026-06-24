#!/bin/bash
# 补跑缺少 input dump 的 6 个用例：example dump + dual 精度比对
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${TEST_DEVICE_ID:-7}"
LOG="${FWD_H_MISSING_LOG:-/tmp/fwd_h_missing_cases.log}"

MISSING_CASES=(
  varlen_t16384_v128_cu2
  fixed_b16_t2048_v128
  varlen_t65536_v128_cu17
  varlen_t65536_v128_cu172
  varlen_t65536_v128_cu668
  varlen_t262144_v128_cu32
)

{
  echo "===== fwd_h missing cases dump+test ====="
  echo "device=$DEVICE start=$(date -Is)"
  for case in "${MISSING_CASES[@]}"; do
    echo ""
    echo "========== $case =========="
    TEST_DEVICE_ID="$DEVICE" FWD_H_CASE="$case" FWD_H_TEST_ONLY=0 FWD_H_VIZ=1 \
      bash "$SCRIPT_DIR/run_fwd_h_gva_cases.sh" || echo "[WARN] $case failed exit=$?"
  done
  echo ""
  echo "===== done $(date -Is) ====="
} 2>&1 | tee "$LOG"
