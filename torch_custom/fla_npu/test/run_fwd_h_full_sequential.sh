#!/bin/bash
# 全量 fwd_h GVA 用例逐个跑（dump 到 fwd_h 前即退出 + 单算子精度），默认 device 5
# 用例按预估耗时从快到慢排序
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="${FWD_H_FULL_LOG:-/tmp/fwd_h_full_sequential.log}"
DEVICE="${TEST_DEVICE_ID:-5}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate wnc
source /data/zs/run/8.5/ascend-toolkit/set_env.sh
if [[ -f /data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer/bin/set_env.bash ]]; then
  source /data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer/bin/set_env.bash
fi

export TEST_DEVICE_ID="$DEVICE"
export GDN_FWD_H_DUMP_EXIT=1
export PYTHONUNBUFFERED=1

# 快 → 慢（B711 在 python 侧 SKIP）
CASES=(
  fixed_b176_t24_v128
  fixed_t4096_v128
  fixed_b16_t2048_v128
  varlen_t16384_v128_cu2
  varlen_t16384_v128_cu128
  varlen_t65536_v128_cu17
  varlen_t65536_v128_cu172
  varlen_t65536_v128_cu668
  varlen_t262144_v128_cu32
)

echo "===== fwd_h full sequential run device=$DEVICE (fast→slow) =====" | tee "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"
echo "GDN_FWD_H_DUMP_EXIT=1 (dump 后跳过 fwd_h/fwd_o/bwd)" | tee -a "$LOG"

cd "$SCRIPT_DIR"
PASS=0
FAIL=0
for case in "${CASES[@]}"; do
  echo "" | tee -a "$LOG"
  echo ">>> [$(date -Is)] CASE=$case" | tee -a "$LOG"
  t0=$(date +%s)
  if FWD_H_CASE="$case" python3 test_npu_fwd_h_gva.py 2>&1 | tee -a "$LOG"; then
    echo ">>> $case: PASS ($(( $(date +%s) - t0 ))s)" | tee -a "$LOG"
    PASS=$((PASS + 1))
  else
    echo ">>> $case: FAIL ($(( $(date +%s) - t0 ))s)" | tee -a "$LOG"
    FAIL=$((FAIL + 1))
  fi
done

echo "" | tee -a "$LOG"
echo "===== DONE $(date -Is) PASS=$PASS FAIL=$FAIL =====" | tee -a "$LOG"
exit "$FAIL"
