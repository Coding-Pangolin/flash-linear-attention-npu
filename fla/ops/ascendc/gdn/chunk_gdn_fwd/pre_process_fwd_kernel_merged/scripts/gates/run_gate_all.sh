#!/bin/bash
# 一键跑全部门禁：L0 静态 → L1 位级（可对比基线）→ L2 smoke → L4 41 条 → L3 序列探针
#
# 用法（容器内，已装本算子 wheel）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   export ASCEND_RT_VISIBLE_DEVICES=7
#   TAG=after_p1a BASE=baseline_p1a bash run_gate_all.sh <op_dir> [rounds]
#
# 说明：L1 只在给了 BASE 时才比对；比对要求**逐元素完全一致**（改调度不改数值时才允许通过）。
set -u
OP=${1:?usage: run_gate_all.sh <op_dir> [rounds]}
ROUNDS=${2:-6}
TAG=${TAG:-run}
BASE=${BASE:-}
export PPFM_OP_DIR="$OP"

fail() { echo "!! $1 失败"; exit 1; }

echo "===== L0 静态门禁 ====="
python3 "$OP/scripts/gates/gate_l0.py" "$OP/op_kernel/pre_process_fwd_kernel_merged.cpp" || fail L0

echo "===== L1 位级 dump ($TAG) ====="
python3 "$OP/scripts/gates/dump_hm.py" "$OP" "$TAG" || fail "L1 dump"
if [ -n "$BASE" ]; then
  echo "===== L1 位级比对 ($BASE vs $TAG) ====="
  python3 "$OP/scripts/gates/cmp_hm.py" \
      "$OP/scripts/.cache/hm/$BASE" "$OP/scripts/.cache/hm/$TAG" || fail "L1 compare"
fi

echo "===== L2 smoke（含 max_abs<=0.05 判据）====="
python3 "$OP/scripts/npu_smoke_ppfm.py" --op-dir "$OP" || fail L2

echo "===== L4 全量 41 条（--repeats 2）====="
python3 "$OP/scripts/run_cases_ppfm.py" --repeats 2 || fail L4

echo "===== L3 序列探针（独立进程 ×$ROUNDS 轮）====="
python3 "$OP/scripts/gates/seq_probe.py" "$OP" "$ROUNDS" || fail L3

echo "GATE_ALL_DONE"
