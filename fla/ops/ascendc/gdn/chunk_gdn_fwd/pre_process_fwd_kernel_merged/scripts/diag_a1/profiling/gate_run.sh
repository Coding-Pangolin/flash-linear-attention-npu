#!/bin/bash
# 241: L0 静态门禁 + L1 位级回归（当前 HEAD vs BASE commit）+ L2 升级版 smoke
#   BASE=8646248 TAG=head bash gate_run.sh
set -u
PY=$HOME/BartonFang/envs/fzy/bin/python
REPO=$HOME/BartonFang/flash-linear-attention-npu
OP=$REPO/fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged
K=$OP/op_kernel/pre_process_fwd_kernel_merged.cpp
W=/tmp/ppfm241/prof
BASE=${BASE:-8646248}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=${DEV:-7} PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore PPFM_REF_THREADS=2

echo "=== [1/5] L0 静态门禁 ==="
"$PY" "$OP/scripts/gates/gate_l0.py" "$K"

echo "=== [2/5] L1 dump: 当前 HEAD ==="
cd "$OP" || exit 1
"$PY" /tmp/ppfm241/run_npu.py "$OP/scripts/gates/dump_hm.py" "$OP" new

echo "=== [3/5] L1 dump: base $BASE（重编 + 装包）==="
cd "$REPO" || exit 1
git checkout -- "$K"
git checkout "$BASE" -- "$K"
grep -c ITER7 "$K" || true
TAG=base bash "$W/build_install.sh" || exit 1
cd "$OP" || exit 1
"$PY" /tmp/ppfm241/run_npu.py "$OP/scripts/gates/dump_hm.py" "$OP" base

echo "=== [4/5] L1 位级比对（要求 BIT_IDENTICAL）==="
"$PY" "$OP/scripts/gates/cmp_hm.py" "$OP/scripts/.cache/hm/new" "$OP/scripts/.cache/hm/base"

echo "=== [5/5] 恢复 HEAD 并重编 + L2 升级版 smoke ==="
cd "$REPO" || exit 1
git checkout HEAD -- "$K"
TAG=head bash "$W/build_install.sh" || exit 1
cd "$OP" || exit 1
"$PY" /tmp/ppfm241/run_npu.py scripts/npu_smoke_ppfm.py --op-dir "$OP" 2>&1 | tail -45
echo GATE_RUN_DONE
