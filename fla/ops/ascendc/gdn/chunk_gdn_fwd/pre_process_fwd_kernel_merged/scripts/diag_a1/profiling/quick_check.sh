#!/bin/bash
# 快速检查：编译 + 装包 + L0 + smoke（10 形状）
set -u
PY=$HOME/BartonFang/envs/fzy/bin/python
REPO=$HOME/BartonFang/flash-linear-attention-npu
OP=$REPO/fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged
K=$OP/op_kernel/pre_process_fwd_kernel_merged.cpp
W=/tmp/ppfm241/prof
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=${DEV:-4} PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore PPFM_REF_THREADS=2

echo "=== 编译 + 装包 ==="
TAG=${TAG:-quick} bash "$W/build_install.sh" || exit 1
echo "=== L0 ==="
"$PY" "$OP/scripts/gates/gate_l0.py" "$K"
echo "=== smoke ==="
cd "$OP" || exit 1
"$PY" /tmp/ppfm241/run_npu.py scripts/npu_smoke_ppfm.py --op-dir "$OP" 2>&1 | tail -40
echo QUICK_DONE
