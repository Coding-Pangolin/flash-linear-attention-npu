#!/bin/bash
# ITER8（A2）验收：编译 + L0 静态门禁 + L2 升级版 smoke + L3 六独立进程序列探针
set -u
PY=$HOME/BartonFang/envs/fzy/bin/python
REPO=$HOME/BartonFang/flash-linear-attention-npu
OP=$REPO/fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged
K=$OP/op_kernel/pre_process_fwd_kernel_merged.cpp
W=/tmp/ppfm241/prof
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=${DEV:-7} PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore PPFM_REF_THREADS=2

echo "=== [1/4] 编译 + 装包 ==="
TAG=iter8 bash "$W/build_install.sh" || exit 1

echo "=== [2/4] L0 静态门禁 ==="
"$PY" "$OP/scripts/gates/gate_l0.py" "$K"

echo "=== [3/4] L2 升级版 smoke（10 形状，含 max_abs 判据）==="
cd "$OP" || exit 1
"$PY" /tmp/ppfm241/run_npu.py scripts/npu_smoke_ppfm.py --op-dir "$OP" 2>&1 | tail -45

echo "=== [4/4] L3 六独立进程序列探针（每进程一轮）==="
for p in 1 2 3 4 5 6; do
  echo "----- process $p -----"
  "$PY" /tmp/ppfm241/run_npu.py "$OP/scripts/gates/seq_probe.py" "$OP" 1 2>&1 \
    | grep -E "^  rd|FAIL|汇总"
done
echo ITER8_RUN_DONE
