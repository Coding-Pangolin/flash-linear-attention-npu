#!/bin/bash
# tile0 对照：TILE=0 + 同一批性能改动 ⇒ 竞态命中率与性能
set -u
PY=$HOME/BartonFang/envs/fzy/bin/python
REPO=$HOME/BartonFang/flash-linear-attention-npu
OP=$REPO/fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged
K=$OP/op_kernel/pre_process_fwd_kernel_merged.cpp
W=/tmp/ppfm241/prof
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=${DEV:-7} PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore PPFM_REF_THREADS=2

echo "=== 切 TILE=0 重编 ==="
bash "$W/switch_tile_and_build.sh" 0 || exit 1

echo "=== TILE=0: 6 次独立进程序列探针 ==="
cd "$OP" || exit 1
for p in 1 2 3 4 5 6; do
  echo "----- process $p -----"
  "$PY" /tmp/ppfm241/run_npu.py "$OP/scripts/gates/seq_probe.py" "$OP" 1 2>&1 \
    | grep -E "^  rd|FAIL|汇总"
done

echo "=== TILE=0: 采集 T=1024/4096 HV=8 ==="
for s in 1024:8 4096:8; do
  T=${s%%:*}; HV=${s##*:}
  OUT=/tmp/ppfm241/out_tile0_T${T}_HV${HV}
  rm -rf "$OUT"
  PPFM_SIM_T=$T PPFM_SIM_HV=$HV timeout 1800 msprof op \
    --application="$PY $W/sim_ppfm.py" --output="$OUT" \
    --kernel-name=PreProcessFwdKernelMerged --launch-count=1 --warm-up=1 \
    --aic-metrics=Default > "$W/msop_tile0_T${T}.log" 2>&1
  D=$(find "$OUT" -maxdepth 1 -name "OPPROF_*" | head -1)
  [ -n "$D" ] && "$PY" "$W/analyze_msopprof.py" ${T}:${HV}:"$D" 2>&1 \
    | grep -E "^=====|duration" | head -3
done

echo "=== 恢复 TILE=1 重编 ==="
cd "$REPO" && git checkout -- "$K"
TAG=restore bash "$W/build_install.sh" || exit 1
echo TILE0_CHECK_DONE
