#!/bin/bash
# 241：一轮迭代 = [可选重编] + smoke(1轮) + msprof(指定 shape) + 自动分析
#   TAG=iter1 REBUILD=1 SHAPES="1024:8 4096:8" bash iter_run.sh
set -u
PY=${PY:-$HOME/BartonFang/envs/fzy/bin/python}
REPO=${REPO:-$HOME/BartonFang/flash-linear-attention-npu}
OP=$REPO/fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged
W=/tmp/ppfm241/prof
TAG=${TAG:-iter}
SHAPES=${SHAPES:-"1024:8 4096:8"}
DEV=${DEV:-6}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=$DEV PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore PPFM_REF_THREADS=2
export PY=$PY REPO=$REPO TAG=$TAG

if [ "${REBUILD:-1}" = "1" ]; then
  bash $W/build_install.sh || exit 1
fi

echo "=== smoke 1 round ($TAG) ==="
cd "$OP" || exit 1
timeout 900 "$PY" /tmp/ppfm241/run_npu.py scripts/npu_smoke_ppfm.py --op-dir "$OP" 2>&1 \
  | grep -E "^\[(PASS|FAIL)\]|条失败"

echo "=== msprof ($TAG) ==="
for s in $SHAPES; do
  T=${s%%:*}; HV=${s##*:}
  OUT=/tmp/ppfm241/out_${TAG}_T${T}_HV${HV}
  rm -rf "$OUT"
  PPFM_SIM_T=$T PPFM_SIM_HV=$HV timeout 1800 msprof op \
    --application="$PY $W/sim_ppfm.py" --output="$OUT" \
    --kernel-name=PreProcessFwdKernelMerged --launch-count=1 --warm-up=1 \
    --aic-metrics=Default > "$W/msop_${TAG}_T${T}_HV${HV}.log" 2>&1
  echo "  rc=$? $(find "$OUT" -maxdepth 2 -name OpBasicInfo.csv | head -1)"
done

echo "=== analyze ($TAG) ==="
for s in $SHAPES; do
  T=${s%%:*}; HV=${s##*:}
  D=$(find /tmp/ppfm241/out_${TAG}_T${T}_HV${HV} -maxdepth 1 -name "OPPROF_*" | head -1)
  [ -n "$D" ] && "$PY" $W/analyze_msopprof.py ${T}:${HV}:$D 2>&1 | grep -E "^=====|duration|归一化|\[cube0|\[vector0" | head -6
done
echo "${TAG}_DONE"
