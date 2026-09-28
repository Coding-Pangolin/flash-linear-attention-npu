#!/bin/bash
# 246：编译 + L0 + smoke + 六进程序列探针（快检，用于竞态对照实验）
set -u
REPO=/workspace/bartonfang/flash-linear-attention-npu
OP=$REPO/fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged
K=$OP/op_kernel/pre_process_fwd_kernel_merged.cpp
W=/tmp/w246
mkdir -p $W
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore PPFM_REF_THREADS=${REF_THREADS:-8}
PY=python3

echo "=== [0/4] 同步 ==="
cd "$REPO" || exit 9
git checkout -B feat/ppfm-tile-a5 fork/feat/ppfm-tile-a5 2>&1 | tail -1
git log --oneline -1
grep -n "define PPFM_LEGACY_CACHEOPS\|define PPFM_VTMP_UB " "$K"

echo "=== [1/4] 编译 + 装包 ==="
rm -rf dist
FLA_NPU_SOC=ascend950 FLA_NPU_OPS=pre_process_fwd_kernel_merged $PY scripts/build_wheel.py > $W/build.log 2>&1
grep -q "\[fla-npu build\] Wheel" $W/build.log && echo BUILD_OK || { echo BUILD_FAIL; tail -25 $W/build.log; exit 1; }
echo "build_errors=$(grep -c 'error:' $W/build.log)"
$PY -m pip install --force-reinstall --no-cache-dir --no-deps "$(ls -t dist/*.whl | head -1)" > $W/install.log 2>&1
grep -q "Successfully installed" $W/install.log && echo INSTALL_OK || { echo INSTALL_FAIL; tail -10 $W/install.log; exit 1; }

echo "=== [2/4] L0 ==="
$PY "$OP/scripts/gates/gate_l0.py" "$K"

echo "=== [3/4] smoke ==="
cd "$OP" || exit 1
$PY scripts/npu_smoke_ppfm.py --op-dir "$OP" 2>&1 | tail -42

echo "=== [4/4] L3 六独立进程序列探针 ==="
for p in 1 2 3 4 5 6; do
  echo "----- process $p -----"
  $PY "$OP/scripts/gates/seq_probe.py" "$OP" 1 2>&1 | grep -E "^  rd|FAIL|汇总"
done
echo GATE246B_DONE
