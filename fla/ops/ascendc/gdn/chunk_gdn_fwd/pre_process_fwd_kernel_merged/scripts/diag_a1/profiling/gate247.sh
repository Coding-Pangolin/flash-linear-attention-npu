#!/bin/bash
# 247（容器内）验收：编译 + L0 静态门禁 + L2 十形状 smoke + L3 六独立进程序列探针
set -u
REPO=/data/admin123/flash-linear-attention-npu
OP=$REPO/fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged
K=$OP/op_kernel/pre_process_fwd_kernel_merged.cpp
LAUNCH=/data/admin123/run_ppfm_npu.py
W=/tmp/ppfm247
mkdir -p $W
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=${DEV:-3} PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export PPFM_REF_THREADS=${REF_THREADS:-4}

echo "=== [1/4] 编译 + 装包 ==="
cd "$REPO" || exit 9
rm -rf dist
FLA_NPU_SOC=ascend950 FLA_NPU_OPS=pre_process_fwd_kernel_merged python3 scripts/build_wheel.py \
  > $W/build.log 2>&1
grep -q "\[fla-npu build\] Wheel" $W/build.log && echo BUILD_OK || { echo BUILD_FAIL; tail -25 $W/build.log; exit 1; }
echo "build_errors=$(grep -c 'error:' $W/build.log)"
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "$(ls -t dist/*.whl | head -1)" \
  > $W/install.log 2>&1
grep -q "Successfully installed" $W/install.log && echo INSTALL_OK || { echo INSTALL_FAIL; tail -10 $W/install.log; exit 1; }

echo "=== [2/4] L0 静态门禁 ==="
python3 "$OP/scripts/gates/gate_l0.py" "$K"

echo "=== [3/4] L2 十形状 smoke（含 max_abs 判据）==="
cd "$OP" || exit 1
python3 $LAUNCH scripts/npu_smoke_ppfm.py --op-dir "$OP" 2>&1 | tail -45

echo "=== [4/4] L3 六独立进程序列探针 ==="
for p in 1 2 3 4 5 6; do
  echo "----- process $p -----"
  python3 $LAUNCH "$OP/scripts/gates/seq_probe.py" "$OP" 1 2>&1 | grep -E "^  rd|FAIL|汇总"
done
echo GATE247_DONE
