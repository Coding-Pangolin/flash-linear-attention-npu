#!/bin/bash
# 241：编译 + 安装 wheel（不切宏）
set -u
PY=${PY:-$HOME/BartonFang/envs/fzy/bin/python}
REPO=${REPO:-$HOME/BartonFang/flash-linear-attention-npu}
TAG=${TAG:-build}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$REPO" || exit 9
rm -rf dist
FLA_NPU_SOC=ascend950 FLA_NPU_OPS=pre_process_fwd_kernel_merged "$PY" scripts/build_wheel.py \
  > "/tmp/ppfm241/prof/build_${TAG}.log" 2>&1
grep -q "\[fla-npu build\] Wheel" "/tmp/ppfm241/prof/build_${TAG}.log" \
  && echo BUILD_OK || { echo BUILD_FAIL; tail -25 "/tmp/ppfm241/prof/build_${TAG}.log"; exit 1; }
"$PY" -m pip install --force-reinstall --no-cache-dir --no-deps "$(ls -t dist/*.whl | head -1)" \
  > "/tmp/ppfm241/prof/install_${TAG}.log" 2>&1
grep -q "Successfully installed" "/tmp/ppfm241/prof/install_${TAG}.log" && echo INSTALL_OK || echo INSTALL_FAIL
