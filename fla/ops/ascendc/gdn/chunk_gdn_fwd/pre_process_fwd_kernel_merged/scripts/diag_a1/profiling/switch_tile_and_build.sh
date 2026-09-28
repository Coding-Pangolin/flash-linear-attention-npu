#!/bin/bash
# 切换 PPFM_TILE_MMAD 并重编装包：bash switch_tile_and_build.sh <0|1>
set -u
TILE=${1:?usage: switch_tile_and_build.sh <0|1>}
PY=${PY:-$HOME/BartonFang/envs/fzy/bin/python}
REPO=${REPO:-$HOME/BartonFang/flash-linear-attention-npu}
K=$REPO/fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged/op_kernel/pre_process_fwd_kernel_merged.cpp
source /usr/local/Ascend/ascend-toolkit/set_env.sh
sed -i "s/^#define PPFM_TILE_MMAD [0-9]*/#define PPFM_TILE_MMAD $TILE/" "$K"
grep -n "^#define PPFM_TILE_MMAD " "$K"
cd "$REPO" || exit 9
rm -rf dist
FLA_NPU_SOC=ascend950 FLA_NPU_OPS=pre_process_fwd_kernel_merged "$PY" scripts/build_wheel.py \
  > /tmp/ppfm241/prof/build_tile${TILE}.log 2>&1
grep -q "\[fla-npu build\] Wheel" /tmp/ppfm241/prof/build_tile${TILE}.log \
  && echo BUILD_OK || { echo BUILD_FAIL; tail -20 /tmp/ppfm241/prof/build_tile${TILE}.log; exit 1; }
"$PY" -m pip install --force-reinstall --no-cache-dir --no-deps "$(ls -t dist/*.whl | head -1)" \
  > /tmp/ppfm241/prof/install_tile${TILE}.log 2>&1
grep -q "Successfully installed" /tmp/ppfm241/prof/install_tile${TILE}.log && echo INSTALL_OK || echo INSTALL_FAIL
