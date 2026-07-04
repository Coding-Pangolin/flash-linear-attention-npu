#!/bin/bash
# fwd_h GVA 用例：example dump 模型输入 + 单算子精度比对
#
# 用法:
#   TEST_DEVICE_ID=1 bash run_fwd_h_gva_cases.sh
#   FWD_H_SUITE=unsupported FWD_H_CASE=gva_fix_3 bash run_fwd_h_gva_cases.sh
#
# 随机输入 CPU dual（推荐，不依赖 example dump）见 FWD_H_TEST.md
#
# 环境：见 FWD_H_TEST.md；先 export CANN_SET_ENV / CANN_OPP_LIB 再跑。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DEVICE="${TEST_DEVICE_ID:-5}"
DUMP_DIR="${GDN_FWD_H_DUMP_DIR:-$REPO_ROOT/examples/fast_kernel_launch_example/tests/chunk_gated_delta_rule_fwd_h/data}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate wnc
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_cann_env.sh"

export TEST_DEVICE_ID="$DEVICE"
export GDN_FWD_H_DUMP_DIR="$DUMP_DIR"
export FWD_H_TEST_ONLY="${FWD_H_TEST_ONLY:-0}"
export FWD_H_DUMP_ONLY="${FWD_H_DUMP_ONLY:-0}"
export FWD_H_FORCE_DUMP="${FWD_H_FORCE_DUMP:-0}"
export FWD_H_OUT_DIR="${FWD_H_OUT_DIR:-$SCRIPT_DIR/fwd_h_out}"
export FWD_H_VIZ="${FWD_H_VIZ:-1}"
export PYTHONUNBUFFERED=1

echo "device=$DEVICE dump_dir=$DUMP_DIR out_dir=$FWD_H_OUT_DIR suite=${FWD_H_SUITE:-builtin}"
cd "$SCRIPT_DIR"
python3 test_npu_fwd_h_gva.py
