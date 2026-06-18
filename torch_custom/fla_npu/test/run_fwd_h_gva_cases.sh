#!/bin/bash
# fwd_h GVA 用例：example dump 模型输入 + 单算子精度比对
#
# 流程（参考 PR #69）:
#   GDN_FWD_H_DUMP_EXIT=1 仅 dump fwd_h 入参后退出，不跑 fwd_h/fwd_o/bwd
#   2) test_npu_fwd_h_gva.py 加载 .pt，用 test_fwd_h.forward_h_trans_cpu 作标杆，ct.isclose 比对 h/v_new
#
# 用法:
#   export TEST_DEVICE_ID=1
#   bash run_fwd_h_gva_cases.sh                          # 全部用例（Vdim=256 自动 SKIP）
#   FWD_H_CASE=varlen_k16v32_t16384_v128 bash run_fwd_h_gva_cases.sh
#   FWD_H_DUMP_ONLY=1 FWD_H_CASE=varlen_t65536_v128_cu17 bash run_fwd_h_gva_cases.sh  # 仅 dump
#   FWD_H_TEST_ONLY=1 FWD_H_CASE=smoke_k16v32_t4096 bash run_fwd_h_gva_cases.sh       # 仅测已有 dump
#
# 手工 dump 示例:
#   GDN_FWD_H_DUMP_DIR=$PWD/../../examples/fast_kernel_launch_example/tests/chunk_gated_delta_rule_fwd_h/data \
#   GDN_FWD_H_DUMP_NAME=fix_k16v32_t16384 \
#   python3 ../../examples/flash_gated_delta_rule.py \
#     --device 1 --batch 1 --tokens 16384 --query-heads 16 --value-heads 32 \
#     --key-dim 128 --value-dim 128 --chunk-size 64 --mean-len 128 --dtype bf16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DEVICE="${TEST_DEVICE_ID:-5}"
DUMP_DIR="${GDN_FWD_H_DUMP_DIR:-$REPO_ROOT/examples/fast_kernel_launch_example/tests/chunk_gated_delta_rule_fwd_h/data}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate wnc
source /data/zs/run/8.5/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
if [[ -f /data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer/bin/set_env.bash ]]; then
  source /data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer/bin/set_env.bash
fi

export TEST_DEVICE_ID="$DEVICE"
export GDN_FWD_H_DUMP_DIR="$DUMP_DIR"
export PYTHONUNBUFFERED=1

echo "device=$DEVICE dump_dir=$DUMP_DIR"
cd "$SCRIPT_DIR"
python3 test_npu_fwd_h_gva.py
