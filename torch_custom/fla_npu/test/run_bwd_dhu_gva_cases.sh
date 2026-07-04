#!/bin/bash
# bwd_dhu GVA 双标杆：随机输入直接测，无需 example dump
#
# 用法:
#   TEST_DEVICE_ID=7 bash run_bwd_dhu_gva_cases.sh
#   BWD_HU_CASE=smoke_varlen_t256_v256 TEST_DEVICE_ID=7 bash run_bwd_dhu_gva_cases.sh
#   BWD_HU_SUITE=unsupported TEST_DEVICE_ID=7 bash run_bwd_dhu_gva_cases.sh   # cases.json 8 项
#   BWD_HU_CASE=gva_fix_3,phase_1_var_4 TEST_DEVICE_ID=7 bash run_bwd_dhu_gva_cases.sh
#   BWD_HU_VIZ=1 BWD_HU_CASE=gva_fix_3 TEST_DEVICE_ID=7 bash run_bwd_dhu_gva_cases.sh  # + ct.viz
#   BWD_HU_SAVE_OUT=0 ...  # skip outputs.pt (default: save)
#
# 环境：见 BWD_DHU_TEST.md；先 export CANN_SET_ENV / VENDOR_SET_ENV 再跑。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${TEST_DEVICE_ID:-5}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate wnc
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_cann_env.sh"

export TEST_DEVICE_ID="$DEVICE"
export BWD_HU_OUT_DIR="${BWD_HU_OUT_DIR:-$SCRIPT_DIR/bwd_dhu_out}"
export PYTHONUNBUFFERED=1

echo "device=$DEVICE suite=${BWD_HU_SUITE:-builtin} case=${BWD_HU_CASE:-<all>} out_dir=$BWD_HU_OUT_DIR viz=${BWD_HU_VIZ:-0}"
cd "$SCRIPT_DIR"
python3 test_npu_bwd_dhu_gva.py
