#!/bin/bash
# 用 msopprof **上板**模式采我们的算子，目标是拿到 visualize_data.bin（MindStudio Insight 用）
# 实测环境：247 / 容器 admin123-ppfm-test / 真卡 Ascend950
set -u
W=${W:-/data/admin123/ppfm-sim}
P=$W/wheel/flash_linear_attention_npu-26.7.0.dev0.data/purelib
METRICS=${METRICS:-PipeUtilization}
T=${T:-64}
HV=${HV:-1}
OUT=${OUT:-$W/out_msopprof_board_T${T}_HV${HV}}

export PYTHONPATH=$P:${PYTHONPATH:-}
export PPFM_SIM_T=$T
export PPFM_SIM_HV=$HV

rm -rf "$OUT"; mkdir -p "$OUT"
echo "[run] msprof op --application=\"python3 $W/sim_ppfm.py\" --kernel-name=<PPFM> --launch-count=1"
timeout 1800 msprof op \
    --application="python3 $W/sim_ppfm.py" \
    --output="$OUT" \
    --kernel-name=PreProcessFwdKernelMerged \
    --launch-count=1 \
    --warm-up=1 \
    --aic-metrics="$METRICS"
echo "EXIT=$?"
echo "=== 产出（找 visualize_data.bin） ==="
find "$OUT" -name "visualize_data.bin" -o -name "*.bin" -o -name "*.csv" -o -name "trace*.json" | head -20
echo "=== 目录树 ==="
find "$OUT" -maxdepth 3 -type d | head -20
