#!/bin/bash
# 上板性能采集：按 "<T>:<HV>" 列表逐个采 msprof op，并打印 Task Duration。
#
# 用法（容器内）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   bash run_msopprof.sh 1024:8 4096:8 11264:32
# 可选环境变量：OUTBASE（产物目录，默认 ./perf）、HARNESS（默认同目录 perf_harness.py）、
#              VARIANT（gdn/kda）、KERNEL_NAME（默认 PreProcessFwdKernelMerged）
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
HARNESS=${HARNESS:-$HERE/perf_harness.py}
OUTBASE=${OUTBASE:-$PWD/perf}
VARIANT=${VARIANT:-gdn}
KERNEL_NAME=${KERNEL_NAME:-PreProcessFwdKernelMerged}
mkdir -p "$OUTBASE"

for spec in "$@"; do
  T=${spec%%:*}
  HV=${spec##*:}
  OUT="$OUTBASE/msop_T${T}_HV${HV}"
  rm -rf "$OUT"
  echo "===== T=$T HV=$HV variant=$VARIANT ====="
  PPFM_T=$T PPFM_HK=$HV PPFM_HV=$HV PPFM_VARIANT=$VARIANT \
    msprof op --application="python3 $HARNESS" \
      --output="$OUT" --kernel-name="$KERNEL_NAME" \
      --launch-count=1 --warm-up=1 --aic-metrics=Default > "${OUT}.log" 2>&1
  csv=$(find "$OUT" -name OpBasicInfo.csv 2>/dev/null | head -n 1)
  if [ -n "$csv" ]; then
    echo "--- OpBasicInfo.csv ---"
    cat "$csv"
  else
    echo "!! 没有 OpBasicInfo.csv；日志尾部："
    tail -n 15 "${OUT}.log"
  fi
done
