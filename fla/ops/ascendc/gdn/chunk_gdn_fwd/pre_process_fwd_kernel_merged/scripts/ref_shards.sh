#!/bin/bash
# 大用例标杆分段并行：每个进程算若干段，写 part 文件（之后用 --merge-parts 合并）。
# 用法： bash ref35_shards.sh [用例ID] [总段数] [进程数] [每进程段数]
cd "$(dirname "$0")" || exit 1
export PPFM_REF_THREADS="${PPFM_REF_THREADS:-8}"
CASE="${1:-PPFM-35}"
TOTAL="${2:-64}"
JOBS="${3:-12}"
PER="${4:-6}"
for ((i = 0; i < JOBS; i++)); do
    lo=$((i * PER))
    hi=$((lo + PER))
    if [ "$hi" -gt "$TOTAL" ]; then hi=$TOTAL; fi
    if [ "$lo" -ge "$TOTAL" ]; then break; fi
    nohup python3 -u run_cases_ppfm.py --ref-only --only "${CASE}" \
        --windows "${lo}:${hi}" > "/tmp/ref_${CASE}_${lo}_${hi}.log" 2>&1 &
done
wait
echo "all shards done"
