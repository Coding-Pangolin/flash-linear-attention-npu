#!/bin/bash
# L3 进程级竞态探针（计划 §5.6 / §11.3）：**每次都是新进程**跑一遍固定用例序列，
# 复现"跨进程顺序/时序相关"的竞态（历史上 gdn-t1023 约 1/6 概率整 head 崩）。
#
# 用法：  PROCS=20 bash race_probe_procs.sh <op_dir>
# 判据：  总 FAIL == 0；并把每轮的 h/m max_abs 打出来看分布（出现两个量级 ⇒ 仍有窗口）。
set -u
OP=${1:?usage: race_probe_procs.sh <op_dir>}
PROCS=${PROCS:-20}
export PPFM_OP_DIR="$OP"
tot_fail=0
for i in $(seq 1 "$PROCS"); do
  out=$(python3 "$OP/scripts/gates/seq_probe.py" "$OP" 1 2>&1)
  line=$(echo "$out" | grep -E '^--- round' | tail -n 1)
  n=$(echo "$line" | sed -n 's/.*FAIL \([0-9]*\)\/.*/\1/p')
  [ -z "$n" ] && n="?"
  echo "[proc $i/$PROCS] $line"
  if [ "$n" != "0" ]; then
    tot_fail=$((tot_fail + 1))
    echo "$out" | grep -E 'FAIL|max|崩' | head -n 6
  fi
done
echo "# 进程级竞态探针：$tot_fail/$PROCS 个进程出现失败"
[ "$tot_fail" -eq 0 ] && echo RACE_PROBE_CLEAN || echo RACE_PROBE_DIRTY
