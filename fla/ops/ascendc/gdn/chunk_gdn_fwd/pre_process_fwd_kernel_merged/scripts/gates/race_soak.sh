#!/bin/bash
# L3 进程级竞态 soak（**带标杆缓存 + 并发批次**）——`race_probe_procs.sh` 的省时版。
#
# 为什么需要它：这类竞态**只对"独立进程"敏感**（同进程内复跑 0/40 全过，进程级历史 ~4/6）。
# 而单进程 ~3.5 min 里 ACL 初始化 ~150 s、CPU 标杆 ~58 s、kernel 只 ~1.9 s ⇒
#   * 标杆缓存（race_probe_cached.py）省 ~58 s/进程；
#   * PAR 个进程并发 → 16 进程从 ~56 min 压到 ~20 min。
#
# 用法：  PROCS=20 PAR=4 DEV=1 TAG=verify bash race_soak.sh [op_dir]
# 判据：  末行 SOAK_CLEAN（= 脏进程数 0）；同时打印每次 gdn-t1023 的 h/m max_abs
#         （历史上命中时是 0.91/1.23 的跳变、正常时恒为 1.633e-03 —— 出现两个量级 ⇒ 仍有窗口）
set -u
GATES_DIR=$(cd "$(dirname "$0")" && pwd)
OP=${1:-$(cd "$GATES_DIR/../.." && pwd)}
PROCS=${PROCS:-20}
PAR=${PAR:-4}
DEV=${DEV:-0}
TAG=${TAG:-soak}
PY=${PY:-python3}
PROBE=${PROBE:-$GATES_DIR/race_probe_cached.py}
LOGDIR=${LOGDIR:-/tmp/race_soak_$TAG}

source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
export ASCEND_RT_VISIBLE_DEVICES=$DEV
export FLA_NPU_DISABLE_PTH=${FLA_NPU_DISABLE_PTH:-1}
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
mkdir -p "$LOGDIR"

dirty=0
i=0
while [ "$i" -lt "$PROCS" ]; do
  launched=0
  for _ in $(seq 1 "$PAR"); do
    i=$((i + 1))
    [ "$i" -gt "$PROCS" ] && break
    ( "$PY" "$PROBE" "$OP" 1 > "$LOGDIR/proc_$i.log" 2>&1 ) &
    launched=$((launched + 1))
  done
  wait
  for n_i in $(seq $((i - launched + 1)) "$i"); do
    line=$(grep -E '^--- round' "$LOGDIR/proc_$n_i.log" | tail -n 1)
    n=$(printf '%s' "$line" | sed -n 's/.*FAIL \([0-9]*\)\/.*/\1/p')
    [ -z "$n" ] && n="?"
    printf '[proc %2d/%d] %s\n' "$n_i" "$PROCS" "$line"
    if [ "$n" != "0" ]; then
      dirty=$((dirty + 1))
      grep -E 'FAIL' "$LOGDIR/proc_$n_i.log" | head -3 | sed 's/^/          /'
    fi
    # 只额外打印历史敏感用例：出现 0.8~1.4 量级即为命中
    grep -E 'gdn-t1023' "$LOGDIR/proc_$n_i.log" | tail -n 1 | sed 's/^/          /'
  done
done
echo "# SOAK[$TAG, dev=$DEV, par=$PAR]: $dirty/$PROCS 个进程出现失败（日志 $LOGDIR）"
[ "$dirty" -eq 0 ] && echo SOAK_CLEAN || echo SOAK_DIRTY
