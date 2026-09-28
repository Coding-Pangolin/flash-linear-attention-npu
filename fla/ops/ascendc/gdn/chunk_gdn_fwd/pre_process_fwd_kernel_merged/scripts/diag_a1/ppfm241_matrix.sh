#!/bin/bash
# 241 上的变体矩阵：build → install → smoke（1 轮）/ diag
set -u
REPO=$HOME/BartonFang/flash-linear-attention-npu
OP=$REPO/fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged
K=$OP/op_kernel/pre_process_fwd_kernel_merged.cpp
PY=$HOME/BartonFang/envs/fzy/bin/python
WORK=/tmp/ppfm241
LOG=$WORK/matrix.log
export ASCEND_RT_VISIBLE_DEVICES=${DEV:-6}
export PPFM_REF_THREADS=${REF_THREADS:-2}
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
source /usr/local/Ascend/ascend-toolkit/set_env.sh || exit 9
: > "$LOG"

run_variant() {
  local tile=$1 sel=$2 mmt=$3 mode=${4:-smoke}
  echo "############ VARIANT TILE=$tile SEL=$sel MMT=$mmt MODE=$mode ############" | tee -a "$LOG"
  sed -i "s/^#define PPFM_TILE_MMAD [0-9]*/#define PPFM_TILE_MMAD $tile/;s/^#define PPFM_TILE_MMAD_SEL [0-9]*/#define PPFM_TILE_MMAD_SEL $sel/;s/^#define PPFM_TILE_MMT [0-9]*/#define PPFM_TILE_MMT $mmt/" "$K"
  grep -n "^#define PPFM_TILE_MMAD \|^#define PPFM_TILE_MMAD_SEL\|^#define PPFM_TILE_MMT" "$K" | tee -a "$LOG"
  cd "$REPO" || return
  rm -rf dist
  FLA_NPU_SOC=ascend950 FLA_NPU_OPS=pre_process_fwd_kernel_merged \
    "$PY" scripts/build_wheel.py > "$WORK/build.log" 2>&1
  if ! grep -q "\[fla-npu build\] Wheel" "$WORK/build.log"; then
    echo "BUILD_FAILED" | tee -a "$LOG"
    tail -25 "$WORK/build.log" | tee -a "$LOG"
    return
  fi
  echo "build_errors=$(grep -c 'error:' "$WORK/build.log")" | tee -a "$LOG"
  local whl
  whl=$(ls -t dist/*.whl | head -1)
  "$PY" -m pip install --force-reinstall --no-cache-dir --no-deps "$whl" \
    > "$WORK/install.log" 2>&1
  grep -q "Successfully installed" "$WORK/install.log" || {
    echo INSTALL_FAILED | tee -a "$LOG"; tail -15 "$WORK/install.log" | tee -a "$LOG"; return; }
  echo "installed $(basename "$whl")" | tee -a "$LOG"
  cd "$OP" || return
  if [ "$mode" = "diag" ]; then
    "$PY" "$WORK/run_npu.py" "$WORK/diag_map.py" "$OP" >> "$LOG" 2>&1
  else
    "$PY" "$WORK/run_npu.py" scripts/npu_smoke_ppfm.py --op-dir "$OP" 2>&1 \
      | grep -E "^\[(PASS|FAIL)\]|matched=|条失败" | tee -a "$LOG"
  fi
  echo "" | tee -a "$LOG"
}

N=${N:-1}
[ "$N" -ge 1 ] && run_variant "${T1:-1}" "${S1:-3}" "${M1:-0}" "${MODE1:-smoke}"
[ "$N" -ge 2 ] && run_variant "${T2:-1}" "${S2:-1}" "${M2:-0}" "${MODE2:-smoke}"
[ "$N" -ge 3 ] && run_variant "${T3:-1}" "${S3:-2}" "${M3:-0}" "${MODE3:-smoke}"
[ "$N" -ge 4 ] && run_variant "${T4:-1}" "${S4:-3}" "${M4:-0}" "${MODE4:-smoke}"
echo "ALL_DONE" | tee -a "$LOG"
