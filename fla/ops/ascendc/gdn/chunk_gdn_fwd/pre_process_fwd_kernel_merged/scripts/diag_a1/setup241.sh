#!/bin/bash
set -u
W=/tmp/ppfm241
mkdir -p "$W"
tar -xzf /tmp/ppfm241.tgz -C "$W"
K=$HOME/BartonFang/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged/op_kernel/pre_process_fwd_kernel_merged.cpp
echo "===PATCH==="
python3 "$W/prep_sel_macro.py" "$K"
python3 "$W/patch_mmlayout.py" "$K"
python3 "$W/guard_mmt.py" "$K"
grep -n "^#define PPFM_TILE_MMAD \|^#define PPFM_TILE_MMAD_SEL\|^#define PPFM_TILE_MMT" "$K"
chmod +x "$W/ppfm241_matrix.sh"
echo "===LAUNCH==="
cd "$W" && T1=1 S1=3 M1=0 MODE1=smoke N=1 nohup bash "$W/ppfm241_matrix.sh" > "$W/matrix.out" 2>&1 &
sleep 3
echo LAUNCHED
tail -3 "$W/matrix.log" 2>/dev/null
