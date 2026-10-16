#!/bin/bash
# 开关整改审计：打印 op_kernel 的"实验化"指标 + 核对 docs/switch_remediation_plan.md §6 的验收项。
# 用途：整改前后各跑一次；也用来回答"这份计划是不是还基于最新代码"（换 commit 后重跑即可）。
#
# 用法：  bash switch_audit.sh [op_dir]
# 判据：  整改完成后应满足 —— 实验开关 0 个、PPFM_ 条件只剩 arch/布局、无 getenv、无死文件、static_assert > 0
set -u
GATES_DIR=$(cd "$(dirname "$0")" && pwd)
OP=${1:-$(cd "$GATES_DIR/../.." && pwd)}
K=$OP/op_kernel/pre_process_fwd_kernel_merged.cpp
[ -f "$K" ] || { echo "找不到 $K"; exit 2; }

commit=$(git -C "$(dirname "$K")" rev-parse --short HEAD 2>/dev/null || echo "?")
verdict=0

echo "=============================================================="
echo "开关整改审计  commit=$commit  file=$K"
echo "=============================================================="

total=$(wc -l < "$K")
ifcnt=$(grep -c '^[[:space:]]*#[[:space:]]*if' "$K" || true)
ppif=$(grep -c '^[[:space:]]*#[[:space:]]*if.*PPFM_' "$K" || true)
sa=$(grep -c 'static_assert' "$K" || true)
echo "[规模] 总行数 $total ; #if 指令 $ifcnt ; 其中条件是 PPFM_ 的 $ppif ; static_assert $sa"

echo
echo "--- 实验开关（\`#ifndef PPFM_X\` 形式；整改目标是 0 个）---"
switches=$(grep -oE '^#ifndef[[:space:]]+PPFM_[A-Z0-9_]+' "$K" | awk '{print $2}' | sort | uniq -c | sort -rn || true)
if [ -z "$switches" ]; then
  echo "  （无）✅"
else
  printf '%s\n' "$switches" | sed 's/^/  /'
  n=$(printf '%s\n' "$switches" | wc -l)
  echo "  ⇒ 共 $n 个开关名未清理"
  verdict=1
fi

echo
echo "--- kernel 里出现过的全部 PPFM_* 名字（整改后应只剩 arch/布局白名单）---"
grep -oE 'PPFM_[A-Z0-9_]+' "$K" | sort -u | sed 's/^/  /'

echo
echo "--- 生产代码里的测试钩子（期望：空）---"
hooks=$(grep -n 'getenv' "$OP"/op_host/*.cpp "$OP"/op_host/op_api/*.cpp \
                       "$OP"/op_kernel/*.cpp "$OP"/op_kernel/*.h 2>/dev/null \
        | grep -v '/build/' || true)
if [ -z "$hooks" ]; then echo "  （无）✅"; else printf '%s\n' "$hooks" | sed 's/^/  /'; verdict=1; fi

echo
echo "--- 诊断/探针残留（期望：空）---"
#   只扫源码目录（op_kernel 根下的 .cpp/.h）；不递归进 build/ 的副本
diag=$(grep -n 'PPFM_DIAG\|PPFM_RD_PROBE\|PPFM_DEBUG_HEADER\|PPFM_SENTINEL_PROBE\|PPFM_VTMP_UB_DIAG' \
       "$OP"/op_kernel/*.cpp "$OP"/op_kernel/*.h 2>/dev/null || true)
if [ -z "$diag" ]; then echo "  （无）✅"; else printf '%s\n' "$diag" | head -8 | sed 's/^/  /'; verdict=1; fi

echo
echo "--- 死文件（期望：不存在）---"
dead=$(find "$OP/op_kernel" "$OP/op_host" -maxdepth 1 -type f \
        \( -name '*.cpp.bak' -o -name '*.cpp.txt' -o -name '*.wip' -o -name '*.bak' \) 2>/dev/null || true)
if [ -z "$dead" ]; then echo "  （无）✅"; else printf '%s\n' "$dead" | sed 's/^/  /'; verdict=1; fi

echo
echo "--- 断言（整改后应 > 0：UB 预算 / workspace 偏移）---"
[ "$sa" -gt 0 ] && echo "  static_assert = $sa ✅" || { echo "  static_assert = 0 ❌"; verdict=1; }

echo
[ "$verdict" -eq 0 ] && echo "AUDIT_CLEAN（已符合整改目标）" || echo "AUDIT_DIRTY（仍有待整改项，见上）"
exit "$verdict"
