#!/bin/bash
# 规范审计：按本仓《算子工程结构规范》(docs/agents/reference/04-operator-development/
# engineering-structure.md，V1；PR#728 为最新版) 逐条审视本算子，另附 cannbot-skills 的两份
# 检视规范（cpp-style.md / ascendc-op-conventions.md）作为佐证。
#
# 用法：  bash convention_audit.sh [op_dir]
# 判据：  末行 CONVENTION_CLEAN；[FAIL]=阻塞项，[WARN]=需在 docs/design.md 或 PR 描述里说明
set -u
GATES_DIR=$(cd "$(dirname "$0")" && pwd)
OP=${1:-$(cd "$GATES_DIR/../.." && pwd)}
K=$OP/op_kernel/pre_process_fwd_kernel_merged.cpp
[ -f "$K" ] || { echo "找不到 $K"; exit 2; }

fail=0; warn=0
say_fail() { echo "  [FAIL] $1"; fail=$((fail + 1)); }
say_warn() { echo "  [WARN] $1"; warn=$((warn + 1)); }

echo "=============================================================="
echo "规范审计（本仓 engineering-structure.md V1 + §4.4/PR#728）  op=$OP"
echo "=============================================================="

echo
echo "【§2 标准目录结构】"
lines=$(wc -l < "$K")
echo "  op_kernel 主 .cpp $lines 行"
[ "$lines" -gt 200 ] && say_fail "主 .cpp $lines 行 ⇒ §4.3 要求入口只接线（样板 77 行）；Stage 计算应在 archXX/*.h"
for d in arch22 arch35; do
  [ -d "$OP/op_kernel/$d" ] || say_warn "缺 op_kernel/$d/（§4.1 两种布局之一：默认+A5 差异，或 A2/A3 与 A5 各一份）"
done
[ -f "$OP/op_kernel/pre_process_fwd_kernel_merged_common.h" ] || say_warn "缺 op_kernel/*_common.h（平台无关常量/同步协议声明/ChunkInfo 换算）"
[ -f "$OP/op_kernel/pre_process_fwd_kernel_merged_tiling_key.h" ] || say_fail "缺 op_kernel/*_tiling_key.h（§4.2：TilingKey 模板化是必需件）"
[ -f "$OP/op_host/pre_process_fwd_kernel_merged_tiling_processor.h" ] || say_warn "缺 op_host/*_tiling_processor.h（§2：tiling 计算主体 header-only）"

echo
echo "【§4.2 TilingKey / §4.3 入口 / §4.1-8 开关命名】"
tki=$(grep -c 'TILING_KEY_IS' "$K" || true)
[ "$tki" -eq 0 ] && echo "  Kernel 未用 TILING_KEY_IS ✅" \
  || say_fail "Kernel 使用 TILING_KEY_IS $tki 处（§4.2 禁止用运行期分支代替模板化）"
tc=$(grep -c 'TORCH_MODE' "$OP"/op_kernel/*.cpp "$OP"/op_kernel/*.h 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
[ "$tc" -eq 0 ] && echo "  未用 TORCH_MODE ✅" \
  || say_fail "使用 TORCH_MODE $tc 处（§4.1-8 点名要求改为按场景命名的 FLA_TORCH_EXTENSION_INLINE_BUILD）"
dg=$(grep -c 'DeviceGemm' "$K" || true)
[ "$dg" -eq 0 ] && echo "  未用 DeviceGemm ✅" || say_fail "使用 DeviceGemm $dg 处（C4 禁止）"
inc=$(grep -n '#include' "$K" | grep -i tiling | grep -v '_struct.h\|_tiling_key.h' || true)
[ -z "$inc" ] && echo "  op_kernel 未 include 自身 tiling 实现 ✅" || say_fail "op_kernel include 自身 tiling 实现（C7）：$inc"
gu=$(grep -c 'GetUserWorkspace' "$K" || true)
[ "$gu" -eq 0 ] && echo "  未用 GetUserWorkspace ✅" \
  || say_warn "使用 GetUserWorkspace $gu 处（cannbot-skills C6 只对 catlass 直调路径禁止；常规 OPP 需在 PR 描述里写豁免理由）"

echo
echo "【§4.4 ① 文件头三张表（Stage 表 / 布局表 / 同步协议表）】"
for t in "Stage 表" "布局表" "同步协议"; do
  n=$(grep -c "$t" "$K" || true)
  if [ "$n" -eq 0 ]; then say_fail "文件头缺『$t』（§4.4 ①：文件最上方三张表）"; else echo "  含『$t』✅"; fi
done

echo
echo "【§4.4 ②③ 计算层与数据结构体】"
sa=$(grep -c 'static_assert' "$K" || true)
[ "$sa" -gt 0 ] && echo "  static_assert $sa 处 ✅" \
  || say_fail "无 static_assert（§4.4 ②：用 traits 钉 L1/L0 空间上限；§7.7 常量集中配 static_assert）"
[ -n "$(grep -n 'CubePrimitives' "$K" || true)" ] && echo "  有 CubePrimitives traits ✅" \
  || say_warn "未见 <算子>CubePrimitives<DT> traits（§4.4 ②：TileCopy*/TileMmad/CopyL1ToL0*/CopyL0CToDst + layout 常量集中一处）"
if grep -qE '^\s*class\s+[A-Za-z_]+\s*\{' "$K" && grep -qE '^\s*__aicore__\s+inline' "$K"; then
  say_fail "存在『类 + 类内 __aicore__ 函数』（§4.4 ③：函数不写进类/结构体，Context 只放数据）"
else
  echo "  未见『函数写进类』✅"
fi
[ -n "$(grep -n 'GetChunkInfo' "$K" || true)" ] && echo "  有 GetChunkInfo 换算 ✅" \
  || say_warn "未见 GetChunkInfo(...)（§4.4 ⑤：任务换算封函数，Stage 只消费 ChunkInfo）"

echo
echo "【§4.4 ④ 事件生命周期 / §7.5 同步】"
if grep -q 'AllocEventID' "$K"; then echo "  含 AllocEventID ✅"; else
  say_warn "未见 AllocEventID（§4.4 ④：Init 里按 slot AllocEventID+SetFlag 开首轮、Process 末 WaitFlag+ReleaseEventID）"
fi
grep -qE 'EVENT_ID[0-9]' "$K" && say_warn "存在裸 EVENT_IDn 用法（§4.4：事件 id 与物理槽一一对应，用命名常量）"

echo
echo "【§7 编码细节 / cpp-style】"
wide=$(awk 'length($0) > 120 {c++} END {print c+0}' "$K")
[ "$wide" -eq 0 ] && echo "  行宽 >120 的行数 0 ✅" || say_fail "行宽 >120 的行数 $wide"
dev=$(grep -cE '//.*(ITER[0-9]|R[0-9]+[^0-9]|R0\.8|待上机|首测|实测负收益|待实测|默认关|回退到|历史做法|旧行为)' "$K" || true)
if [ "$dev" -eq 0 ]; then echo "  开发阶段注释 0 ✅"; else
  say_fail "开发阶段注释 $dev 行（cpp-style 规则 3.3 禁止 TODO/开发阶段注释）；抽样："
  grep -nE '//.*(ITER[0-9]|R[0-9]+[^0-9]|待上机|首测|实测负收益|默认关)' "$K" | head -3 | sed 's/^/          /'
fi
deco=$(grep -cE '⚠|✅|❌' "$K" || true)
[ "$deco" -eq 0 ] && echo "  装饰符号 0 ✅" || say_warn "装饰符号（⚠/✅/❌）$deco 处，建议改为文字"
dead=$(find "$OP/op_kernel" "$OP/op_host" -maxdepth 1 -type f \( -name '*.cpp.bak' -o -name '*.cpp.txt' -o -name '*.wip' \) 2>/dev/null || true)
[ -z "$dead" ] && echo "  无死文件 ✅" || { printf '%s\n' "$dead" | sed 's/^/  [FAIL] 死文件 /'; fail=$((fail + 1)); }

echo
echo "【§8 交付前清单（可机械查的部分）】"
repo=$(git -C "$OP" rev-parse --show-toplevel 2>/dev/null || echo "")
if [ -n "$repo" ]; then
  dirty=$(git -C "$repo" status --short | wc -l)
  echo "  git status --short 条目数 $dirty"
  git -C "$repo" diff --check >/dev/null 2>&1 && echo "  git diff --check 干净 ✅" || say_warn "git diff --check 有空白问题"
else
  say_warn "不在 git 仓库内，跳过 §8 的 git 检查"
fi
[ -f "$OP/README.md" ] && echo "  README.md ✅" || say_fail "缺算子 README.md（§7.10：能力与输入限制的唯一来源）"
[ -f "$OP/docs/design.md" ] && echo "  docs/design.md ✅" || say_fail "缺 docs/design.md（§7.10）"
[ -f "$OP/docs/api.md" ] && echo "  docs/api.md ✅" || say_fail "缺 docs/api.md（§7.10）"

echo
echo "--------------------------------------------------------------"
echo "合计 FAIL=$fail  WARN=$warn"
[ "$fail" -eq 0 ] && echo "CONVENTION_CLEAN（WARN 需在 docs/design.md 或 PR 描述里逐条说明）" || echo "CONVENTION_DIRTY"
exit "$fail"
