"""A1 根因修复：L1A/L1B 的 tla tensor 必须用「容量形状」构造（与 BlockMmad 的
L1_TILE_M/K/N 一致），再用 GetTile 取实际子块。

zZ/nZ 是 rank-4 且带 originShape 的分形布局：直接用 (m,k) 构造会让分形 stride 与实际
写入位置不一致，L1→L0 读到的不是刚写进去的数据 ⇒ mmad 读到空操作数，initC=true
清 0 后无有效累加 ⇒ C 恒为 0。
"""

import sys

path = sys.argv[1]
src = open(path).read()

if "TILED_L1_CAP_M" in src:
    print("already patched")
    raise SystemExit(0)

anchor = "constexpr int32_t TILED_L1_B_OFF = 32 * 1024;"
assert anchor in src, "TILED_L1_B_OFF anchor not found"
src = src.replace(
    anchor,
    anchor
    + """
// L1A/L1B 的「容量形状」：必须与 BlockMmad 的 L1_TILE_M/K/N 一致（zZ/nZ 分形布局的
// stride 由 originShape 决定，用实际 (m,k) 构造会让 GM→L1 的落点与 L1→L0 的读点错位，
// 表现为 mmad 读到空 L0、C 恒为 0）。TILED_L1_B_OFF=32KiB 正是 128x128 bf16 的 footprint。
constexpr int32_t TILED_L1_CAP_M = 128;
constexpr int32_t TILED_L1_CAP_K = 128;
constexpr int32_t TILED_L1_CAP_N = 128;""",
    1,
)

old_a = "LayoutTagL1A>(m, k),"
old_b = "LayoutTagL1B>(k, n),"
new_a = "LayoutTagL1A>(TILED_L1_CAP_M, TILED_L1_CAP_K),"
new_b = "LayoutTagL1B>(TILED_L1_CAP_K, TILED_L1_CAP_N),"
assert src.count(old_a) == 2, f"L1A anchor count={src.count(old_a)}"
assert src.count(old_b) == 2, f"L1B anchor count={src.count(old_b)}"
src = src.replace(old_a, new_a).replace(old_b, new_b)

open(path, "w").write(src)
print("patched", path)
