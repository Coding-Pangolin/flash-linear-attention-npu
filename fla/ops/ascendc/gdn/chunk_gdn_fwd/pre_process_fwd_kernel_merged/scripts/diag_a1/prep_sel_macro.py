import re
import sys

path = sys.argv[1]
src = open(path).read()

old_default = """#ifndef PPFM_TILE_MMAD
#define PPFM_TILE_MMAD 1
#endif"""

new_default = """#ifndef PPFM_TILE_MMAD
#define PPFM_TILE_MMAD 1
#endif
// 隔离实验开关：bit0 = mm1(vTmp) 走 tile，bit1 = mm3(T1) 走 tile
#ifndef PPFM_TILE_MMAD_SEL
#define PPFM_TILE_MMAD_SEL 3
#endif"""

if "PPFM_TILE_MMAD_SEL" not in src:
    assert old_default in src, "default macro block not found"
    src = src.replace(old_default, new_default, 1)

src = src.replace(
    "#if PPFM_TILE_MMAD\n        RunTiledNT(wBf_, hBf_, vTmpF_, CV_BT, CV_V, CV_K);",
    "#if (PPFM_TILE_MMAD && (PPFM_TILE_MMAD_SEL & 1))\n"
    "        RunTiledNT(wBf_, hBf_, vTmpF_, CV_BT, CV_V, CV_K);",
    1,
)
src = src.replace(
    "#if PPFM_TILE_MMAD\n        RunTiledNT(wBf_, mBf_, t1F_, CV_BT, CV_K, CV_K);",
    "#if (PPFM_TILE_MMAD && (PPFM_TILE_MMAD_SEL & 2))\n"
    "        RunTiledNT(wBf_, mBf_, t1F_, CV_BT, CV_K, CV_K);",
    1,
)

open(path, "w").write(src)
print("patched", path)
