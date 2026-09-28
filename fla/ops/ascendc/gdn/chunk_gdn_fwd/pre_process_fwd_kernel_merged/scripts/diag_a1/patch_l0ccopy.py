"""A1 根因修复：L0C→GM 的 fixpipe 调用误选了 4 参「批处理」重载。

Catlass::Gemm::Tile::CopyL0CToGmTla（Ascend950, dst=GM+RowMajor）有两个重载：
  (dst, src, uint8_t unitFlag = 0)                     <- 正常单块搬运
  (dst, src, uint32_t l0Batch, uint32_t dstNdStride)   <- ND 批处理变体

代码里写的是 copyC(bC, tL0C, 0, 0)，于是命中后者且 l0Batch=0，
SetFixpipeNz2ndFlag(0, ...) ⇒ 一个块都不搬，C 保持初值 0。
"""

import sys

path = sys.argv[1]
src = open(path).read()

old = "        copyC(bC, tL0C, 0, 0);"
new = ("        // ⚠ 必须走 3 参重载 (dst, src, unitFlag)：4 参会误选 (l0Batch, dstNdStride)\n"
       "        //    批处理变体，l0Batch=0 ⇒ fixpipe 一个块都不搬，C 恒为初值 0。\n"
       "        copyC(bC, tL0C, static_cast<uint8_t>(0));")

count = src.count(old)
assert count == 2, f"copyC anchor count={count}"
assert "static_cast<uint8_t>(0));" not in src, "already patched"
src = src.replace(old, new)
open(path, "w").write(src)
print(f"patched {path} ({count} sites)")
