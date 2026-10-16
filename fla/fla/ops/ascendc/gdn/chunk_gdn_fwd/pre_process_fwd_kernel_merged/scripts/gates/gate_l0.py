"""L0 静态门禁：事件配对/计数、收敛指标（每次编译后 10 秒内可跑）。

用法: python3 gate_l0.py <op_kernel cpp> [--expect-pipe-all N]
"""

from __future__ import annotations

import re
import sys


def count(src: str, pat: str) -> int:
    return len(re.findall(pat, src))


def main() -> int:
    path = sys.argv[1]
    src = open(path, encoding="utf-8").read()
    expect = None
    if "--expect-pipe-all" in sys.argv:
        expect = int(sys.argv[sys.argv.index("--expect-pipe-all") + 1])

    n = {
        "SetFlag<HardEvent": count(src, r"SetFlag<HardEvent"),
        "WaitFlag<HardEvent": count(src, r"WaitFlag<HardEvent"),
        "CrossCoreSetFlag": count(src, r"CrossCoreSetFlag"),
        "CrossCoreWaitFlag": count(src, r"CrossCoreWaitFlag"),
        "PipeBarrier<PIPE_ALL>": count(src, r"PipeBarrier<PIPE_ALL>"),
        "TQue": count(src, r"\bTQue\b"),
        "TBuf": count(src, r"\bTBuf\b"),
    }
    for k, v in n.items():
        print(f"  {k:24s} {v}")

    ok = True
    if n["SetFlag<HardEvent"] != n["WaitFlag<HardEvent"]:
        print("  [FAIL] HardEvent Set/Wait 数量不等")
        ok = False
    if expect is not None and n["PipeBarrier<PIPE_ALL>"] > expect:
        print(f"  [FAIL] PIPE_ALL 计数 {n['PipeBarrier<PIPE_ALL>']} > 期望上限 {expect}")
        ok = False
    m = re.search(r"#define PPFM_TILE_MMAD (\d+)", src)
    if m:
        print(f"  PPFM_TILE_MMAD = {m.group(1)}")
    m = re.search(r"#define PPFM_DIAG (\d+)", src)
    if m:
        print(f"  PPFM_DIAG = {m.group(1)}（必须为 0）")
        if m.group(1) != "0":
            print("  [FAIL] PPFM_DIAG 未关")
            ok = False
    print("L0_PASS" if ok else "L0_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
