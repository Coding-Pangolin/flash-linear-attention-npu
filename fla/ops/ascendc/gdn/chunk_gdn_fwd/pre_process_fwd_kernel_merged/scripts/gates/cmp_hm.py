"""L1 位级比对：python cmp_hm.py <dirA> <dirB> —— 要求逐元素完全一致。"""

from __future__ import annotations

import os
import sys

import torch


def main() -> int:
    a, b = sys.argv[1], sys.argv[2]
    bad = 0
    files = sorted(f for f in os.listdir(a) if f.endswith(".pt"))
    for f in files:
        x = torch.load(os.path.join(a, f)).float()
        y = torch.load(os.path.join(b, f)).float()
        d = (x - y).abs()
        nz = int((d > 0).sum().item())
        print(f"{f}: shape={tuple(x.shape)} max|diff|={d.max().item():.3e} 不同元素={nz}")
        if nz:
            bad += 1
    print("BIT_IDENTICAL" if bad == 0 else f"DIFF_IN_{bad}_OF_{len(files)}_FILES")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
