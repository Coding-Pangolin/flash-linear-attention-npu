"""定位 h 半边出错的行块：按 16 行一块统计 matched / max_abs。"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/gdn/"
                   "chunk_gdn_fwd/pre_process_fwd_kernel_merged/reference")
sys.path.insert(0, "/tmp")

from npu_one_case import build_case, compare, to_npu_bnsd  # noqa: E402

T, K, V, BT = 256, 128, 128, 64
CU = [0, 256]


def main() -> int:
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm
    from reference import pre_process_fwd_kernel_merged as ref_ppfm

    dev = "npu:0"
    case = build_case(1000, T, 2, 2, K, V, "gdn", BT=BT)
    npu_in = {k: (to_npu_bnsd(v) if v is not None else None) for k, v in case.items()}
    got = npu_ppfm(npu_in["k"].to(dev), npu_in["w"].to(dev), npu_in["u"].to(dev),
                   cu_seqlens=CU, chunk_size=BT, g=npu_in["g"].to(dev), gk=None)
    torch.npu.synchronize()
    got = got.cpu()[0]
    want = ref_ppfm(case["k"], case["v"], case["w"], g=case["g"], gk=case["gk"],
                    chunk_size=BT, cu_seqlens=CU)

    print("=== h 半边：按 16 行块 ===")
    for hv in range(2):
        print(f"-- hv={hv}")
        for rb in range(0, K, 16):
            s = compare(got[hv, rb:rb + 16, :V], want[hv, rb:rb + 16, :V])
            print(f"   rows[{rb:3d}:{rb+16:3d}) matched={s['matched']:.4f} "
                  f"max_abs={s['max_abs']:.3e}")
    print("=== m 半边：按 16 行块 ===")
    for hv in range(2):
        for rb in range(0, K, 16):
            s = compare(got[hv, rb:rb + 16, V:], want[hv, rb:rb + 16, V:])
            print(f"   hv={hv} rows[{rb:3d}:{rb+16:3d}) matched={s['matched']:.4f} "
                  f"max_abs={s['max_abs']:.3e}")
    print("=== h 半边：按 V 列块 ===")
    for hv in range(2):
        for cb in range(0, V, 16):
            s = compare(got[hv, :, cb:cb + 16], want[hv, :, cb:cb + 16])
            print(f"   hv={hv} cols[{cb:3d}:{cb+16:3d}) matched={s['matched']:.4f} "
                  f"max_abs={s['max_abs']:.3e}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PPFM_OP_DIR", "")
    raise SystemExit(main())
