#!/usr/bin/env python3
"""KDA + GVA（HK < HV）验证。

绕过 _aclnn_ctypes 里 "gk 要求 HK==HV" 的守卫，直接按 wrapper 同构方式下发 aclnn，
再与仓内 CPU 标杆（reference.py，标杆本身支持 GVA）逐半对比。
"""

from __future__ import annotations

import ctypes
import sys

import torch
import torch_npu  # noqa: F401

REF_DIR = ("/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/gdn/"
           "chunk_gdn_fwd/pre_process_fwd_kernel_merged/reference")


def build(T, HK, HV, K, V, BT, seed):
    gen = torch.Generator().manual_seed(seed)
    k = torch.nn.functional.normalize(torch.randn(T, HK, K, generator=gen), dim=-1).bfloat16()
    u = torch.randn(T, HV, V, generator=gen).bfloat16()
    beta = torch.rand(T, HV, 1, generator=gen) * 0.02
    head_of_k = torch.arange(HV) // (HV // HK)
    w = (beta * k[:, head_of_k].float()).bfloat16()
    nblk = -(-T // BT)
    gk = (-0.013 / BT * (1 + torch.rand(nblk, BT, HV, K, generator=gen) * 0.5))
    gk = gk.cumsum(1).reshape(-1, HV, K)[:T].contiguous()
    return k, u, w, gk


def main() -> int:
    T, HK, HV, K, V, BT = 256, 2, 4, 128, 128, 64
    CU = [0, T]
    dev = "npu:0"
    from fla_npu.ops.ascendc import _aclnn_ctypes as ct

    sys.path.insert(0, REF_DIR)
    from reference import pre_process_fwd_kernel_merged as ref

    def bnsd(x):
        return x.movedim(1, 0).unsqueeze(0).contiguous()

    for rep in range(5):
        k, u, w, gk = build(T, HK, HV, K, V, BT, seed=100 + rep)
        kd = bnsd(k).to(dev)
        wd = bnsd(w).to(dev)
        ud = bnsd(u).to(dev)
        gkd = bnsd(gk).to(dev)
        hm = ct._zeros((1, HV, K, V + K), kd, dtype=torch.float32)
        ct._call_aclnn(
            "aclnnPreProcessFwdKernelMerged",
            lambda ctx: [
                ctx.tensor(kd, "k"), ctx.tensor(wd, "w"), ctx.tensor(ud, "u"),
                ctx.tensor(None, "g"), ctx.tensor(gkd, "gk"), ctx.tensor(None, "bg"),
                ctx.tensor(None, "v"), ctx.int_array(CU), ctypes.c_int64(BT),
                ctx.tensor(hm, "hm"),
            ],
            hm,
        )
        torch.npu.synchronize()
        got = hm.float().cpu()[0]
        want = ref(k, u, w, gk=gk, chunk_size=BT, cu_seqlens=CU)
        d = (got - want).abs()
        ok = d <= (1.5e-2 + 2e-3 * want.abs())
        print(f"rep{rep}: KDA+GVA HK={HK} HV={HV} matched={float(ok.float().mean()):.6f} "
              f"max_abs={float(d.max()):.3e}  h={float(d[..., :V].max()):.2e} "
              f"m={float(d[..., V:].max()):.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
