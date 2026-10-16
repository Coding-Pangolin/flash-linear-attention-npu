#!/usr/bin/env python3
"""Stable-ABI 适配层 vs ctypes 参考：同输入两次调用，比 h/m 两半 + 对 CPU 标杆。"""

from __future__ import annotations

import sys

import torch
import torch_npu  # noqa: F401

T, HK, HV, K, V, BT = 256, 2, 4, 128, 128, 64
CU = [0, T]
REF_DIR = ("/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/gdn/"
           "chunk_gdn_fwd/pre_process_fwd_kernel_merged/reference")


def main() -> int:
    gen = torch.Generator().manual_seed(7)
    k = torch.nn.functional.normalize(torch.randn(T, HK, K, generator=gen), dim=-1).bfloat16()
    u = torch.randn(T, HV, V, generator=gen).bfloat16()
    beta = torch.rand(T, HV, 1, generator=gen) * 0.02
    head_of_k = torch.arange(HV) // (HV // HK)
    w = (beta * k[:, head_of_k].float()).bfloat16()
    nblk = -(-T // BT)
    g = (-0.013 / BT * (1 + torch.rand(nblk, BT, HV, generator=gen) * 0.5))
    g = g.cumsum(1).reshape(-1, HV)[:T].contiguous()

    def bnsd(x):
        return x.movedim(1, 0).unsqueeze(0).to("npu:0").contiguous()

    kd, wd, ud, gd = bnsd(k), bnsd(w), bnsd(u), bnsd(g)

    import fla_npu.ops.ascendc as A
    from fla_npu.ops.ascendc import _aclnn_ctypes as ct

    hm_stable = A.npu_pre_process_fwd_kernel_merged(kd, wd, ud, g=gd, cu_seqlens=CU, chunk_size=BT)
    torch.npu.synchronize()
    hm_ctypes = ct.npu_pre_process_fwd_kernel_merged(kd, wd, ud, g=gd, cu_seqlens=CU, chunk_size=BT)
    torch.npu.synchronize()

    a = hm_stable.float().cpu()
    b = hm_ctypes.float().cpu()
    print(f"backend[op] = {A.BACKENDS.get('npu_pre_process_fwd_kernel_merged')}")
    print(f"shapes: stable={tuple(a.shape)} ctypes={tuple(b.shape)}")
    d = (a - b).abs()
    print(f"stable vs ctypes: max_abs={float(d.max()):.3e}")
    for name, sl in (("h", slice(0, V)), ("m", slice(V, V + K))):
        print(f"  {name}: max_abs={float(d[..., sl].max()):.3e}")

    sys.path.insert(0, REF_DIR)
    from reference import pre_process_fwd_kernel_merged as ref
    want = ref(k, u, w, g=g, chunk_size=BT, cu_seqlens=CU)
    for tag, got in (("stable", a[0]), ("ctypes", b[0])):
        dd = (got - want).abs()
        ok = dd <= (1.5e-2 + 2e-3 * want.abs())
        print(f"{tag} vs CPU ref: matched={float(ok.float().mean()):.6f} max_abs={float(dd.max()):.3e} "
              f"h={float(dd[..., :V].max()):.2e} m={float(dd[..., V:].max()):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
