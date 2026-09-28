"""用标杆自测同款真实 gate 分布（chunk 内 cumsum、小量级）做精度对拍。"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    sys.path.insert(0, "/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/"
                       "gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged/reference")
    from reference import pre_process_fwd_kernel_merged as ref_ppfm
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm

    import os
    gen = torch.Generator().manual_seed(0)
    T, HK, HV, K, V, BT = int(os.environ.get("PPFM_T", 256)), 2, 2, 128, 128, 64
    dev = "npu:0"
    k = torch.nn.functional.normalize(torch.randn(T, HK, K, generator=gen), dim=-1).bfloat16()
    beta = torch.rand(T, HV, 1, generator=gen) * 0.02
    w = (beta * k[:, torch.arange(HV)]).bfloat16()
    v = torch.randn(T, HV, V, generator=gen).bfloat16()
    n = -(-T // BT)
    # 标杆 self_test 同款：每 chunk 内 cumsum 的负对数衰减
    g = (-0.013 / BT * (1 + torch.rand(T, HV, generator=gen) * 0.5))
    g = g.view(n, BT, HV).cumsum(1).reshape(-1, HV)[:T].contiguous()

    ref = ref_ppfm(k, v, w, g=g, chunk_size=BT, cu_seqlens=[0, T])[0]
    got = npu_ppfm(
        k.movedim(1, 0).unsqueeze(0).to(dev),
        w.movedim(1, 0).unsqueeze(0).to(dev),
        v.movedim(1, 0).unsqueeze(0).to(dev),
        g=g.movedim(1, 0).unsqueeze(0).to(dev),
        cu_seqlens=[0, T], chunk_size=BT,
    ).cpu()[0, 0]
    torch.npu.synchronize()

    for tag, a, b in (("h", got[:, :V], ref[:, :V]), ("m", got[:, V:], ref[:, V:])):
        d = (a - b).abs()
        limit = 1.5e-2 + 2e-3 * b.abs()
        print(f"{tag}: matched={(d <= limit).float().mean().item():.6f} "
              f"max_abs={d.max().item():.3e} MARE={(d / b.abs().clamp_min(1e-6)).mean().item():.3e} "
              f"|ref|max={b.abs().max().item():.3e}")
        bad = (d > limit)
        if bad.any():
            idx = bad.nonzero()
            rows = sorted(set(idx[:, 0].tolist()))
            cols = sorted(set(idx[:, 1].tolist()))
            print(f"   错元素 {int(bad.sum())}/{bad.numel()} 行范围={rows[:6]}…{rows[-3:]} "
                  f"列范围={cols[:6]}…{cols[-3:]}")
            for r, c in idx[:4].tolist():
                print(f"   例 [row={r}, col={c}] NPU={a[r, c].item():.6f} 标杆={b[r, c].item():.6f}")
            if tag == "m":
                print("   NPU m 行0 前4:", [f"{x:.6f}" for x in a[0, :4].tolist()],
                      " 尾4:", [f"{x:.6f}" for x in a[0, -4:].tolist()])
                print("   标杆 m 行0 前4:", [f"{x:.6f}" for x in b[0, :4].tolist()],
                      " 尾4:", [f"{x:.6f}" for x in b[0, -4:].tolist()])
                print("   NPU m 行1 前4:", [f"{x:.6f}" for x in a[1, :4].tolist()],
                      " 尾4:", [f"{x:.6f}" for x in a[1, -4:].tolist()])
                print("   标杆 m 行1 前4:", [f"{x:.6f}" for x in b[1, :4].tolist()],
                      " 尾4:", [f"{x:.6f}" for x in b[1, -4:].tolist()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
