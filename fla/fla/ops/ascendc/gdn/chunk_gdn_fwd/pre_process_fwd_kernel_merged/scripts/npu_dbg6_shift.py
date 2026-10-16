"""检查 m 结果是否整体错位（行/列平移）。"""

from __future__ import annotations

import torch


def main() -> int:
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm

    torch.manual_seed(0)
    T, HK, HV, K, V = 64, 1, 1, 128, 128
    dev = "npu:0"
    k = torch.randn(T, HK, K).to(torch.bfloat16)
    v = torch.randn(T, HV, V).to(torch.bfloat16)
    w = torch.randn(T, HV, K).to(torch.bfloat16)
    got = npu_ppfm(
        k.movedim(1, 0).unsqueeze(0).to(dev),
        w.movedim(1, 0).unsqueeze(0).to(dev),
        v.movedim(1, 0).unsqueeze(0).to(dev),
        g=torch.zeros(1, HV, T, dtype=torch.float32, device=dev),
        cu_seqlens=[0, T], chunk_size=64,
    ).cpu()[0, 0]
    torch.npu.synchronize()

    got_m = got[:, V:]
    kw_ref = k[:, 0].float().t() @ w[:, 0].float()
    ref = (torch.eye(K) - kw_ref)          # g=0 时 M_c = I - Kw
    for name, cand in (
        ("ref[i]", ref),
        ("ref[i-1] (下移一行)", torch.roll(ref, 1, 0)),
        ("-Kw", -kw_ref),
        ("-Kw[i-1]", torch.roll(-kw_ref, 1, 0)),
    ):
        d = (got_m - cand).abs()
        print(f"got_m vs {name:20s}: max_abs={d.max().item():.3e} mean={d.mean().item():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
