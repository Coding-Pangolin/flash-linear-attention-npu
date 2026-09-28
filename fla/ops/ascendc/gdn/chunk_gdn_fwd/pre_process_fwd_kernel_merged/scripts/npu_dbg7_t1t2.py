"""调试分支专用：核对 T1 = W_c @ m 与 T2 = L_c^T @ T1。"""

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

    w_ref = w[:, 0].float()                      # T1 期望值
    kw_ref = k[:, 0].float().t() @ w[:, 0].float()   # T2 期望值
    t1 = got[:T, :V]
    t2 = got[:, V:]
    for tag, dump, ref in (("T1", t1, w_ref), ("T2", t2, kw_ref)):
        d = (dump - ref).abs()
        print(f"{tag}: max_abs={d.max().item():.3e} mean={d.mean().item():.3e}")
        print(f"   {tag} err 行分布(每16行最大err):",
              [f"{d[i:i+16].max().item():.2e}" for i in range(0, d.shape[0], 16)])
        print(f"   {tag} err 列分布(每16列最大err):",
              [f"{d[:, i:i+16].max().item():.2e}" for i in range(0, K, 16)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
