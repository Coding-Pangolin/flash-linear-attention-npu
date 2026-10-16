"""调试分支专用：核对 S3 累加出来的 Kw 是否与 K^T W 一致。"""

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

    kw_ref = k[:, 0].float().t() @ w[:, 0].float()          # [K, K]
    dump = got[:32, V:]                                      # Kw 前 32 行
    ref = kw_ref[:32]
    d = (dump - ref).abs()
    print(f"Kw(前32行): max_abs={d.max().item():.3e}  mean_abs={d.mean().item():.3e}")
    print("Kw dump 每 16 列非零个数:",
          [int((dump.abs() > 1e-12).sum(0)[i:i + 16].sum().item()) for i in range(0, K, 16)])
    print("Kw ref  每 16 列非零个数:",
          [int((ref.abs() > 1e-12).sum(0)[i:i + 16].sum().item()) for i in range(0, K, 16)])
    print("行0 dump 88..99:", [round(x, 3) for x in dump[0, 88:100].tolist()])
    print("行0 ref  88..99:", [round(x, 3) for x in ref[0, 88:100].tolist()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
