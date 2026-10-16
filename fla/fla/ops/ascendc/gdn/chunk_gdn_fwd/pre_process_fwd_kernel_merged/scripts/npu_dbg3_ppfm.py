"""调试分支专用：核对转储出来的 vNewBf_ / wUb_ 是否与数学值一致。"""

from __future__ import annotations

import os
import sys

import torch


def main() -> int:
    sys.path.insert(0, "/usr/local/python3.12.13/lib/python3.12/site-packages")
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

    # 第一 chunk、h 初值 0 => v_new == v（bf16 量化后）
    v_new_ref = v[:, 0].float()
    w_ref = w[:, 0].float()
    dump_v = got[:T, :V]
    dump_w = got[:T, V:]
    for tag, dump, ref in (("vNewBf_", dump_v, v_new_ref), ("wUb_", dump_w, w_ref)):
        d = (dump - ref).abs()
        zdump = (dump.abs() < 1e-12).sum(0)
        zref = (ref.abs() < 1e-12).sum(0)
        print(f"{tag}: max_abs={d.max().item():.3e}  全零列(dump)={int((zdump == T).sum())}"
              f"  全零列(ref)={int((zref == T).sum())}")
        print(f"   每列非零个数(每 16 列均值): "
              f"{[int((dump.abs() > 1e-12).sum(0)[i:i+16].float().mean().item()) for i in range(0, V, 16)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
