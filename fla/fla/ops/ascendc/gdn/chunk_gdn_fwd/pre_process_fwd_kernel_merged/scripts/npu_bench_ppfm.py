"""NPU 侧计时：当前 v1 实现的 kernel 耗时（用于量化与 1.0x H20 目标的差距）。"""

from __future__ import annotations

import time

import torch
import torch_npu  # noqa: F401


def bench(T: int, HK: int, HV: int, K: int, V: int, BT: int, warmup: int = 5,
          repeat: int = 20):
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm

    dev = "npu:0"
    gen = torch.Generator().manual_seed(0)
    k = torch.nn.functional.normalize(torch.randn(T, HK, K, generator=gen), dim=-1).bfloat16()
    v = torch.randn(T, HV, V, generator=gen).bfloat16()
    beta = torch.rand(T, HV, 1, generator=gen) * 0.02
    head_of_k = torch.arange(HV) // (HV // HK)
    w = (beta * k[:, head_of_k].float()).bfloat16()
    nblk = -(-T // BT)
    g = (-0.013 / BT * (1 + torch.rand(nblk, BT, HV, generator=gen) * 0.5))
    g = g.cumsum(1).reshape(-1, HV)[:T].contiguous()

    kd = k.movedim(1, 0).unsqueeze(0).to(dev)
    wd = w.movedim(1, 0).unsqueeze(0).to(dev)
    vd = v.movedim(1, 0).unsqueeze(0).to(dev)
    gd = g.movedim(1, 0).unsqueeze(0).to(dev)
    call = lambda: npu_ppfm(kd, wd, vd, g=gd, cu_seqlens=[0, T], chunk_size=BT)

    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        call()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / repeat
    # 该算子的 MAC 数（与 H20 基准同口径，含上游列块冗余）
    nchunk = -(-T // BT)
    mac = HV * nchunk * (K * V + K * BT + K * K + K * K)
    print(f"T={T} HK={HK} HV={HV} K={K} V={V} BT={BT}: {ms:.3f} ms/call  "
          f"({mac / (ms * 1e-3) / 1e12:.2f} TMAC/s 有效)")
    return ms


def main() -> int:
    bench(1024, 4, 4, 128, 128, 64)
    bench(1024, 8, 8, 128, 128, 64)
    bench(4096, 8, 8, 128, 128, 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
