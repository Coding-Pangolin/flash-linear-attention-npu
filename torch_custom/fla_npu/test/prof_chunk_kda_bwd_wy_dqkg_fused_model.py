#!/data/miniconda3/envs/fzy_atk/bin/python
"""Model-shape smoke for msprof: ChunkKdaBwdWyDqkgFused.

Shape: B=1, H=HV=32, T=8192, K=128, V=128, BT=64 (DESIGN model target).
Import fla_npu before torch_npu (tiling SO / 561103).
"""
from __future__ import annotations

import math
import os
import time

import fla_npu.ops.ascendc as ascendc_ops  # noqa: F401 — before torch_npu
import torch
import torch_npu  # noqa: F401


def main() -> None:
    device = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    torch.npu.set_device(device)

    B, H, HV, T, K, V, BT = 1, 32, 32, 8192, 128, 128, 64
    dtype = torch.bfloat16
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(K)
    NT = (T + BT - 1) // BT

    q = torch.randn(B, H, T, K, dtype=dtype, device="npu")
    k = torch.randn(B, H, T, K, dtype=dtype, device="npu")
    v = torch.randn(B, HV, T, V, dtype=dtype, device="npu")
    v_new = torch.randn(B, HV, T, V, dtype=dtype, device="npu")
    g = (torch.randn(B, HV, T, K, dtype=dtype, device="npu") * 0.05).contiguous()
    beta = torch.rand(B, HV, T, dtype=dtype, device="npu")
    a = torch.randn(B, HV, T, BT, dtype=dtype, device="npu")
    h = torch.randn(B, HV, NT, K, V, dtype=dtype, device="npu")
    dh = torch.randn(B, HV, NT, K, V, dtype=dtype, device="npu")
    do = torch.randn(B, HV, T, V, dtype=dtype, device="npu")
    dv = torch.randn(B, HV, T, V, dtype=dtype, device="npu")

    # warmup
    for _ in range(3):
        outs = ascendc_ops.npu_chunk_kda_bwd_wy_dqkg_fused(
            q, k, v, v_new, g, beta, a, h, dh, do, dv, scale, BT, state_v_first=False
        )
        torch.npu.synchronize()

    iters = int(os.environ.get("FLA_NPU_PROF_ITERS", "5"))
    t0 = time.time()
    for _ in range(iters):
        outs = ascendc_ops.npu_chunk_kda_bwd_wy_dqkg_fused(
            q, k, v, v_new, g, beta, a, h, dh, do, dv, scale, BT, state_v_first=False
        )
        torch.npu.synchronize()
    elapsed = time.time() - t0
    dq = outs[0]
    print(
        f"[prof] shape=B{B}_H{H}_HV{HV}_T{T}_K{K}_V{V} BT={BT} dtype={dtype} iters={iters} "
        f"avg={elapsed / iters * 1e3:.3f} ms  dq={tuple(dq.shape)}"
    )


if __name__ == "__main__":
    main()
