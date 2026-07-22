#!/usr/bin/env python3
"""Model-shape smoke for msprof: ChunkKdaFwdIntraSubChunk Cube path.

Shape: B=1, H=32, T=8192, K=128, BT=64 (README model target).
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

    B, H, T, K, BT = 1, 32, 8192, 128, 64
    dtype = torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(B, H, T, K, dtype=dtype, device="npu")
    k = torch.randn(B, H, T, K, dtype=dtype, device="npu")
    g = (-torch.linspace(0, 8, T, device="npu").view(1, 1, T, 1).expand(B, H, T, K)).to(dtype).contiguous()
    beta = torch.rand(B, H, T, dtype=dtype, device="npu")
    scale = 1.0 / math.sqrt(K)

    # warmup
    for _ in range(3):
        aqk, akkd = ascendc_ops.npu_chunk_kda_fwd_intra_sub_chunk(q, k, g, beta, scale, BT)
        torch.npu.synchronize()

    # timed runs (msprof wraps whole process; keep a few iters)
    iters = int(os.environ.get("FLA_NPU_PROF_ITERS", "5"))
    t0 = time.time()
    for _ in range(iters):
        aqk, akkd = ascendc_ops.npu_chunk_kda_fwd_intra_sub_chunk(q, k, g, beta, scale, BT)
        torch.npu.synchronize()
    elapsed = time.time() - t0
    print(
        f"[prof] shape=({B},{H},{T},{K}) BT={BT} dtype={dtype} iters={iters} "
        f"avg={elapsed/iters*1e3:.3f} ms  aqk={tuple(aqk.shape)} akkd={tuple(akkd.shape)}"
    )


if __name__ == "__main__":
    main()
