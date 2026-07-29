#!/usr/bin/env python3
"""Simulator smoke for ChunkKdaBwdWyDqkgFused (small shape; model 8192 too slow under sim).

Default: B1 H2 T64 K128 V128 BT64. Override via FLA_SIM_{B,H,T,K,V,BT}.
"""
from __future__ import annotations

import math
import os

import fla_npu.ops.ascendc as ascendc_ops  # noqa: F401 — before torch_npu
import torch
import torch_npu  # noqa: F401


def main() -> None:
    device = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    torch.npu.set_device(device)

    B = int(os.environ.get("FLA_SIM_B", "1"))
    H = int(os.environ.get("FLA_SIM_H", "2"))
    HV = int(os.environ.get("FLA_SIM_HV", str(H)))
    T = int(os.environ.get("FLA_SIM_T", "64"))
    K = int(os.environ.get("FLA_SIM_K", "128"))
    V = int(os.environ.get("FLA_SIM_V", "128"))
    BT = int(os.environ.get("FLA_SIM_BT", "64"))
    warmup = int(os.environ.get("FLA_SIM_WARMUP", "1"))
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

    outs = None
    for _ in range(max(warmup, 1)):
        outs = ascendc_ops.npu_chunk_kda_bwd_wy_dqkg_fused(
            q, k, v, v_new, g, beta, a, h, dh, do, dv, scale, BT, state_v_first=False
        )
        torch.npu.synchronize()
    assert outs is not None
    print(
        f"[sim] shape=B{B}_H{H}_HV{HV}_T{T}_K{K}_V{V} BT={BT} "
        f"dq={tuple(outs[0].shape)} finite_dq={torch.isfinite(outs[0]).all().item()}"
    )


if __name__ == "__main__":
    main()
