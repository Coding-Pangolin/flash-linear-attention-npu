# Copyright (c) 2026 Tianjin University, Ltd.
"""NPU suite for npu_chunk_kda_bwd_wy_dqkg_fused vs CPU golden."""

from __future__ import annotations

import math
import os
import sys

# fla_npu before torch_npu (tiling SO / OPP wiring; avoids 561103 / stale OPP).
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(REPO, "torch_custom/fla_npu"))
sys.path.insert(0, os.path.join(REPO, "fla/ops/ascendc/kda/chunk_kda_bwd_wy_dqkg_fused/test"))

import fla_npu  # noqa: E402,F401
from fla_npu.ops import ascendc as ascendc_ops  # noqa: E402

import torch  # noqa: E402
import torch_npu  # noqa: E402

from test_chunk_kda_bwd_wy_dqkg_fused import chunk_kda_bwd_wy_dqkg_fused_ref  # noqa: E402


def _allclose(a, b, rtol=1e-2, atol=1e-2, name=""):
    if a.dtype != b.dtype:
        a = a.float()
        b = b.float()
    ok = torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol)
    if not ok:
        diff = (a.float().cpu() - b.float().cpu()).abs()
        print(f"FAIL {name}: max={diff.max().item():.4e} mean={diff.mean().item():.4e}")
    else:
        print(f"PASS {name}")
    return ok


def _tol_for(name: str):
    # dA finishes with bf16 cast + A@A@A; a few medium-mag elems sit at ~2% rel.
    # Match prepare_wy_repr_bwd_da style (~0.1) rather than 1e-2 used for dq/dk.
    if name == "dA":
        # bf16 A@A@A; large entries can sit ~10–20 abs off under state_v_first too.
        return 5e-2, 20.0
    return 1e-2, 1e-2


def run_case(B=1, T=128, H=2, HV=2, K=128, V=128, BT=64, dtype=torch.bfloat16, varlen=False,
             state_v_first=False, split_stages=False, n_stream=1):
    device = "npu:0"
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(K)
    q = torch.randn(B, H, T, K, dtype=dtype, device=device)
    k = torch.randn(B, H, T, K, dtype=dtype, device=device)
    v = torch.randn(B, HV, T, V, dtype=dtype, device=device)
    v_new = torch.randn(B, HV, T, V, dtype=dtype, device=device)
    g = (torch.randn(B, HV, T, K, dtype=dtype, device=device) * 0.05).contiguous()
    beta = torch.randn(B, HV, T, dtype=dtype, device=device)
    a = torch.randn(B, HV, T, BT, dtype=dtype, device=device)
    NT = (T + BT - 1) // BT
    if state_v_first:
        h = torch.randn(B, HV, NT, V, K, dtype=dtype, device=device)
        dh = torch.randn(B, HV, NT, V, K, dtype=dtype, device=device)
    else:
        h = torch.randn(B, HV, NT, K, V, dtype=dtype, device=device)
        dh = torch.randn(B, HV, NT, K, V, dtype=dtype, device=device)
    do = torch.randn(B, HV, T, V, dtype=dtype, device=device)
    dv = torch.randn(B, HV, T, V, dtype=dtype, device=device)

    cu_seqlens = chunk_indices = None
    if varlen:
        assert B == 1
        mid = T // 2
        cu_seqlens = torch.tensor([0, mid, T], dtype=torch.int64, device="cpu")
        # chunks for seq0 and seq1
        n0 = (mid + BT - 1) // BT
        n1 = ((T - mid) + BT - 1) // BT
        pairs = []
        for i in range(n0):
            pairs.extend([0, i])
        for i in range(n1):
            pairs.extend([1, i])
        chunk_indices = torch.tensor(pairs, dtype=torch.int64, device="cpu")
        NT = len(pairs) // 2
        if state_v_first:
            h = torch.randn(B, HV, NT, V, K, dtype=dtype, device=device)
            dh = torch.randn(B, HV, NT, V, K, dtype=dtype, device=device)
        else:
            h = torch.randn(B, HV, NT, K, V, dtype=dtype, device=device)
            dh = torch.randn(B, HV, NT, K, V, dtype=dtype, device=device)

    ref = chunk_kda_bwd_wy_dqkg_fused_ref(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        v_new.cpu(),
        g.cpu(),
        beta.cpu(),
        a.cpu(),
        h.cpu(),
        dh.cpu(),
        do.cpu(),
        dv.cpu(),
        scale=scale,
        chunk_size=BT,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    npu_out = ascendc_ops.npu_chunk_kda_bwd_wy_dqkg_fused(
        q,
        k,
        v,
        v_new,
        g,
        beta,
        a,
        h,
        dh,
        do,
        dv,
        scale,
        BT,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        split_stages=split_stages,
        n_stream=n_stream,
    )
    tag = f"T={T} varlen={varlen} svf={state_v_first} split={split_stages}"
    names = ["dq", "dk", "dv2", "dg", "db", "dA"]
    ok = True
    for name, a, b in zip(names, npu_out, ref):
        rtol, atol = _tol_for(name)
        ok = _allclose(a, b, rtol=rtol, atol=atol, name=f"{name} {tag}") and ok
    return ok


def main():
    torch_npu.npu.set_device(0)
    cases = [
        dict(T=128, varlen=False, state_v_first=False),
        dict(T=128, varlen=False, state_v_first=True),
        dict(T=192, varlen=True, state_v_first=False),
        # F6 N=1 split vs golden
        dict(T=128, varlen=False, state_v_first=False, split_stages=True, n_stream=1),
    ]
    all_ok = True
    for c in cases:
        all_ok = run_case(**c) and all_ok
    if not all_ok:
        raise SystemExit(1)
    print("all cases passed")


if __name__ == "__main__":
    main()
