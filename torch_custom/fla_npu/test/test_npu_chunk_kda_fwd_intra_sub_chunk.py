#!/usr/bin/env python3
# Copyright (c) 2026 Tianjin University, Ltd.
"""NPU single-op test for npu_chunk_kda_fwd_intra_sub_chunk (requires installed wheel + NPU)."""

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Sequence

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_GOLDEN_DIR = os.path.join(
    _REPO_ROOT, "fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/test"
)
sys.path.insert(0, _GOLDEN_DIR)
from test_chunk_kda_fwd_intra_sub_chunk import (  # noqa: E402
    chunk_kda_fwd_intra_sub_chunk_ref,
    prepare_chunk_indices,
)


def _make_gate(B: int, HV: int, T: int, K: int, dtype: torch.dtype, mode: str) -> torch.Tensor:
    if mode == "lin_strong":
        # Strong decaying gate (stresses (I-L)^{-1} magnitude).
        g = -torch.linspace(0, 30, T).view(1, 1, T, 1).expand(B, HV, T, K)
    elif mode == "lin_mild":
        g = -torch.linspace(0, 8, T).view(1, 1, T, 1).expand(B, HV, T, K)
    elif mode == "rand":
        g = torch.randn(B, HV, T, K)
    else:
        raise ValueError(mode)
    return g.to(dtype).contiguous()


def _run_case(
    B: int,
    H: int,
    T: int,
    K: int,
    BT: int,
    *,
    HV: Optional[int] = None,
    varlen: bool = False,
    cu: Optional[Sequence[int]] = None,
    dtype: torch.dtype = torch.bfloat16,
    gate: str = "lin_strong",
    seed: int = 0,
    aqk_tol: float = 5e-2,
    akkd_rel_tol: float = 1e-3,
    ref_dtype: torch.dtype = torch.float64,
):
    import time

    import fla_npu.ops.ascendc as ascendc_ops  # noqa: F401 — OPP path must be ready

    if HV is None:
        HV = H
    assert HV >= H and HV % H == 0

    t0 = time.time()
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, K, dtype=dtype)
    k = torch.randn(B, H, T, K, dtype=dtype)
    g = _make_gate(B, HV, T, K, dtype, gate)
    beta = torch.rand(B, HV, T, dtype=dtype)
    scale = 1.0 / math.sqrt(K)

    cu_t = None
    idx = None
    if varlen or cu is not None:
        assert B == 1
        if cu is None:
            mid = T // 2
            cu = [0, mid, T]
        assert cu[0] == 0 and cu[-1] == T
        cu_t = torch.tensor(list(cu), dtype=torch.long)
        idx = torch.tensor(prepare_chunk_indices(cu_t, BT), dtype=torch.long)

    t_ref0 = time.time()
    aqk_ref, akkd_ref = chunk_kda_fwd_intra_sub_chunk_ref(
        q.float(),
        k.float(),
        g.float(),
        beta.float(),
        scale,
        BT,
        cu_t,
        idx,
        dtype=ref_dtype,
    )
    t_ref = time.time() - t_ref0

    t_npu0 = time.time()
    aqk_n, akkd_n = ascendc_ops.npu_chunk_kda_fwd_intra_sub_chunk(
        q.npu(),
        k.npu(),
        g.npu(),
        beta.npu(),
        scale,
        BT,
        cu_seqlens=None if cu_t is None else cu_t.tolist(),
        chunk_indices=None if idx is None else idx.tolist(),
    )
    torch.npu.synchronize()
    t_npu = time.time() - t_npu0

    aqk_err = (aqk_n.float().cpu() - aqk_ref.float()).abs().max().item()
    akkd_n_c = akkd_n.float().cpu()
    akkd_ref_c = akkd_ref.float()
    akkd_abs = (akkd_n_c - akkd_ref_c).abs()
    akkd_err = akkd_abs.max().item()
    # (I-L)^{-1} can be huge under strong gates; use relative tolerance.
    akkd_rel = (akkd_abs / akkd_ref_c.abs().clamp_min(1.0)).max().item()
    tag = f"cu={list(cu)}" if cu is not None else "dense"
    print(
        f"[case] B={B} H={H} HV={HV} T={T} K={K} BT={BT} {tag} gate={gate} dtype={dtype} ref={ref_dtype} "
        f"aqk_max_err={aqk_err:.6g} akkd_max_err={akkd_err:.6g} akkd_max_rel={akkd_rel:.6g} "
        f"t_ref={t_ref:.2f}s t_npu={t_npu:.2f}s total={time.time()-t0:.2f}s"
    )
    assert aqk_n.shape == (B, HV, T, BT) and akkd_n.shape == (B, HV, T, 16)
    assert torch.isfinite(aqk_n).all() and torch.isfinite(akkd_n).all()
    assert aqk_err < aqk_tol, aqk_err
    assert akkd_rel < akkd_rel_tol, (akkd_err, akkd_rel)


def main():
    # Import fla_npu before torch_npu so ASCEND_CUSTOM_OPP_PATH is set
    # before GE loads tiling SOs (otherwise GetWorkspaceSize -> 561103).
    import fla_npu.ops.ascendc  # noqa: F401
    import torch_npu  # noqa: F401

    device = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    torch.npu.set_device(device)

    # FLA_NPU_ONLY_MODEL=1 → skip smoke/small cases, only model-scale.
    # FLA_NPU_ONLY_GVA=1 → only GVA smoke cases (after rebuild).
    only_model = os.environ.get("FLA_NPU_ONLY_MODEL", "0") == "1"
    only_gva = os.environ.get("FLA_NPU_ONLY_GVA", "0") == "1"

    if only_gva:
        _run_case(1, 2, 64, 128, 64, HV=4)  # GVA 2x
        _run_case(1, 4, 64, 128, 64, HV=8)  # GVA 2x
        _run_case(1, 2, 96, 128, 64, HV=4, varlen=True)
        # Larger GVA: Akkd fp32 accumulate vs fp64 golden (same class as K=256 / long-T).
        _run_case(1, 8, 128, 128, 64, HV=16, gate="lin_mild", akkd_rel_tol=5e-2)
        _run_case(1, 4, 80, 128, 32, HV=8)  # BT=32 + tail + GVA
        print("all GVA cases passed")
        return

    if not only_model:
        # ---- original smoke ----
        _run_case(1, 2, 64, 128, 64)
        _run_case(1, 2, 64, 128, 32)
        _run_case(1, 2, 128, 128, 128)
        _run_case(1, 2, 96, 128, 64, varlen=True)

        # ---- GVA (HV > H) ----
        _run_case(1, 2, 64, 128, 64, HV=4)
        _run_case(1, 4, 64, 128, 64, HV=8)
        _run_case(1, 2, 96, 128, 64, HV=4, varlen=True)
        _run_case(1, 8, 128, 128, 64, HV=16, gate="lin_mild", akkd_rel_tol=5e-2)
        _run_case(1, 4, 80, 128, 32, HV=8)
        # ---- dense: batch / heads / K / tail / BT ----
        _run_case(2, 2, 64, 128, 64)  # B>1
        _run_case(1, 4, 64, 128, 64)  # more heads
        _run_case(1, 2, 64, 64, 64)  # smaller K
        # K=256: fp32 forward-sub vs fp64 golden drifts more (Akkd ~1e12); keep aqk tight, loosen akkd.
        _run_case(1, 2, 64, 256, 64, gate="lin_mild", akkd_rel_tol=1e-2)
        _run_case(1, 2, 48, 128, 64)  # T not multiple of BT (tail)
        _run_case(1, 2, 80, 128, 64)  # T > BT with tail
        _run_case(1, 2, 32, 128, 32)  # BT=32 full
        _run_case(1, 2, 40, 128, 32)  # BT=32 tail
        # Longer / multi-chunk: more forward-sub steps → slightly looser akkd rel vs fp64.
        _run_case(1, 2, 256, 128, 128, gate="lin_mild", akkd_rel_tol=5e-3)
        _run_case(1, 2, 200, 128, 128, gate="lin_mild", akkd_rel_tol=5e-3)

        # ---- gates / dtype ----
        _run_case(1, 2, 64, 128, 64, gate="lin_mild")
        _run_case(1, 2, 64, 128, 64, gate="rand")
        _run_case(1, 2, 64, 128, 64, dtype=torch.float16, akkd_rel_tol=5e-3)

        # ---- varlen: unequal / multi-seq / short / BT variants ----
        _run_case(1, 2, 96, 128, 64, cu=[0, 40, 96])  # unequal lens
        _run_case(1, 2, 120, 128, 64, cu=[0, 16, 80, 120])  # 3 sequences
        _run_case(1, 2, 20, 128, 64, cu=[0, 8, 20])  # short seqs (< BC and < BT)
        _run_case(1, 2, 96, 128, 32, cu=[0, 48, 96])  # varlen BT=32
        _run_case(1, 2, 160, 128, 128, cu=[0, 70, 160], gate="lin_mild", akkd_rel_tol=5e-3)
        _run_case(1, 2, 256, 128, 64, cu=[0, 100, 180, 256], gate="lin_mild", akkd_rel_tol=5e-3)

        _run_case(1, 8, 512, 128, 64, gate="lin_mild", akkd_rel_tol=5e-3)

    # ---- model target shapes (README: B=1,T=8192,H=32,K=128,BT=64) ----
    # Large shapes: compare vs fp32 golden (matches NPU accumulate); fp64 alone already ~1e-2.
    _run_case(1, 16, 2048, 128, 64, gate="lin_mild", ref_dtype=torch.float32, akkd_rel_tol=5e-3)
    _run_case(1, 32, 4096, 128, 64, gate="lin_mild", ref_dtype=torch.float32, akkd_rel_tol=5e-3)
    _run_case(1, 32, 8192, 128, 64, gate="lin_mild", ref_dtype=torch.float32, akkd_rel_tol=5e-3)
    _run_case(1, 32, 8192, 128, 64, gate="lin_strong", ref_dtype=torch.float32, akkd_rel_tol=1e-2)
    _run_case(1, 32, 8192, 128, 64, gate="rand", ref_dtype=torch.float32, akkd_rel_tol=5e-3)

    print("all cases passed")


if __name__ == "__main__":
    main()
