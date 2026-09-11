"""Generated stable adapters: parity against the ctypes reference.

usage: FLA_NPU_STABLE_LIB=... python tests/regression_stable_abi_generated.py
"""
from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.ascendc import _aclnn_ctypes as ct  # noqa: E402
from fla_npu.ops.ascendc import _stable  # noqa: E402

STABLE_LIB = os.environ.get("FLA_NPU_STABLE_LIB", "")


def stream():
    return int(torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device()))


def host_ints(values):
    """int[] arguments travel as host int64 tensors (no list support in the
    stable conversions)."""

    if values is None:
        return None
    return torch.tensor(list(values), dtype=torch.int64, device="cpu")


def diff_of(a, b):
    if a is None or b is None:
        return 0.0 if (a is None and b is None) else float("inf")
    assert tuple(a.shape) == tuple(b.shape), (tuple(a.shape), tuple(b.shape))
    return float((a.float() - b.float()).abs().max().item())


def case_fast_gelu():
    x = torch.randn(64, 128, dtype=torch.bfloat16, device="npu")
    ref = ct.npu_fast_gelu_custom(x)
    got = torch.ops.fla_npu_thin.npu_fast_gelu_custom(x, stream())
    torch.npu.synchronize()
    return "fast_gelu", diff_of(ref, got)


def case_kda_gate_cumsum():
    B, T, H, K = 2, 64, 4, 128
    g = -torch.rand(B, H, T, K, dtype=torch.float32, device="npu") * 5 - 1e-3
    a_log = torch.randn(H, dtype=torch.float32, device="npu") * 0.1
    dt_bias = torch.randn(H * K, dtype=torch.float32, device="npu") * 0.5 - 3.0
    ref = ct.npu_kda_gate_cumsum(g, 64, A_log=a_log, dt_bias=dt_bias,
                                 use_gate_in_kernel=True, safe_gate=True,
                                 lower_bound=-1.0)
    got = torch.ops.fla_npu_thin.npu_kda_gate_cumsum(
        g, a_log, dt_bias, None, 64, True, True, -1.0, stream())
    torch.npu.synchronize()
    return "kda_gate_cumsum", diff_of(ref, got)


def case_scaled_dot_kkt():
    B, H, T, K, cs = 1, 4, 128, 128, 64
    k = torch.randn(B, H, T, K, dtype=torch.bfloat16, device="npu") * 0.05
    g = torch.randn(B, H, T, dtype=torch.float32, device="npu")
    beta = torch.rand(B, H, T, dtype=torch.float32, device="npu")
    ref = ct.npu_chunk_scaled_dot_kkt(k, g, beta, chunk_size=cs)
    got = torch.ops.fla_npu_thin.npu_chunk_scaled_dot_kkt(
        k, g, beta, None, None, cs, stream())
    torch.npu.synchronize()
    return "chunk_scaled_dot_kkt", diff_of(ref, got)


def case_bwd_dv_local():
    B, H, T, K, cs = 1, 4, 128, 128, 64
    q = torch.randn(B, H, T, K, dtype=torch.bfloat16, device="npu") * 0.05
    k = torch.randn(B, H, T, K, dtype=torch.bfloat16, device="npu") * 0.05
    d_o = torch.randn(B, H, T, K, dtype=torch.bfloat16, device="npu") * 0.05
    g = torch.randn(B, H, T, dtype=torch.float32, device="npu")
    ref = ct.npu_chunk_bwd_dv_local(q, k, d_o, g, K ** -0.5, cs)
    got = torch.ops.fla_npu_thin.npu_chunk_bwd_dv_local(
        q, k, d_o, g, None, None, None, None, K ** -0.5, cs, stream())
    torch.npu.synchronize()
    return "chunk_bwd_dv_local", diff_of(ref, got)


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260911)
    if not STABLE_LIB:
        raise SystemExit("set FLA_NPU_STABLE_LIB")
    _stable.load()
    failures = []
    for case in (case_fast_gelu, case_kda_gate_cumsum, case_scaled_dot_kkt,
                 case_bwd_dv_local):
        try:
            name, diff = case()
        except Exception as exc:  # noqa: BLE001
            failures.append((case.__name__, f"{type(exc).__name__}: {exc}"[:140]))
            print(f"FAIL {case.__name__}: {type(exc).__name__}: {str(exc)[:140]}")
            continue
        status = "PASS" if diff == 0.0 else f"DIFF {diff}"
        print(f"{status} {name}")
        if diff != 0.0:
            failures.append((name, diff))
    print("ALL PASS: generated stable adapters" if not failures
          else f"FAILURES: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
