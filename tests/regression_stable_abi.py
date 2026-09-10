"""Stable-ABI 竖切验证：recurrent GDR 走 torch.ops 注册的 stable 实现。

用法（221/241，已 build 出 libfla_npu_thin.so）：
    FLA_NPU_STABLE_LIB=/path/libfla_npu_thin.so PYTHONPATH=<env> \
        python tests/regression_stable_abi.py

覆盖计划里的 T1（parity）、T2（mutation 契约）、T5（三后端 host A/B）。
"""
from __future__ import annotations

import os
import time

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.ascendc import _aclnn_ctypes as ct  # noqa: E402

STABLE_LIB = os.environ.get("FLA_NPU_STABLE_LIB", "")
OP_NAME = "npu_recurrent_gated_delta_rule"


def pct(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def bench(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    torch.npu.synchronize()
    return pct(ts, 0.5)


def gdr_inputs(batch=8, nk=8, nv=16, dim=128, gap=16384, offset=12288):
    block_stride = nv * dim * dim + gap
    backing = torch.empty(batch * block_stride * 4, dtype=torch.int8,
                          device="npu")

    def make_state():
        state = torch.as_strided(
            backing.clone().view(torch.float32),
            size=(batch, nv, dim, dim),
            stride=(block_stride, dim * dim, dim, 1),
            storage_offset=offset)
        state.zero_()
        return state

    norm = lambda t: torch.nn.functional.normalize(t, p=2, dim=-1)  # noqa: E731
    query = norm(torch.randn(batch, nk, dim, device="npu")).to(torch.bfloat16)
    key = norm(torch.randn(batch, nk, dim, device="npu")).to(torch.bfloat16)
    value = torch.randn(batch, nv, dim, dtype=torch.bfloat16, device="npu")
    beta = torch.rand(batch, nv, dtype=torch.bfloat16, device="npu")
    g = torch.rand(batch, nv, dtype=torch.float32, device="npu")
    asl = torch.tensor([0] + [1] * batch, dtype=torch.int32, device="npu")
    ssi = torch.arange(batch, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    kw = dict(beta=beta, g=g, scale=dim ** -0.5, actual_seq_lengths=asl,
              ssm_state_indices=ssi, num_accepted_tokens=None)
    return query, key, value, make_state, kw


def call_stable(query, key, value, state, stream, kw):
    return torch.ops.fla_npu_thin.npu_recurrent_gated_delta_rule(
        query, key, value, state, kw["beta"], kw["actual_seq_lengths"],
        kw["ssm_state_indices"], kw["num_accepted_tokens"], kw["g"], None,
        float(kw["scale"]), int(stream))


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260911)
    if not STABLE_LIB:
        raise SystemExit("set FLA_NPU_STABLE_LIB to the built libfla_npu_thin.so")
    torch.ops.load_library(STABLE_LIB)
    print(f"loaded {STABLE_LIB}")

    query, key, value, make_state, kw = gdr_inputs()
    stream = int(torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device()))

    # --- T1 parity vs ctypes -------------------------------------------------
    state_c = make_state()
    state_s = make_state()
    out_c = ct.npu_recurrent_gated_delta_rule(query, key, value, state_c, **kw)
    out_s = call_stable(query, key, value, state_s, stream, kw)
    torch.npu.synchronize()
    diff_out = float((out_c.float() - out_s.float()).abs().max().item())
    diff_state = float((state_c.float() - state_s.float()).abs().max().item())
    assert diff_out == 0.0, f"T1 out diff = {diff_out}"
    assert diff_state == 0.0, f"T1 state diff = {diff_state}"
    print(f"PASS T1 parity (out={diff_out}, state={diff_state})")

    # --- T2 mutation contract ------------------------------------------------
    state_m = make_state()
    before = int(state_m._version)
    call_stable(query, key, value, state_m, stream, kw)
    torch.npu.synchronize()
    after = int(state_m._version)
    print(f"T2 state._version {before} -> {after} "
          f"({'bumped' if after > before else 'NOT bumped'})")
    grad_state = state_m.detach().clone().requires_grad_(True)
    try:
        call_stable(query, key, value, grad_state, stream, kw)
        torch.npu.synchronize()
        print("T2 requires_grad state: accepted (no rejection)")
    except RuntimeError as exc:
        print(f"T2 requires_grad state: rejected ({str(exc)[:80]})")

    # --- T5 host A/B ---------------------------------------------------------
    state_t = make_state()
    from fla_npu.ops.ascendc import _thin

    with torch.no_grad():
        a = bench(lambda: ct.npu_recurrent_gated_delta_rule(
            query, key, value, state_c, **kw))
        b = bench(lambda: _thin.npu_recurrent_gated_delta_rule(
            query, key, value, state_t, **kw))
        c = bench(lambda: call_stable(query, key, value, state_s, stream, kw))
    print(f"T5 host P50  ctypes={a:.4f} ms  pybind-thin={b:.4f} ms  "
          f"stable={c:.4f} ms")
    print("ALL PASS: stable-abi spike")


if __name__ == "__main__":
    main()
