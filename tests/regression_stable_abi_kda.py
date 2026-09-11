"""Stable-ABI 竖切验证（Phase 3 第二个算子）：npu_recurrent_kda。

覆盖双输出（out + 可选 final_state）与 optional 输入；用法同
regression_stable_abi.py。
"""
from __future__ import annotations

import os
import time

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.ascendc import _aclnn_ctypes as ct  # noqa: E402
from fla_npu.ops.ascendc import _stable, _wrap_mutable_direct_op  # noqa: E402

STABLE_LIB = os.environ.get("FLA_NPU_STABLE_LIB", "")
B, T, H, HV, K, V = 2, 2, 2, 4, 128, 128


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


def kda_inputs():
    dt = torch.bfloat16
    q = torch.randn(B, T, H, K, dtype=dt, device="npu") * 0.05
    k = torch.randn(B, T, H, K, dtype=dt, device="npu") * 0.05
    v = torch.randn(B, T, HV, V, dtype=dt, device="npu") * 0.05
    g = -torch.rand(B, T, HV, K, dtype=torch.float32, device="npu") * 5 - 1e-3
    beta = torch.rand(B, T, HV, dtype=torch.float32, device="npu") * 0.8 + 0.1
    cu = torch.tensor([0, T, 2 * T], dtype=torch.int64, device="npu")
    torch.npu.synchronize()

    def make_state():
        return torch.zeros(B, HV, V, K, dtype=torch.float32, device="npu")

    kw = dict(cu_seqlens=cu, scale=K ** -0.5, layout="BSND",
              state_v_first=True)
    return q, k, v, g, beta, make_state, kw


def call_stable(q, k, v, g, beta, state, kw, **overrides):
    args = dict(cu_seqlens=kw["cu_seqlens"], scale=kw["scale"],
                layout=kw["layout"], state_v_first=kw["state_v_first"],
                output_final_state=True)
    args.update(overrides)
    return _stable.npu_recurrent_kda(q, k, v, g, beta, state, **args)


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260911)
    if not STABLE_LIB:
        raise SystemExit("set FLA_NPU_STABLE_LIB to the built libfla_npu_thin.so")
    _stable.load()
    if not hasattr(torch.ops.fla_npu_thin, "npu_recurrent_kda"):
        raise SystemExit("stable library has no npu_recurrent_kda registered")
    print(f"loaded {STABLE_LIB}")
    q, k, v, g, beta, make_state, kw = kda_inputs()

    # --- T1 parity vs ctypes (inplace, two outputs) --------------------------
    state_c = make_state()
    state_s = make_state()
    ref = ct.npu_recurrent_kda(q, k, v, g, beta, state_c,
                               output_final_state=True, **kw)
    got = call_stable(q, k, v, g, beta, state_s, kw)
    torch.npu.synchronize()
    assert len(ref) == 2 and len(got) == 2, f"arity {len(ref)} vs {len(got)}"
    for index, (a, b) in enumerate(zip(ref, got)):
        if a is None or b is None:
            assert a is None and b is None, f"T1[{index}]: None mismatch"
            continue
        assert tuple(a.shape) == tuple(b.shape), f"T1[{index}]: shape"
        diff = float((a.float() - b.float()).abs().max().item())
        assert diff == 0.0, f"T1[{index}]: diff={diff}"
    diff_state = float((state_c.float() - state_s.float()).abs().max().item())
    assert diff_state == 0.0, f"T1 state diff={diff_state}"
    print(f"PASS T1 kda parity (out/final_state/state all 0.0)")

    # --- T2 mutation contract ------------------------------------------------
    state_op = _wrap_mutable_direct_op("npu_recurrent_kda",
                                       _stable.npu_recurrent_kda)
    state_w = make_state()
    before = int(state_w._version)
    state_op(q, k, v, g, beta, state_w, output_final_state=True, **kw)
    torch.npu.synchronize()
    assert int(state_w._version) == before + 1, "T2: inplace must bump version"
    state_n = make_state()
    before = int(state_n._version)
    state_op(q, k, v, g, beta, state_n, inplace_final_state=False,
             output_final_state=True, **kw)
    torch.npu.synchronize()
    assert int(state_n._version) == before, "T2: non-inplace must not bump"
    print("PASS T2 kda mutation contract (inplace +1, non-inplace +0)")

    # --- T5 host A/B ---------------------------------------------------------
    from fla_npu.ops.ascendc import _thin

    state_ct = make_state()
    state_pb = make_state()
    state_st = make_state()
    with torch.no_grad():
        a = bench(lambda: ct.npu_recurrent_kda(q, k, v, g, beta, state_ct,
                                               output_final_state=True, **kw))
        b = bench(lambda: _thin.npu_recurrent_kda(
            q, k, v, g, beta, state_pb, output_final_state=True, **kw))
        c = bench(lambda: call_stable(q, k, v, g, beta, state_st, kw))
        d = bench(lambda: state_op(q, k, v, g, beta, state_st,
                                   output_final_state=True, **kw))
    print("T5 host P50 (ms):")
    print(f"  1 ctypes                     {a:.4f}")
    print(f"  2 pybind thin                {b:.4f}")
    print(f"  3 stable                     {c:.4f}")
    print(f"  4 stable + mutation contract {d:.4f}")
    print("ALL PASS: stable-abi kda")


if __name__ == "__main__":
    main()
