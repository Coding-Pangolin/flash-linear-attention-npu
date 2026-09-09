"""Wheel 安装态数值回归：已迁移 thin 算子的 ctypes-vs-thin parity。

用法（221 上，wheel 已 pip install --target envXXX）:
    PYTHONPATH=/path/envXXX python tests/regression_thin_ops.py

每个场景对同一输入分别走 ctypes 与 thin 两条 host 路径，断言每个输出
tensor 的逐元素差为 0（同一 OPP kernel，期望 bitwise 相同）。host P50
仅作参考；合法域说明见 thin-migration-inventory.md。
"""
from __future__ import annotations

import json
import time

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.ascendc import _aclnn_ctypes as ct  # noqa: E402
from fla_npu.ops.ascendc import _thin  # noqa: E402


def pct(vals, q):
    vals = sorted(vals)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def host_p50(fn, n=200):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    torch.npu.synchronize()
    return pct(ts, 0.5)


def assert_parity(name, oc, ot):
    if not isinstance(oc, tuple):
        oc = (oc,)
        ot = (ot,)
    assert len(oc) == len(ot), f"{name}: output count mismatch"
    for i, (a, b) in enumerate(zip(oc, ot)):
        if a is None or b is None:
            assert a is None and b is None, f"{name}[{i}]: None mismatch"
            continue
        assert tuple(a.shape) == tuple(b.shape), f"{name}[{i}]: shape"
        diff = float((a.float() - b.float()).abs().max().item())
        assert diff == 0.0, f"{name}[{i}]: diff={diff}"
    print(f"PASS {name}")


def scenario_fast_gelu():
    x = torch.randn(4, 128, 256, dtype=torch.float16, device="npu")
    torch.npu.synchronize()
    assert_parity("fast_gelu_custom",
                  ct.npu_fast_gelu_custom(x), _thin.npu_fast_gelu_custom(x))
    grad = torch.randn_like(x)
    assert_parity("fast_gelu_custom_backward",
                  ct.npu_fast_gelu_custom_backward(grad, x),
                  _thin.npu_fast_gelu_custom_backward(grad, x))


def scenario_recurrent_gated_delta_rule():
    batch = 8
    num_key_heads, num_value_heads, dim = 8, 16, 128
    gap, offset = 16384, 12288
    inner = num_value_heads * dim * dim
    block_stride = inner + gap

    def make_state():
        backing = torch.empty(batch * block_stride * 4, dtype=torch.int8,
                              device="npu")
        typed = backing.view(torch.float32)
        state = torch.as_strided(
            typed,
            size=(batch, num_value_heads, dim, dim),
            stride=(block_stride, dim * dim, dim, 1),
            storage_offset=offset,
        )
        state.zero_()
        return state

    def norm(t):
        return torch.nn.functional.normalize(t, p=2, dim=-1)

    query = norm(torch.randn(batch, num_key_heads, dim, device="npu")).to(
        torch.bfloat16)
    key = norm(torch.randn(batch, num_key_heads, dim, device="npu")).to(
        torch.bfloat16)
    value = torch.randn(batch, num_value_heads, dim, dtype=torch.bfloat16,
                        device="npu")
    beta = torch.rand(batch, num_value_heads, dtype=torch.bfloat16,
                      device="npu")
    g = torch.rand(batch, num_value_heads, dtype=torch.float32, device="npu")
    actual_seq_lengths = torch.tensor([0] + [1] * batch, dtype=torch.int32,
                                      device="npu")
    ssm_state_indices = torch.arange(batch, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    state_c = make_state()
    state_t = make_state()
    kw = dict(beta=beta, g=g, scale=dim ** -0.5,
              actual_seq_lengths=actual_seq_lengths,
              ssm_state_indices=ssm_state_indices,
              num_accepted_tokens=None)
    out_c = ct.npu_recurrent_gated_delta_rule(query, key, value, state_c, **kw)
    out_t = _thin.npu_recurrent_gated_delta_rule(query, key, value, state_t,
                                                 **kw)
    torch.npu.synchronize()
    assert_parity("recurrent_gated_delta_rule", out_c, out_t)
    assert float((state_c.float() - state_t.float()).abs().max().item()) == 0.0
    print("PASS recurrent_gated_delta_rule(state)")


def scenario_recompute():
    B, Hk, Hv, T, K, V, cs = 1, 2, 4, 256, 128, 256, 64
    dt = torch.float16
    k = torch.randn(B, Hk, T, K, dtype=dt, device="npu")
    v = torch.randn(B, Hv, T, V, dtype=dt, device="npu")
    beta = torch.randn(B, Hv, T, dtype=dt, device="npu")
    A = torch.randn(B, Hv, T, cs, dtype=dt, device="npu")
    g = torch.randn(B, Hv, T, dtype=dt, device="npu")
    torch.npu.synchronize()
    kw = dict(g=g, gk=None, cu_seqlens=None, chunk_indices=None)
    assert_parity("recompute_w_u_fwd",
                  ct.npu_recompute_w_u_fwd(k, v, beta, A, cs, **kw),
                  _thin.npu_recompute_w_u_fwd(k, v, beta, A, cs, **kw))


def scenario_pwy_full():
    B, H, T, K, V, cs = 1, 4, 256, 128, 128, 64
    dt = torch.float16
    k = torch.rand(B, H, T, K, dtype=dt, device="npu")
    v = torch.rand(B, H, T, V, dtype=dt, device="npu")
    beta = torch.rand(B, H, T, dtype=dt, device="npu")
    A = torch.rand(B, H, T, cs, dtype=dt, device="npu")
    dA = torch.rand(B, H, T, cs, dtype=dt, device="npu")
    dw = torch.rand(B, H, T, K, dtype=dt, device="npu")
    du = torch.rand(B, H, T, V, dtype=dt, device="npu")
    g = torch.rand(B, H, T, dtype=dt, device="npu")
    torch.npu.synchronize()
    kw = dict(cu_seqlens=None, chunk_indices=None)
    assert_parity("prepare_wy_repr_bwd_full",
                  ct.npu_prepare_wy_repr_bwd_full(
                      k, v, beta, A, dA, dw, du, g, cs, **kw),
                  _thin.npu_prepare_wy_repr_bwd_full(
                      k, v, beta, A, dA, dw, du, g, cs, **kw))


def scenario_pwy():
    B, HK, HV, T, K, V, cs = 1, 4, 8, 256, 128, 128, 64
    dt = torch.bfloat16
    k = torch.rand(B, HK, T, K, dtype=dt, device="npu")
    v = torch.rand(B, HV, T, V, dtype=dt, device="npu")
    beta = torch.rand(B, HV, T, dtype=torch.float32, device="npu")
    A = torch.rand(B, HV, T, cs, dtype=dt, device="npu")
    dw = torch.rand(B, HV, T, K, dtype=dt, device="npu")
    du = torch.rand(B, HV, T, V, dtype=dt, device="npu")
    g = torch.rand(B, HV, T, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    kw = dict(chunk_size=cs, cu_seqlens=None, chunk_indices=None)
    assert_parity("prepare_wy_repr_bwd",
                  ct.npu_prepare_wy_repr_bwd(k, v, beta, A, dw, du, g, **kw),
                  _thin.npu_prepare_wy_repr_bwd(k, v, beta, A, dw, du, g, **kw))


def scenario_dv_local():
    B, Hqk, Hdo, T, K, V, cs = 1, 2, 4, 256, 128, 128, 64
    dt = torch.float16
    q = torch.randn(B, Hqk, T, K, dtype=dt, device="npu")
    k = torch.randn(B, Hqk, T, K, dtype=dt, device="npu")
    d_o = torch.randn(B, Hdo, T, V, dtype=dt, device="npu")
    g = torch.randn(B, Hdo, T, dtype=dt, device="npu")
    torch.npu.synchronize()
    kw = dict(scale=0.0625, chunk_size=cs, g_gamma=None, A=None,
              cu_seqlens=None, chunk_indices=None)
    assert_parity("chunk_bwd_dv_local",
                  ct.npu_chunk_bwd_dv_local(q, k, d_o, g, **kw),
                  _thin.npu_chunk_bwd_dv_local(q, k, d_o, g, **kw))


def scenario_pwy_da():
    B, H, T, K, V, cs = 1, 4, 256, 128, 128, 64
    dt = torch.float16
    k = torch.rand(B, H, T, K, dtype=dt, device="npu")
    v = torch.rand(B, H, T, V, dtype=dt, device="npu")
    beta = torch.rand(B, H, T, dtype=dt, device="npu")
    A = torch.rand(B, H, T, cs, dtype=dt, device="npu")
    dw = torch.rand(B, H, T, K, dtype=dt, device="npu")
    du = torch.rand(B, H, T, V, dtype=dt, device="npu")
    g = torch.rand(B, H, T, dtype=dt, device="npu")
    torch.npu.synchronize()
    kw = dict(chunk_size=cs, cu_seqlens=None, chunk_indices=None)
    assert_parity("prepare_wy_repr_bwd_da",
                  ct.npu_prepare_wy_repr_bwd_da(
                      k, v, beta, A, dw, du, g, **kw),
                  _thin.npu_prepare_wy_repr_bwd_da(
                      k, v, beta, A, dw, du, g, **kw))


def _fwd_h_inputs(B, Hk, Hv, T, K, V, dt=torch.bfloat16):
    k = torch.randn(B, Hk, T, K, dtype=dt, device="npu")
    w = torch.randn(B, Hv, T, K, dtype=dt, device="npu")
    u = torch.randn(B, Hv, T, V, dtype=dt, device="npu")
    g = -torch.rand(B, Hv, T, dtype=dt, device="npu") * 5 - 1e-3
    return k, w, u, g


def scenario_gated_fwd_h():
    B, Hk, Hv, T, K, V, cs = 1, 2, 2, 256, 128, 128, 64
    k, w, u, g = _fwd_h_inputs(B, Hk, Hv, T, K, V)
    torch.npu.synchronize()
    assert_parity(
        "chunk_gated_delta_rule_fwd_h(dense)",
        ct.npu_chunk_gated_delta_rule_fwd_h(k, w, u, g, chunk_size=cs),
        _thin.npu_chunk_gated_delta_rule_fwd_h(k, w, u, g, chunk_size=cs))
    is0 = torch.randn(B, Hv, K, V, dtype=torch.float32, device="npu")
    assert_parity(
        "chunk_gated_delta_rule_fwd_h(final)",
        ct.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g, initial_state=is0, output_final_state=True,
            chunk_size=cs),
        _thin.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g, initial_state=is0, output_final_state=True,
            chunk_size=cs))


def scenario_chunk_fwd_h():
    B, Hk, Hv, T, K, V, cs = 1, 2, 2, 256, 128, 128, 64
    k, w, u, g = _fwd_h_inputs(B, Hk, Hv, T, K, V)
    torch.npu.synchronize()
    assert_parity("chunk_fwd_h",
                  ct.npu_chunk_fwd_h(k, w, u, g=g, chunk_size=cs),
                  _thin.npu_chunk_fwd_h(k, w, u, g=g, chunk_size=cs))


def scenario_chunk_fwd_o():
    B, Hk, Hv, T, K, V, cs = 1, 2, 2, 256, 128, 128, 64
    dt = torch.bfloat16
    q = torch.randn(B, Hk, T, K, dtype=dt, device="npu")
    k = torch.randn(B, Hk, T, K, dtype=dt, device="npu")
    h = torch.randn(B, Hv, T // cs, K, V, dtype=dt, device="npu")
    v = torch.randn(B, Hv, T, V, dtype=dt, device="npu")
    g = torch.randn(B, Hv, T, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    scale = 0.08838834764831845
    assert_parity(
        "chunk_fwd_o(BNSD)",
        ct.npu_chunk_fwd_o(q, k, v, h, scale, g=g, chunk_size=cs,
                           output_layout="BNSD"),
        _thin.npu_chunk_fwd_o(q, k, v, h, scale, g=g, chunk_size=cs,
                              output_layout="BNSD"))


def scenario_bwd_dhu():
    H, T, K, V, cs = 4, 256, 128, 128, 64
    dt = torch.float16
    cu = [0, 128, 256]
    ci = [0, 0, 0, 1, 1, 0, 1, 1]
    q = torch.randn(1, H, T, K, dtype=dt, device="npu")
    k = torch.randn(1, H, T, K, dtype=dt, device="npu")
    w = torch.randn(1, H, T, K, dtype=dt, device="npu")
    do = torch.randn(1, H, T, V, dtype=dt, device="npu")
    dv = torch.randn(1, H, T, V, dtype=dt, device="npu")
    g = (-torch.sort(torch.rand(H * T, device="npu"), descending=True)[0]
         .reshape(1, H, T).to(dt))
    torch.npu.synchronize()
    kw = dict(scale=K ** -0.5, chunk_size=cs, g=g, gK=None, h0=None,
              dht=None, cu_seqlens=cu, chunk_indices=ci)
    assert_parity("chunk_gated_delta_rule_bwd_dhu",
                  ct.npu_chunk_gated_delta_rule_bwd_dhu(q, k, w, do, dv, **kw),
                  _thin.npu_chunk_gated_delta_rule_bwd_dhu(q, k, w, do, dv, **kw))


def scenario_conv1d_bwd_bnsd():
    batch, num_heads, seqlen, head_dim, width = 2, 2, 9, 16, 2
    dim = num_heads * head_dim
    dt = torch.bfloat16
    x = (torch.arange(batch * seqlen * dim).reshape(batch, seqlen, dim).float()
         + 11).to(dt).npu()
    weight = (torch.arange(width * dim).reshape(width, dim).float()
              + 111).to(dt).npu()
    dy = (torch.arange(batch * seqlen * dim).reshape(batch, seqlen, dim).float()
          + 211).to(dt).npu()
    st = (torch.arange(batch * width * dim).reshape(batch, width, dim).float()
          + 311).to(dt).npu()
    dht = (torch.arange(batch * width * dim).reshape(batch, width, dim).float()
           + 411).to(dt).npu()
    ylog = torch.zeros_like(x)
    for i in range(width):
        if i == 0:
            ylog += x * weight[width - 1 - i].view(1, 1, -1)
        else:
            ylog[:, i:, :] += x[:, :-i, :] * weight[width - 1 - i].view(1, 1, -1)
    yb = (ylog.reshape(batch, seqlen, num_heads, head_dim)
          .permute(0, 2, 1, 3).contiguous())
    dyb = (dy.reshape(batch, seqlen, num_heads, head_dim)
           .permute(0, 2, 1, 3).contiguous())
    torch.npu.synchronize()
    kw = dict(x=x, y=yb, weight=weight, dy=dyb, initial_state=st, dht=dht,
              activation=2, input_layout="BNSD")
    assert_parity("causal_conv1d_bwd(BNSD)",
                  ct.npu_causal_conv1d_bwd(**kw),
                  _thin.npu_causal_conv1d_bwd(**kw))


def scenario_chunk_kda_fwd():
    B, T, H, HV, K, V, cs = 1, 128, 4, 4, 128, 128, 64
    dt = torch.bfloat16
    q = torch.randn(B, T, H, K, dtype=dt, device="npu") * 5e-2
    k = torch.randn(B, T, H, K, dtype=dt, device="npu") * 5e-2
    v = torch.randn(B, T, HV, V, dtype=dt, device="npu") * 5e-2
    g = torch.randn(B, T, HV, K, dtype=torch.float32, device="npu")
    beta = torch.randn(B, T, HV, dtype=dt, device="npu")
    A_log = torch.randn(HV, dtype=torch.float32, device="npu") * 0.1
    dtb = torch.randn(HV * K, dtype=torch.float32, device="npu") * 0.5 - 3.0
    torch.npu.synchronize()
    kw = dict(layout="BSND", chunk_size=cs, scale=K ** -0.5, safe_gate=True,
              use_gate_in_kernel=True, A_log=A_log, dt_bias=dtb,
              disable_recompute=True)
    assert_parity("chunk_kda_fwd(dense BSND)",
                  ct.npu_chunk_kda_fwd(q, k, v, g, beta, **kw),
                  _thin.npu_chunk_kda_fwd(q, k, v, g, beta, **kw))


def scenario_dqkwg():
    B, HK, HV, T, K, V, cs = 1, 4, 4, 1024, 128, 128, 64
    NT = T // cs
    dt = torch.float16

    def make4(*shape, scale_):
        return (torch.randn(shape) * scale_).to(dt).permute(
            0, 2, 1, 3).contiguous().npu()

    def make5(*shape, scale_):
        return (torch.randn(shape) * scale_).to(dt).permute(
            0, 2, 1, 3, 4).contiguous().npu()

    q = make4(B, T, HK, K, scale_=5e-2)
    k = make4(B, T, HK, K, scale_=5e-2)
    v = make4(B, T, HV, V, scale_=5e-2)
    do = make4(B, T, HV, V, scale_=5e-2)
    dv = make4(B, T, HV, V, scale_=5e-1)
    h = make5(B, NT, HV, K, V, scale_=5e-2)
    dh = make5(B, NT, HV, K, V, scale_=5e-2)
    g = (-torch.sort(torch.rand(B * T * HV), descending=False)[0]
         .reshape(B, T, HV)).permute(0, 2, 1).to(dt).contiguous().npu()
    torch.npu.synchronize()
    kw = dict(cu_seqlens=None, chunk_indices=None, w=None, g_gamma=None,
              scale=0.088, use_exp2=None, transpose_state_layout=None)
    assert_parity("chunk_bwd_dqkwg",
                  ct.npu_chunk_bwd_dqkwg(q, k, v, g, h, do, dh, dv, cs, **kw),
                  _thin.npu_chunk_bwd_dqkwg(q, k, v, g, h, do, dh, dv, cs,
                                            **kw))


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260909)
    scenarios = [
        scenario_fast_gelu,
        scenario_recurrent_gated_delta_rule,
        scenario_recompute,
        scenario_pwy_full,
        scenario_pwy,
        scenario_dv_local,
        scenario_pwy_da,
        scenario_gated_fwd_h,
        scenario_chunk_fwd_h,
        scenario_chunk_fwd_o,
        scenario_bwd_dhu,
        scenario_conv1d_bwd_bnsd,
        scenario_chunk_kda_fwd,
        scenario_dqkwg,
    ]
    for fn in scenarios:
        fn()
    print("ALL PASS: 14 thin-op parity scenarios")


if __name__ == "__main__":
    main()
