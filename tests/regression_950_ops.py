"""Ascend950-only 算子安装态数值回归（ctypes vs thin parity）。

用法（950 主机，wheel 已 pip install --target envXXX）:
    PYTHONPATH=/path/envXXX python regression_950_ops.py

覆盖：chunk_gated_delta_rule_fwd_prepare / bwd_finalize。
"""
from __future__ import annotations

import time

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.ascendc import _aclnn_ctypes as ct  # noqa: E402
from fla_npu.ops.ascendc import _thin  # noqa: E402


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


def scenario_fwd_prepare():
    dt = torch.bfloat16
    B, HK, HV, T, K, V, cs = 1, 4, 8, 1792, 128, 128, 64
    q = torch.randn(B, HK, T, K, dtype=dt, device="npu")
    k = torch.randn(B, HK, T, K, dtype=dt, device="npu")
    v = torch.randn(B, HV, T, V, dtype=dt, device="npu")
    g = torch.randn(B, HV, T, dtype=torch.float32, device="npu")
    beta = torch.randn(B, HV, T, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    kw = dict(chunk_size=cs, use_qk_l2norm_in_kernel=True, use_exp2=True,
              use_beta_sigmoid_in_kernel=True, output_a=True)
    assert_parity("fwd_prepare(case0)",
                  ct.npu_chunk_gated_delta_rule_fwd_prepare(q, k, v, g, beta,
                                                            **kw),
                  _thin.npu_chunk_gated_delta_rule_fwd_prepare(q, k, v, g,
                                                               beta, **kw))


def scenario_bwd_finalize():
    dt = torch.bfloat16
    B, HK, HV, T, K, V, cs = 1, 4, 8, 256, 128, 128, 64
    NT = T // cs
    q = torch.randn(B, HK, T, K, dtype=dt, device="npu")
    k = torch.randn(B, HK, T, K, dtype=dt, device="npu")
    v = torch.randn(B, HV, T, V, dtype=dt, device="npu")
    v_new = torch.randn(B, HV, T, V, dtype=dt, device="npu")
    do = torch.randn(B, HV, T, V, dtype=dt, device="npu")
    du = torch.randn(B, HV, T, V, dtype=dt, device="npu")
    g = torch.randn(B, HV, T, dtype=torch.float32, device="npu")
    beta = torch.randn(B, HV, T, dtype=torch.float32, device="npu")
    h = torch.randn(B, HV, NT, K, V, dtype=dt, device="npu")
    dh = torch.randn(B, HV, NT, K, V, dtype=dt, device="npu")
    a = torch.randn(B, HV, T, cs, dtype=dt, device="npu")
    q_rstd = torch.randn(B, HK, T, dtype=torch.float32, device="npu")
    k_rstd = torch.randn(B, HK, T, dtype=torch.float32, device="npu")
    beta_raw = torch.randn(B, HV, T, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    kw = dict(q_rstd=q_rstd, k_rstd=k_rstd, beta_raw=beta_raw,
              chunk_size=cs, use_qk_l2_norm_in_kernel=True,
              use_beta_sigmoid_in_kernel=True)
    assert_parity("bwd_finalize(dense)",
                  ct.npu_chunk_gated_delta_rule_bwd_finalize(
                      q, k, v, v_new, do, du, g, beta, h, dh, a, **kw),
                  _thin.npu_chunk_gated_delta_rule_bwd_finalize(
                      q, k, v, v_new, do, du, g, beta, h, dh, a, **kw))


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260909)
    device = str(torch.npu.get_device_name(0))
    assert "950" in device, f"requires Ascend950, got {device}"
    scenario_fwd_prepare()
    scenario_bwd_finalize()
    print("ALL PASS: 2 Ascend950-only parity scenarios")


if __name__ == "__main__":
    main()
