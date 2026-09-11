"""Ascend950-only stable parity: the A5 kernels the A2 host cannot load.

``regression_stable_full.py`` runs these itself when the device reports
Ascend950, and prints an explicit SKIP naming them otherwise.  This driver
exists so the A5 kernels can be checked on their own as well -- for instance on
a host whose installed OPP carries the A5 kernels but rejects some A2-era
inputs, where the full A2 suite is not the right first check.

The mechanics are identical: every call to the thin backend is rerouted to the
Stable-ABI launcher through the same shim, and the ctypes implementation stays
the reference.

Usage (241, wheel/package importable)::

    PYTHONPATH=<pkg> FLA_NPU_STABLE_LIB=/path/libfla_npu_thin.so \
        python tests/regression_stable_a5.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regression_stable_full import StableShim, check_baseline  # noqa: E402


def main() -> int:
    import regression_950_ops as a5
    import regression_thin_ops as suite_module

    shim = StableShim()
    a5._thin = shim
    suite_module._thin = shim
    torch.npu.set_device(0)
    torch.manual_seed(20260909)
    device = str(torch.npu.get_device_name(0))
    print(f"device: {device}")
    scenarios = [
        a5.scenario_fwd_prepare,
        a5.scenario_bwd_finalize,
        a5.scenario_recurrent_kda,
        a5.scenario_chunk_gated_delta_rule_fwd_a5,
    ]
    # Everything below has an Ascend950 kernel.  The list comes from nm on the
    # OPP (GetWorkspaceSize symbols), which carries 11 of the 26 operators:
    # ChunkFwdH, ChunkFwdO, ChunkGatedDeltaRule{BwdFinalize,Fwd,FwdH,FwdPrepare},
    # ChunkLocalCumsum, ChunkScaledDotKkt, RecomputeWUFwd, RecurrentKda,
    # SolveTri.  The rest cannot run here at all -- no aclnnCausalConv1d, no
    # aclnnRecurrentGatedDeltaRule, no KDA backward, no fast_gelu -- which is
    # why this driver is the A5 slice of the suite rather than the whole thing.
    scenarios += [
        # regression_thin_ops' KDA case covers BSND and TND; the 950-only one
        # above is BSND, so this adds the varlen spelling to the A5 record too.
        suite_module.scenario_recurrent_kda,
        suite_module.scenario_chunk_fwd_h,
        suite_module.scenario_chunk_fwd_o,
        suite_module.scenario_gated_fwd_h,
        suite_module.scenario_chunk_local_cumsum,
        suite_module.scenario_scaled_dot_kkt,
        suite_module.scenario_solve_tri_dense,
        suite_module.scenario_solve_tri_guards,
    ]
    # npu_recompute_w_u_fwd is deliberately absent: its A5 kernel never returns
    # for these inputs (measured with ctypes alone and a 180s timeout, so it is
    # the kernel, not the launcher).  A hang cannot be turned into an error from
    # the host, so it is excluded and recorded here instead.
    for scenario in scenarios:
        print(f"--- entering {scenario.__name__}", flush=True)
        scenario()
    suite_module.SCENARIOS.update(a5.SCENARIOS)
    print(f"\nstable ops exercised: {len(shim.calls)}")
    for name in sorted(shim.calls):
        print(f"  {name}: {shim.calls[name]} call(s)")
    if shim.missing:
        print(f"MISSING ADAPTERS: {sorted(set(shim.missing))}")
        return 1
    status = check_baseline(device, suite_module)
    if status != 0:
        return status
    print("ALL PASS: Ascend950 stable parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
