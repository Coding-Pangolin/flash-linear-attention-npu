"""全量算子 stable parity：复用 regression_thin_ops 的每个场景，但把
`_thin`（pybind）整体改道到 stable 后端，逐个与 ctypes 参考对比。

用法（221/241，wheel 已装入环境）：
    FLA_NPU_STABLE_LIB=/path/libfla_npu_thin.so PYTHONPATH=<env> \
        python tests/regression_stable_full.py

任何算子/场景在 stable 侧缺失或回退，都会在这里暴露为 AttributeError/差异。
"""
from __future__ import annotations

import os
import sys

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fla_npu.ops.ascendc import _stable  # noqa: E402

STABLE_LIB = os.environ.get("FLA_NPU_STABLE_LIB", "")


class StableShim:
    """Stands in for the `_thin` module inside the regression suite."""

    def __init__(self):
        self.calls: dict[str, int] = {}
        self.missing: list[str] = []

    def __getattr__(self, name):
        import fla_npu.ops.ascendc._stable as stable_mod

        try:
            target = getattr(stable_mod, name)
        except AttributeError:
            from fla_npu.ops.ascendc import _stable_generated as generated

            target = getattr(generated, name, None)
            if target is None:
                self.missing.append(name)
                raise AttributeError(
                    f"stable backend has no adapter for {name!r} "
                    f"(coverage gap)")

        def counted(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            return target(*args, **kwargs)

        return counted


def main() -> int:
    if not STABLE_LIB:
        raise SystemExit("set FLA_NPU_STABLE_LIB to the built libfla_npu_thin.so")
    import regression_thin_ops as suite

    shim = StableShim()
    suite._thin = shim  # every scenario now exercises the stable backend
    torch.npu.set_device(0)
    torch.manual_seed(20260909)
    scenarios = [
        suite.scenario_fast_gelu,
        suite.scenario_recurrent_gated_delta_rule,
        suite.scenario_recurrent_kda,
        suite.scenario_recompute,
        suite.scenario_pwy_full,
        suite.scenario_pwy,
        suite.scenario_dv_local,
        suite.scenario_pwy_da,
        suite.scenario_gated_fwd_h,
        suite.scenario_chunk_fwd_h,
        suite.scenario_chunk_fwd_o,
        suite.scenario_bwd_dhu,
        suite.scenario_conv1d_prefill,
        suite.scenario_conv1d_varlen_initial_state,
        suite.scenario_conv1d_update,
        suite.scenario_conv1d_bwd_bnsd,
        suite.scenario_chunk_kda_fwd,
        suite.scenario_chunk_kda_fwd_variants,
        suite.scenario_chunk_kda_bwd_intra,
        suite.scenario_chunk_kda_bwd,
        suite.scenario_dqkwg,
        suite.scenario_chunk_local_cumsum,
        suite.scenario_scaled_dot_kkt,
        suite.scenario_solve_tri_dense,
        suite.scenario_kda_gate_cumsum,
        suite.scenario_chunk_gated_delta_rule_fwd,
    ]
    for scenario in scenarios:
        print(f"--- entering {scenario.__name__}", flush=True)
        scenario()

    # Two operators only exist in the Ascend950 OPP
    # (chunk_gated_delta_rule_fwd_prepare / _bwd_finalize).  They are exercised
    # by their own driver instead of being silently absent: on a non-950 host
    # this prints an explicit SKIP naming them, so the coverage record says why
    # rather than counting them as covered.
    device = str(torch.npu.get_device_name(0))
    if "950" in device:
        import regression_950_ops as a5

        a5._thin = shim
        for scenario in (a5.scenario_fwd_prepare, a5.scenario_bwd_finalize,
                         a5.scenario_chunk_gated_delta_rule_fwd_a5):
            print(f"--- entering {scenario.__name__} (Ascend950)",
                  flush=True)
            scenario()
    else:
        print(f"SKIP chunk_gated_delta_rule_fwd_prepare / _bwd_finalize / "
              f"fwd(a5): requires Ascend950, this host reports {device!r}")
    print(f"\nstable ops exercised: {len(shim.calls)}")
    for name in sorted(shim.calls):
        print(f"  {name}: {shim.calls[name]} call(s)")
    if shim.missing:
        print(f"MISSING ADAPTERS: {sorted(set(shim.missing))}")
        return 1
    print("ALL PASS: full stable parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
