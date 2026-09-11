"""Per-operator host timings: ctypes vs the Stable-ABI backend.

Every scenario in ``regression_thin_ops`` already builds correct inputs for its
operator and calls it through both backends; this driver reuses exactly those
calls (by wrapping the two namespaces) instead of duplicating input
construction, so the numbers are measured on the inputs that are known to be
legal and bit-identical.

What is measured is **host enqueue time**: the wall time of the Python call
that hands the work to aclnn, with no ``synchronize`` inside the timed region.
That is the part this migration changed.  The comparison that ``assert_parity``
does afterwards forces a synchronisation between iterations, so both backends
face the same (idle-pipeline) condition -- which is what makes the ratio
comparable even though the absolute numbers are lower than in a back-to-back
decode loop.

``--baseline`` chooses what the Stable-ABI path is compared against:

* ``ctypes`` (default) -- the shipped-before reference implementation, i.e.
  "how much did the host path improve over what we used to ship";
* ``pybind`` -- the other thin launcher (`_C_thin`, FLA_NPU_THIN_ABI=pybind),
  i.e. "how does the ABI-free backend compare with the ABI-pinned one".  That
  run needs a wheel built with FLA_NPU_BUILD_THIN=1.

Usage (device host, package importable)::

    PYTHONPATH=<env> python tests/bench_stable_host.py [--rounds 5] [--json out.json]
    PYTHONPATH=<pybind env> FLA_NPU_STABLE_LIB=<so> \
        python tests/bench_stable_host.py --baseline pybind
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regression_stable_full import StableShim  # noqa: E402

import regression_thin_ops as suite  # noqa: E402
from fla_npu.ops.ascendc import _aclnn_ctypes as ct  # noqa: E402

# name -> {"ctypes": [...], "stable": [...]}
SAMPLES: dict[str, dict[str, list]] = {}


def _record(backend: str, op_name: str, seconds: float) -> None:
    SAMPLES.setdefault(op_name, {}).setdefault(backend, []).append(seconds * 1e3)


def _wrap(namespace, backend: str, names: list[str]) -> None:
    for name in names:
        original = getattr(namespace, name, None)
        if original is None or not callable(original):
            continue

        def timed(*args, __original=original, __name=name, **kwargs):
            start = time.perf_counter()
            result = __original(*args, **kwargs)
            _record(backend, __name, time.perf_counter() - start)
            return result

        setattr(namespace, name, timed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", default="")
    parser.add_argument("--baseline", choices=("ctypes", "pybind"),
                        default="ctypes")
    args = parser.parse_args()

    if not suite._thin.__class__.__name__ == "StableShim":
        pass  # StableShim is installed below either way
    shim = StableShim()
    suite._thin = shim
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
        suite.scenario_conv1d_update,
        suite.scenario_conv1d_gather_padding,
        suite.scenario_conv1d_bwd_bnsd,
        suite.scenario_chunk_kda_fwd,
        suite.scenario_chunk_kda_bwd_intra,
        suite.scenario_chunk_kda_bwd,
        suite.scenario_dqkwg,
        suite.scenario_chunk_local_cumsum,
        suite.scenario_scaled_dot_kkt,
        suite.scenario_solve_tri_dense,
        suite.scenario_kda_gate_cumsum,
        suite.scenario_chunk_gated_delta_rule_fwd,
    ]

    # Warm up once with timing off so the first (compile/alloc-heavy) pass does
    # not land in the samples.
    for scenario in scenarios:
        scenario()
    torch.npu.synchronize()

    if args.baseline == "pybind":
        # The scenarios call `ct.<op>` for the reference and `_thin.<op>` for the
        # backend under test; pointing `ct` at the pybind wrapper turns the same
        # comparison into pybind-vs-stable on identical inputs.
        from fla_npu.ops.ascendc import _thin as pybind

        pybind._extension()  # fail loudly if this environment has no _C_thin
        suite.ct = pybind
        suite.BASELINE_GAP_TOLERANT = True
        baseline_label = "pybind"
    else:
        baseline_label = "ctypes"
    baseline_ns = suite.ct
    ops = [name for name in dir(baseline_ns)
           if name.startswith("npu_") and callable(getattr(baseline_ns, name))]
    _wrap(baseline_ns, baseline_label, ops)
    _wrap(shim, "stable", ops)

    for index in range(args.rounds):
        for scenario in scenarios:
            scenario()
        torch.npu.synchronize()
        print(f"round {index + 1}/{args.rounds} done", flush=True)

    def p50(values):
        values = sorted(values)
        return values[len(values) // 2] if values else float("nan")

    rows = []
    for op_name, per_backend in sorted(SAMPLES.items()):
        ctypes_ms = p50(per_backend.get(baseline_label, []))
        stable_ms = p50(per_backend.get("stable", []))
        if not per_backend.get(baseline_label) or not per_backend.get("stable"):
            continue
        rows.append({
            "op": op_name,
            "ctypes_ms": round(ctypes_ms, 4),
            "stable_ms": round(stable_ms, 4),
            "stable_over_ctypes": round(stable_ms / ctypes_ms, 3),
            "samples": len(per_backend[baseline_label]),
        })

    rows.sort(key=lambda row: row["stable_over_ctypes"], reverse=True)
    print(f"\n{'operator':<44}{baseline_label:>10}{'stable':>10}{'ratio':>8}")
    for row in rows:
        print(f"{row['op']:<44}{row['ctypes_ms']:>10.4f}"
              f"{row['stable_ms']:>10.4f}{row['stable_over_ctypes']:>8.2f}")
    if not rows:
        print("no samples collected")
        return 1
    worst = rows[0]
    print(f"\nworst ratio: {worst['op']} {worst['stable_over_ctypes']}x "
          f"(ctypes {worst['ctypes_ms']} ms, stable {worst['stable_ms']} ms)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"rounds": args.rounds, "rows": rows}, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
