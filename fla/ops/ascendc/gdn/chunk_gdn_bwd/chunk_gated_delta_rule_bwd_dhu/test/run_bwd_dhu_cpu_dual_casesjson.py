#!/usr/bin/env python3
"""CPU dual benchmark for bwd_dhu — aligned with test_npu_bwd_dhu_gva.py."""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch_npu
import fla_npu  # noqa: F401

REPO = Path(__file__).resolve().parents[7]
GDN_DIR = REPO / "fla/ops/ascendc/gdn"
TORCH_CUSTOM_TEST = REPO / "torch_custom/fla_npu/test"

for p in (REPO / "gpu" / "scripts", GDN_DIR, TORCH_CUSTOM_TEST):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gdn_cpu_dual_casesjson import (  # noqa: E402
    BWD_DHU_SMOKE_CASES,
    DEFAULT_CASES_JSON,
    default_out_dir,
    dual_then_viz_cpu,
    generate_cu_seqlens_for_case,
    resolve_cases,
    write_batch_report,
    write_case_report,
)
from gdn_case_utils import parse_dtype  # noqa: E402

_BWD_DHU_GOLDEN = TORCH_CUSTOM_TEST / "test_bwd_dhu.py"
_spec = importlib.util.spec_from_file_location("test_bwd_dhu_golden", _BWD_DHU_GOLDEN)
_golden_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_golden_mod)

chunk_gated_delta_rule_bwd_dhu_cpu = _golden_mod.chunk_gated_delta_rule_bwd_dhu_cpu
create_bwd_dhu_random_inputs = _golden_mod.create_bwd_dhu_random_inputs
effective_scale = _golden_mod.effective_scale
generate_cu_seqlens = _golden_mod.generate_cu_seqlens
prepare_chunk_indices = _golden_mod.prepare_chunk_indices
scale_for_compute_dtype = _golden_mod.scale_for_compute_dtype


def build_bwd_dhu_inputs(case: dict[str, Any], *, seed: int) -> dict[str, Any]:
    B = int(case["B"])
    Hk = int(case["query_head"])
    Hv = int(case["value_head"])
    T = int(case["T"])
    K = int(case["Kdim"])
    V = int(case["Vdim"])
    chunk_size = int(case.get("chunk_size", 64))
    ktype = parse_dtype(case.get("dtype", "bf16"))
    gtype = parse_dtype(case.get("gtype", "fp32"))
    varlen = bool(case.get("varlen", False))

    torch.manual_seed(seed)
    q, k, w, do, dv, g = create_bwd_dhu_random_inputs(B, Hk, Hv, T, K, V, ktype, gtype)

    cu_seqlens = None
    chunk_indices = None
    if varlen:
        cu_seqlens_len = int(case.get("mean_len", 2))
        if cu_seqlens_len <= 2:
            cu_seqlens = [0, T]
        elif T <= 4096 and cu_seqlens_len <= 8:
            cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
        else:
            cu_seqlens = generate_cu_seqlens_for_case(case).tolist()
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    scale = scale_for_compute_dtype(effective_scale(1.0 / math.sqrt(K), K), ktype)

    return {
        "q": q, "k": k, "w": w, "do": do, "dv": dv, "g": g,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "scale": scale,
        "chunk_size": chunk_size,
    }


def run_case(
    case: dict[str, Any],
    *,
    device_id: int,
    seed: int,
    dual_level: str,
    enable_viz: bool,
    viz_dir: Path | None,
    sample_count: int,
) -> dict[str, Any]:
    name = case["name"]
    t0 = time.time()
    record: dict[str, Any] = {"case": name, "status": "error", "checks": {}}

    try:
        torch.npu.set_device(device_id)
        inputs = build_bwd_dhu_inputs(case, seed=seed)
        common = dict(
            q=inputs["q"], k=inputs["k"], w=inputs["w"], do=inputs["do"], dv=inputs["dv"],
            g=inputs["g"],
            cu_seqlens=inputs["cu_seqlens"],
            chunk_indices=inputs["chunk_indices"],
            scale=inputs["scale"],
            chunk_size=inputs["chunk_size"],
        )

        dh_fp64, _, dv2_fp64 = chunk_gated_delta_rule_bwd_dhu_cpu(**common, golden_mode="fp64")
        dh_npu_bench, _, dv2_npu_bench = chunk_gated_delta_rule_bwd_dhu_cpu(**common, golden_mode="npu")

        dh_npu, _, dv2_npu = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
            inputs["q"].npu(), inputs["k"].npu(), inputs["w"].npu(),
            inputs["do"].npu(), inputs["dv"].npu(),
            scale=inputs["scale"],
            chunk_size=inputs["chunk_size"],
            g=inputs["g"].npu(),
            gK=None,
            h0=None,
            dht=None,
            cu_seqlens=inputs["cu_seqlens"],
            chunk_indices=inputs["chunk_indices"],
        )

        ok_dh, res_dh = dual_then_viz_cpu(
            "dh", dh_npu, dh_fp64, dh_npu_bench,
            case_name=name, viz_dir=viz_dir, sample_count=sample_count,
            enable_viz=enable_viz, level=dual_level,
        )
        ok_dv2, res_dv2 = dual_then_viz_cpu(
            "dv2", dv2_npu, dv2_fp64, dv2_npu_bench,
            case_name=name, viz_dir=viz_dir, sample_count=sample_count,
            enable_viz=enable_viz, level=dual_level,
        )

        record["checks"] = {"dh": res_dh, "dv2": res_dv2}
        record["status"] = "pass" if ok_dh and ok_dv2 else "fail"
        record["elapsed_s"] = round(time.time() - t0, 2)
    except Exception as exc:
        record["error"] = str(exc)
        record["elapsed_s"] = round(time.time() - t0, 2)
        import traceback
        traceback.print_exc()

    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="bwd_dhu CPU dual benchmark (cases.json / smoke)")
    parser.add_argument("--cases-json", type=Path, default=DEFAULT_CASES_JSON)
    parser.add_argument("--case", action="append", dest="cases", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("TEST_DEVICE_ID", "0")))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dual-level", default="L1")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--sample-count", type=int, default=8)
    args = parser.parse_args()

    cases = resolve_cases(
        cases_json=args.cases_json,
        case_names=args.cases,
        smoke=args.smoke,
        smoke_cases=BWD_DHU_SMOKE_CASES,
    )
    out_dir = args.out_dir or default_out_dir("bwd_dhu", smoke=args.smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz" if args.viz else None

    print(f"[bwd_dhu cpu_dual] device={args.device_id} cases={len(cases)} out={out_dir}", flush=True)
    results: list[dict[str, Any]] = []
    for case in cases:
        print(f"\n=== case: {case['name']} ===", flush=True)
        record = run_case(
            case,
            device_id=args.device_id,
            seed=args.seed,
            dual_level=args.dual_level,
            enable_viz=args.viz,
            viz_dir=viz_dir,
            sample_count=args.sample_count,
        )
        write_case_report(out_dir, record)
        results.append(record)
        print(f"[{case['name']}] {record['status']} ({record.get('elapsed_s', '?')}s)", flush=True)

    report_path = write_batch_report(
        out_dir,
        op="bwd_dhu",
        results=results,
        device_id=args.device_id,
        cases_json=args.cases_json,
        dual_level=args.dual_level,
        smoke=args.smoke,
    )
    summary = {r["case"]: r["status"] for r in results}
    print(f"\nSummary: {summary}", flush=True)
    print(f"Report: {report_path}", flush=True)
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
