#!/usr/bin/env python3
"""bwd_dhu CPU dual benchmark from gpu/cases.json.

Compare: ct.dual(npu_out, cpu_fp64_golden, cpu_npu_aligned_bench)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import fla_npu
import torch
import torch_npu

GDN_DIR = Path(__file__).resolve().parents[3]
TEST_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO / "gpu" / "scripts"))
sys.path.insert(0, str(GDN_DIR))
sys.path.insert(0, str(TEST_DIR))

from gdn_case_utils import parse_dtype  # noqa: E402
from gdn_cpu_dual_casesjson import (  # noqa: E402
    DEFAULT_CASES_JSON,
    dual_then_viz_cpu,
    generate_cu_seqlens_for_case,
    prepare_chunk_indices_list,
    resolve_cases,
    default_out_dir,
    write_batch_report,
    write_case_report,
)
from test_chunk_gated_delta_rule_bwd_dhu import (  # noqa: E402
    chunk_gated_delta_rule_bwd_dhu_torch,
    create_bwd_dhu_random_inputs,
    effective_scale,
    scale_for_compute_dtype,
)

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

OP_NAME = "bwd_dhu"


def build_bwd_dhu_inputs(case: dict[str, Any], *, seed: int = 0) -> dict[str, Any]:
    B = int(case["B"])
    T = int(case["T"])
    Hk = int(case["query_head"])
    Hv = int(case["value_head"])
    K = int(case["Kdim"])
    V = int(case["Vdim"])
    chunk_size = int(case.get("chunk_size", 64))
    ktype = parse_dtype(case["dtype"])
    gtype = parse_dtype(case["gtype"])

    torch.manual_seed(seed)
    q, k, w, d_o, dv, g = create_bwd_dhu_random_inputs(B, Hk, Hv, T, K, V, ktype, gtype)

    cu_seqlens_t = generate_cu_seqlens_for_case(case)
    cu_list = None if cu_seqlens_t is None else cu_seqlens_t.tolist()
    chunk_indices = None if cu_list is None else prepare_chunk_indices_list(cu_list, chunk_size)
    scale = scale_for_compute_dtype(effective_scale(float(case.get("scale", K ** -0.5)), K), ktype)

    meta = {
        "case_name": case["name"],
        "B": B,
        "T": T,
        "Hk": Hk,
        "Hv": Hv,
        "K": K,
        "V": V,
        "chunk_size": chunk_size,
        "dtype": case["dtype"],
        "varlen": bool(case.get("varlen", False)),
        "cu_seqlens": cu_list,
        "chunk_indices": chunk_indices,
        "scale": scale,
        "seed": seed,
    }
    return {
        "q": q,
        "k": k,
        "w": w,
        "do": d_o,
        "dv": dv,
        "g": g,
        "cu_seqlens": cu_list,
        "chunk_indices": chunk_indices,
        "scale": scale,
        "meta": meta,
    }


def run_one_case(
    case: dict[str, Any],
    *,
    device_id: int,
    seed: int,
    dual_level: str,
    enable_viz: bool,
    viz_sample_count: int,
    out_dir: Path,
) -> dict[str, Any]:
    case_name = str(case["name"])
    t_start = time.time()
    record: dict[str, Any] = {
        "case": case_name,
        "status": "fail",
        "dual": {},
        "meta": {},
        "error": None,
    }
    try:
        torch.npu.set_device(device_id)
        payload = build_bwd_dhu_inputs(case, seed=seed)
        meta = payload["meta"]
        record["meta"] = meta
        print(f"\n=== {case_name} ===", flush=True)
        print(json.dumps(meta, indent=2), flush=True)

        q = payload["q"]
        k = payload["k"]
        w = payload["w"]
        d_o = payload["do"]
        dv = payload["dv"]
        g = payload["g"]
        cu_list = payload["cu_seqlens"]
        chunk_indices = payload["chunk_indices"]
        scale = payload["scale"]
        chunk_size = meta["chunk_size"]

        golden_kwargs = dict(
            cu_seqlens=cu_list,
            chunk_indices=chunk_indices,
            g=g,
            scale=scale,
            chunk_size=chunk_size,
        )
        print("[CPU] fp64 + npu-aligned golden ...", flush=True)
        t0 = time.time()
        dh_fp64, _, dv2_fp64 = chunk_gated_delta_rule_bwd_dhu_torch(
            q.double(), k.double(), w.double(), d_o.double(), dv.double(),
            **golden_kwargs,
            accum_dtype=torch.float64,
        )
        dh_npu_bench, _, dv2_npu_bench = chunk_gated_delta_rule_bwd_dhu_torch(
            q, k, w, d_o, dv,
            **golden_kwargs,
            accum_dtype=torch.float32,
            matmul_elem_dtype=q.dtype,
        )
        record["cpu_elapsed_s"] = round(time.time() - t0, 3)

        print("[NPU] npu_chunk_gated_delta_rule_bwd_dhu ...", flush=True)
        t0 = time.time()
        dh_npu, dh0_npu, dv2_npu = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
            q.npu(),
            k.npu(),
            w.npu(),
            d_o.npu(),
            dv.npu(),
            scale=scale,
            chunk_size=chunk_size,
            g=g.npu(),
            gK=None,
            h0=None,
            dht=None,
            cu_seqlens=cu_list,
            chunk_indices=chunk_indices,
        )
        torch.npu.synchronize()
        record["npu_elapsed_s"] = round(time.time() - t0, 3)
        print(f"[NPU OK] dh={tuple(dh_npu.shape)} dv2={tuple(dv2_npu.shape)}", flush=True)

        case_out = out_dir / case_name
        case_out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "meta": meta,
                "dh_npu": dh_npu.cpu(),
                "dh_ref_fp64": dh_fp64.cpu(),
                "dh_ref_npu": dh_npu_bench.cpu(),
                "dv2_npu": dv2_npu.cpu(),
                "dv2_ref_fp64": dv2_fp64.cpu(),
                "dv2_ref_npu": dv2_npu_bench.cpu(),
            },
            case_out / "outputs.pt",
        )
        viz_dir = case_out / "viz" if enable_viz else None
        dual_results = {}
        for out_name, npu_t, hp_t, sp_t, spatial in (
            ("dh", dh_npu, dh_fp64, dh_npu_bench, False),
            ("dv2", dv2_npu, dv2_fp64, dv2_npu_bench, True),
        ):
            ok, dual_res = dual_then_viz_cpu(
                out_name,
                npu_t,
                hp_t,
                sp_t,
                case_name=case_name,
                viz_dir=viz_dir,
                sample_count=viz_sample_count,
                enable_viz=enable_viz,
                level=dual_level,
                spatial=spatial,
            )
            dual_results[out_name] = {
                "pass": ok,
                "checks": dual_res.get("checks"),
                "ratios": dual_res.get("ratios"),
            }
        record["dual"] = dual_results
        record["status"] = "pass" if all(v["pass"] for v in dual_results.values()) else "fail"
    except Exception:
        record["status"] = "error"
        record["error"] = traceback.format_exc()
        print(f"[{case_name}] ERROR:\n{record['error']}", flush=True)
    record["elapsed_s"] = round(time.time() - t_start, 3)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="bwd_dhu CPU dual from cases.json")
    parser.add_argument("--cases-json", type=Path, default=DEFAULT_CASES_JSON)
    parser.add_argument("--cases", default="", help="comma-separated case names")
    parser.add_argument("--smoke", action="store_true", help="run built-in small smoke cases")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dual-level", default="L1")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("-sc", "--sample-count", type=int, default=200_000)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", "0")))
    args = parser.parse_args()

    case_names = [x.strip() for x in args.cases.split(",") if x.strip()] or None
    cases = resolve_cases(cases_json=args.cases_json, case_names=case_names, smoke=args.smoke)
    out_dir = args.out_dir or default_out_dir(OP_NAME, smoke=args.smoke)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] op={OP_NAME} device={args.device} smoke={args.smoke}", flush=True)
    print(f"[CONFIG] cases={[c['name'] for c in cases]}", flush=True)
    print(f"[CONFIG] out_dir={out_dir}", flush=True)

    results = []
    for case in cases:
        rec = run_one_case(
            case,
            device_id=args.device,
            seed=args.seed,
            dual_level=args.dual_level,
            enable_viz=not args.no_viz,
            viz_sample_count=args.sample_count,
            out_dir=out_dir,
        )
        results.append(rec)
        write_case_report(out_dir, rec)

    report_path = write_batch_report(
        out_dir,
        op=OP_NAME,
        results=results,
        device_id=args.device,
        cases_json=args.cases_json,
        dual_level=args.dual_level,
        smoke=args.smoke,
    )
    print("\n========== SUMMARY ==========", flush=True)
    for r in results:
        print(f"  {r['case']}: {r['status']}", flush=True)
    print(f"report -> {report_path}", flush=True)
    summary = {"fail": 0, "error": 0}
    for r in results:
        if r["status"] in summary:
            summary[r["status"]] += 1
    return 0 if summary["fail"] == 0 and summary["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
