#!/usr/bin/env python3
"""recompute_wu CPU dual benchmark from gpu/cases.json.

Compare: ct.dual(npu_out, cpu_fp64_golden, cpu_npu_aligned_bench)
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
sys.path.insert(0, str(GDN_DIR))
sys.path.insert(0, str(TEST_DIR))

from gdn_case_utils import (  # noqa: E402
    _LOW_PRECISION_INPUT_HALF_RANGE_QK,
    _LOW_PRECISION_INPUT_HALF_RANGE_V,
    _create_gate_g,
    _rand_uniform,
    parse_dtype,
)
from gdn_cpu_dual_casesjson import (  # noqa: E402
    DEFAULT_CASES_JSON,
    dual_then_viz_cpu,
    generate_cu_seqlens_for_case,
    matmul_npu_aligned,
    prepare_chunk_indices_list,
    resolve_cases,
    default_out_dir,
    write_batch_report,
    write_case_report,
)
from test import get_bos_eos  # noqa: E402
from test_recompute_wu_gpu_dump_dual import (  # noqa: E402
    compute_u_golden_fp64,
    compute_w_golden_fp64,
)

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

OP_NAME = "recompute_wu"


def compute_w_golden_npu_aligned(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens,
    chunk_indices,
    B: int,
    H: int,
    T: int,
    D: int,
    chunk_size: int,
    NT: int,
    Hk: int = 0,
) -> torch.Tensor:
    if Hk == 0:
        Hk = H
    hv_per_hk = H // Hk
    elem_dtype = k.dtype
    w = torch.zeros(B, H, T, D, dtype=torch.float32)
    for i_b in range(B):
        for idx in range(NT):
            bos, eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            for i_h in range(H):
                hk = i_h // hv_per_hk
                a_chunk = A[i_b, i_h, bos:eos, : eos - bos]
                k_chunk = k[i_b, hk, bos:eos, :]
                beta_chunk = beta[i_b, i_h, bos:eos].float()
                g_chunk = g[i_b, i_h, bos:eos].float()
                kbg = k_chunk.float() * (beta_chunk * torch.exp(g_chunk)).unsqueeze(-1)
                w[i_b, i_h, bos:eos, :] = matmul_npu_aligned(a_chunk, kbg, elem_dtype)
    return w


def compute_u_golden_npu_aligned(
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens,
    chunk_indices,
    B: int,
    Hv: int,
    T: int,
    chunk_size: int,
    NT: int,
) -> torch.Tensor:
    elem_dtype = v.dtype
    u = torch.zeros(B, Hv, T, v.shape[-1], dtype=torch.float32)
    for i_b in range(B):
        for idx in range(NT):
            bos, eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            for i_h in range(Hv):
                a_chunk = A[i_b, i_h, bos:eos, : eos - bos]
                v_chunk = v[i_b, i_h, bos:eos, :]
                beta_chunk = beta[i_b, i_h, bos:eos].float()
                vb = v_chunk.float() * beta_chunk.unsqueeze(-1)
                u[i_b, i_h, bos:eos, :] = matmul_npu_aligned(a_chunk, vb, elem_dtype)
    return u


def build_recompute_wu_inputs(case: dict[str, Any], *, seed: int = 0) -> dict[str, Any]:
    B = int(case["B"])
    T = int(case["T"])
    Hk = int(case["query_head"])
    Hv = int(case["value_head"])
    K = int(case["Kdim"])
    V = int(case["Vdim"])
    chunk_size = int(case.get("chunk_size", 64))
    ktype = parse_dtype(case["dtype"])
    gtype = parse_dtype(case["gtype"])
    gate_function = str(case.get("gate_function", "negative_linear")).strip().lower()

    torch.manual_seed(seed)
    random.seed(seed)

    low = ktype in (torch.float16, torch.bfloat16)
    hr = _LOW_PRECISION_INPUT_HALF_RANGE_QK if low else 2e-2
    hr_v = _LOW_PRECISION_INPUT_HALF_RANGE_V if low else 2e-2

    k = torch.randn(B, Hk, T, K, dtype=ktype)
    v = torch.randn(B, Hv, T, V, dtype=ktype)
    beta = torch.randn(B, Hv, T, dtype=ktype)
    A = torch.randn(B, Hv, T, chunk_size, dtype=ktype)
    if gate_function == "randn":
        g = torch.randn(B, Hv, T, dtype=ktype)
    else:
        g = _create_gate_g(
            B, Hv, T, gtype, torch.device("cpu"), narrow=low, gate_function=gate_function,
        )

    cu_seqlens_t = generate_cu_seqlens_for_case(case)
    cu_list = None if cu_seqlens_t is None else cu_seqlens_t.tolist()
    chunk_indices = None if cu_list is None else prepare_chunk_indices_list(cu_list, chunk_size)
    NT = (T + chunk_size - 1) // chunk_size if chunk_indices is None else len(chunk_indices) // 2

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
        "NT": NT,
        "seed": seed,
    }
    return {
        "k": k,
        "v": v,
        "beta": beta,
        "A": A,
        "g": g,
        "cu_seqlens": cu_list,
        "chunk_indices": chunk_indices,
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
    save_outputs: bool = True,
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
        payload = build_recompute_wu_inputs(case, seed=seed)
        meta = payload["meta"]
        record["meta"] = meta
        print(f"\n=== {case_name} ===", flush=True)
        print(json.dumps(meta, indent=2), flush=True)

        k = payload["k"]
        v = payload["v"]
        beta = payload["beta"]
        A = payload["A"]
        g = payload["g"]
        cu_list = payload["cu_seqlens"]
        chunk_indices = payload["chunk_indices"]
        B, Hk, T, K = meta["B"], meta["Hk"], meta["T"], meta["K"]
        Hv, V, chunk_size, NT = meta["Hv"], meta["V"], meta["chunk_size"], meta["NT"]

        print("[CPU] fp64 + npu-aligned golden ...", flush=True)
        t0 = time.time()
        w_fp64 = compute_w_golden_fp64(
            k.double(), v.double(), beta.double(), A.double(), g.double(),
            cu_list, chunk_indices, B, Hv, T, K, chunk_size, NT, Hk=Hk,
        )
        u_fp64 = compute_u_golden_fp64(
            v.double(), beta.double(), A.double(),
            cu_list, chunk_indices, B, Hv, T, chunk_size, NT,
        )
        w_npu_bench = compute_w_golden_npu_aligned(
            k, v, beta, A, g, cu_list, chunk_indices, B, Hv, T, K, chunk_size, NT, Hk=Hk,
        )
        u_npu_bench = compute_u_golden_npu_aligned(
            v, beta, A, cu_list, chunk_indices, B, Hv, T, chunk_size, NT,
        )
        record["cpu_elapsed_s"] = round(time.time() - t0, 3)

        print("[NPU] npu_recompute_w_u_fwd ...", flush=True)
        t0 = time.time()
        w_npu, u_npu = torch.ops.npu.npu_recompute_w_u_fwd(
            k.npu(),
            v.npu(),
            beta.npu(),
            A.npu(),
            chunk_size,
            g=g.npu(),
            gk=None,
            cu_seqlens=cu_list,
            chunk_indices=chunk_indices,
        )
        torch.npu.synchronize()
        record["npu_elapsed_s"] = round(time.time() - t0, 3)
        print(f"[NPU OK] w={tuple(w_npu.shape)} u={tuple(u_npu.shape)}", flush=True)

        case_out = out_dir / case_name
        if save_outputs or enable_viz:
            case_out.mkdir(parents=True, exist_ok=True)
        if save_outputs:
            torch.save(
                {
                    "meta": meta,
                    "w_npu": w_npu.cpu(),
                    "w_ref_fp64": w_fp64.cpu(),
                    "w_ref_npu": w_npu_bench.cpu(),
                    "u_npu": u_npu.cpu(),
                    "u_ref_fp64": u_fp64.cpu(),
                    "u_ref_npu": u_npu_bench.cpu(),
                },
                case_out / "outputs.pt",
            )
        viz_dir = case_out / "viz" if enable_viz else None
        dual_results = {}
        for out_name, npu_t, hp_t, sp_t in (
            ("w", w_npu, w_fp64, w_npu_bench),
            ("u", u_npu, u_fp64, u_npu_bench),
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
    parser = argparse.ArgumentParser(description="recompute_wu CPU dual from cases.json")
    parser.add_argument("--cases-json", type=Path, default=DEFAULT_CASES_JSON)
    parser.add_argument("--cases", default="", help="comma-separated case names")
    parser.add_argument("--smoke", action="store_true", help="run built-in small smoke cases")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dual-level", default="L1")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument(
        "--no-save-outputs",
        action="store_true",
        help="skip saving outputs.pt (only json reports + dual logs)",
    )
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
            save_outputs=not args.no_save_outputs,
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
