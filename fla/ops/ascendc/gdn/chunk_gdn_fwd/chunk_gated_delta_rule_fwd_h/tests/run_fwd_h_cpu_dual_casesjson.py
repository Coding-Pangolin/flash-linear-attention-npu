#!/usr/bin/env python3
"""fwd_h CPU dual benchmark from gpu/cases.json (GPU-unsupported cases).

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
from datetime import datetime
from pathlib import Path
from typing import Any

import ct
import fla_npu
import torch
import torch_npu

REPO = Path(__file__).resolve().parents[7]
GDN_DIR = Path(__file__).resolve().parents[3]
PTA_DIR = Path(__file__).resolve().parent / "pta"
FLA_NPU_TEST = REPO / "torch_custom" / "fla_npu" / "test"
sys.path.insert(0, str(GDN_DIR))
sys.path.insert(0, str(FLA_NPU_TEST))

from gdn_cpu_dual_casesjson import (  # noqa: E402
    add_cases_cli_args,
    default_out_dir,
    generate_cu_seqlens_for_case,
    resolve_cases_from_args,
    write_batch_report,
    write_case_report,
)
from gdn_case_utils import (  # noqa: E402
    _LOW_PRECISION_INPUT_HALF_RANGE_QK,
    _LOW_PRECISION_INPUT_HALF_RANGE_V,
    _create_gate_g,
    _rand_uniform,
    parse_dtype,
)
from gpu_dump_loader import prepare_pairwise_chunk_indices_list  # noqa: E402
from test_fwd_h import forward_h_trans_cpu  # noqa: E402

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)


def chunk_local_cumsum_cpu(g_bth: torch.Tensor, cu_list: list[int] | None, chunk_size: int) -> torch.Tensor:
    g = g_bth.float().clone()
    if cu_list is None:
        bos, eos = 0, g.shape[1]
        for chunk_start in range(0, eos - bos, chunk_size):
            s = bos + chunk_start
            e = min(s + chunk_size, eos)
            g[:, s:e, :] = g[:, s:e, :].cumsum(dim=0)
        return g
    for i in range(len(cu_list) - 1):
        bos, eos = cu_list[i], cu_list[i + 1]
        seg_len = eos - bos
        for chunk_start in range(0, seg_len, chunk_size):
            s = bos + chunk_start
            e = min(s + chunk_size, eos)
            g[:, s:e, :] = g[:, s:e, :].cumsum(dim=0)
    return g


def build_fwd_h_inputs(case: dict[str, Any], device: torch.device, seed: int = 0):
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

    k = _rand_uniform((B, T, Hk, K), ktype, hr, device).transpose(1, 2).contiguous()
    w = _rand_uniform((B, T, Hv, K), ktype, hr, device).transpose(1, 2).contiguous()
    u = _rand_uniform((B, T, Hv, V), ktype, hr_v, device).transpose(1, 2).contiguous()

    g_bh_t = _create_gate_g(B, Hv, T, gtype, device, narrow=low, gate_function=gate_function)
    g_bth = g_bh_t.transpose(1, 2).contiguous()

    cu_seqlens_t = generate_cu_seqlens_for_case(case)
    cu_list = None if cu_seqlens_t is None else cu_seqlens_t.detach().cpu().tolist()
    chunk_indices = None if cu_list is None else prepare_pairwise_chunk_indices_list(cu_list, chunk_size)

    g_cum = chunk_local_cumsum_cpu(g_bth.cpu(), cu_list, chunk_size).to(device)
    g = g_cum.transpose(1, 2).contiguous()

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
        "num_seqs": 0 if cu_list is None else len(cu_list) - 1,
        "num_chunks": 0 if chunk_indices is None else len(chunk_indices) // 2,
        "seed": seed,
    }
    return k, w, u, g, cu_list, chunk_indices, meta


def _dual_check(name: str, npu_out: torch.Tensor, ref_fp64: torch.Tensor, ref_npu: torch.Tensor, level: str):
    print(f"================== {name} (dual: fp64 gt / npu-aligned bench) ==================", flush=True)
    result = ct.dual(npu_out.cpu(), ref_fp64.cpu(), ref_npu.cpu(), level=level)
    ok = bool(result.get("success"))
    print(f"[{name}] dual {'PASS' if ok else 'FAIL'}: checks={result.get('checks')} ratios={result.get('ratios')}", flush=True)
    return ok, result


def run_one_case(
    case: dict[str, Any],
    *,
    device_id: int,
    seed: int,
    dual_level: str,
    enable_viz: bool,
    viz_sample_count: int,
    out_dir: Path,
    npu_only: bool = False,
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
        "mode": "npu_only" if npu_only else "cpu_dual",
    }
    try:
        torch.npu.set_device(device_id)
        device = torch.device(f"npu:{device_id}")
        k, w, u, g, cu_list, chunk_indices, meta = build_fwd_h_inputs(case, device, seed=seed)
        record["meta"] = meta
        print(f"\n=== {case_name} ===", flush=True)
        print(json.dumps(meta, indent=2), flush=True)

        k_cpu, w_cpu, u_cpu, g_cpu = k.cpu(), w.cpu(), u.cpu(), g.cpu()
        ref_h_fp64 = ref_v_fp64 = ref_h_npu = ref_v_npu = None
        if not npu_only:
            cu_tensor = None if cu_list is None else torch.tensor(cu_list, dtype=torch.long)
            chunk_indices_tensor = (
                torch.tensor(chunk_indices, dtype=torch.long) if chunk_indices is not None else None
            )
            golden_kwargs = dict(
                initial_state=None,
                chunk_size=meta["chunk_size"],
                cu_seqlens=cu_tensor,
                chunk_indices=chunk_indices_tensor,
            )
            print("[CPU] fp64 golden ...", flush=True)
            t0 = time.time()
            ref_h_fp64, ref_v_fp64, _ = forward_h_trans_cpu(
                k_cpu, w_cpu, u_cpu, g_cpu, **golden_kwargs, golden_mode="fp64",
            )
            ref_h_npu, ref_v_npu, _ = forward_h_trans_cpu(
                k_cpu, w_cpu, u_cpu, g_cpu, **golden_kwargs, golden_mode="npu",
            )
            record["cpu_elapsed_s"] = round(time.time() - t0, 3)
        else:
            record["cpu_elapsed_s"] = 0.0

        print("[NPU] npu_chunk_gated_delta_rule_fwd_h ...", flush=True)
        t0 = time.time()
        h_npu, v_npu, _ = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
            k,
            w,
            u,
            g=g.float(),
            gk=None,
            initial_state=None,
            output_final_state=False,
            chunk_size=meta["chunk_size"],
            save_new_value=True,
            cu_seqlens=cu_list,
            chunk_indices=chunk_indices,
            use_exp2=False,
            transpose_state_layout=False,
        )
        torch.npu.synchronize()
        record["npu_elapsed_s"] = round(time.time() - t0, 3)
        print(
            f"[NPU OK] h={tuple(h_npu.shape)} v_new={tuple(v_npu.shape)}",
            flush=True,
        )

        case_out = out_dir / case_name
        if save_outputs or (enable_viz and not npu_only):
            case_out.mkdir(parents=True, exist_ok=True)

        if npu_only:
            if save_outputs:
                torch.save(
                    {"meta": meta, "h_npu": h_npu.cpu(), "v_new_npu": v_npu.cpu()},
                    case_out / "outputs.pt",
                )
            record["status"] = "pass"
        else:
            case_out.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "meta": meta,
                    "h_npu": h_npu.cpu(),
                    "h_ref_fp64": ref_h_fp64.cpu(),
                    "h_ref_npu": ref_h_npu.cpu(),
                    "v_new_npu": v_npu.cpu(),
                    "v_new_ref_fp64": ref_v_fp64.cpu(),
                    "v_new_ref_npu": ref_v_npu.cpu(),
                },
                case_out / "outputs.pt",
            )

            viz_dir = case_out / "viz"
            for out_name, npu_t, hp_t, sp_t, spatial in (
                ("h", h_npu, ref_h_fp64, ref_h_npu, False),
                ("v_new", v_npu, ref_v_fp64, ref_v_npu, True),
            ):
                ok, dual_res = _dual_check(out_name, npu_t, hp_t, sp_t, dual_level)
                record["dual"][out_name] = {
                    "pass": ok,
                    "checks": dual_res.get("checks"),
                    "ratios": dual_res.get("ratios"),
                }
                if enable_viz:
                    os.makedirs(viz_dir, exist_ok=True)
                    ct.viz(
                        npu_t.cpu().float(),
                        hp_t.cpu().float(),
                        bench=sp_t.cpu().float(),
                        out_dir=str(viz_dir),
                        name=f"{case_name}_{out_name}_npu_vs_fp64",
                        diff_thd=0.001,
                        spatial=spatial,
                        sample_count=viz_sample_count,
                    )

            all_ok = all(v.get("pass") for v in record["dual"].values())
            record["status"] = "pass" if all_ok else "fail"
    except Exception as exc:
        record["status"] = "error"
        record["error"] = traceback.format_exc()
        print(f"[{case_name}] ERROR:\n{record['error']}", flush=True)
    record["elapsed_s"] = round(time.time() - t_start, 3)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="fwd_h cases.json benchmark")
    add_cases_cli_args(parser)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dual-level", default="L1")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--no-save-outputs", action="store_true")
    parser.add_argument("-sc", "--sample-count", type=int, default=200_000)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", "0")))
    args = parser.parse_args()

    if args.npu_only and not args.no_viz:
        args.no_viz = True

    cases = resolve_cases_from_args(args)
    out_dir = args.out_dir or default_out_dir(
        "fwd_h", smoke=args.smoke, npu_only=args.npu_only, all_cases=args.all_cases,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[CONFIG] device={args.device} smoke={args.smoke} all_cases={args.all_cases} "
        f"npu_only={args.npu_only} cases={len(cases)}",
        flush=True,
    )
    print(f"[CONFIG] case_names={[c['name'] for c in cases]}", flush=True)
    print(f"[CONFIG] out_dir={out_dir}", flush=True)
    if not args.npu_only:
        print(f"[CONFIG] dual=ct.dual(npu, fp64, npu_aligned) level={args.dual_level}", flush=True)

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
            npu_only=args.npu_only,
            save_outputs=not args.no_save_outputs,
        )
        results.append(rec)
        write_case_report(out_dir, rec)

    report_path = write_batch_report(
        out_dir,
        op="fwd_h",
        results=results,
        device_id=args.device,
        cases_json=args.cases_json,
        dual_level=args.dual_level,
        smoke=args.smoke,
        npu_only=args.npu_only,
        all_cases=args.all_cases,
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
