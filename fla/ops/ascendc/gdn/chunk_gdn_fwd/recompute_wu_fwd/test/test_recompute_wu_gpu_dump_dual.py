#!/usr/bin/env python3
"""recompute_wu NPU vs GPU dual benchmark using GPU-collected .pt dumps.

Compare: ct.dual(npu_out, cpu_fp64_golden, gpu_out_from_dump)
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

import ct
import fla_npu
import torch
import torch_npu

from gpu_dump_loader import (
    find_op_dump_pt,
    list_case_dirs,
    load_case_meta,
    load_dump_for_npu,
)

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))
from test import get_bos_eos  # noqa: E402

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

OP_NAME = "recompute_wu"


def compute_w_golden_fp64(
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
    acc = torch.float64
    w = torch.zeros(B, H, T, D, dtype=torch.float64)
    for i_b in range(B):
        for idx in range(NT):
            bos, eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            for i_h in range(H):
                hk = i_h // hv_per_hk
                a_chunk = A[i_b, i_h, bos:eos, : eos - bos].to(acc)
                k_chunk = k[i_b, hk, bos:eos, :].to(acc)
                beta_chunk = beta[i_b, i_h, bos:eos].to(acc)
                g_chunk = g[i_b, i_h, bos:eos].to(acc)
                g_exp = torch.exp(g_chunk)
                kbg = k_chunk * (beta_chunk * g_exp).unsqueeze(-1)
                w[i_b, i_h, bos:eos, :] = torch.matmul(a_chunk, kbg)
    return w


def compute_u_golden_fp64(
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
    acc = torch.float64
    u = torch.zeros(B, Hv, T, v.shape[-1], dtype=torch.float64)
    for i_b in range(B):
        for idx in range(NT):
            bos, eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            for i_h in range(Hv):
                a_chunk = A[i_b, i_h, bos:eos, : eos - bos].to(acc)
                v_chunk = v[i_b, i_h, bos:eos, :].to(acc)
                beta_chunk = beta[i_b, i_h, bos:eos].to(acc)
                vb = v_chunk * beta_chunk.unsqueeze(-1)
                u[i_b, i_h, bos:eos, :] = torch.matmul(a_chunk, vb)
    return u


def _resolve_seq_meta(meta: dict[str, Any], case_meta: dict[str, Any]) -> tuple[list[int] | None, list[int] | None, int]:
    cu = meta.get("cu_seqlens")
    if cu is None:
        cu = case_meta.get("cu_seqlens")
    if cu is not None and len(cu) == 0:
        cu = None

    chunk_indices = meta.get("chunk_indices_npu")
    if chunk_indices is None:
        chunk_indices = meta.get("chunk_indices")

    chunk_size = int(meta.get("chunk_size") or case_meta.get("chunk_size") or 64)
    return cu, chunk_indices, chunk_size


def run_one_case(
    case_dir: Path,
    *,
    dump_phase: str | None = "bwd",
    verbose: bool = True,
) -> dict[str, Any]:
    case_name = case_dir.name
    case_meta = load_case_meta(case_dir)
    pt_path, _raw = find_op_dump_pt(case_dir, OP_NAME, phase=dump_phase)
    inputs, meta, gpu_outputs = load_dump_for_npu(pt_path)

    k = inputs["k"]
    v = inputs["v"]
    beta = inputs["beta"]
    A = inputs["A"]
    g = inputs["g"]
    gpu_w = gpu_outputs["w"]
    gpu_u = gpu_outputs["u"]

    B, Hk, T, K = k.shape
    Hv = v.shape[1]
    V = v.shape[-1]
    cu_seqlens, chunk_indices, chunk_size = _resolve_seq_meta(meta, case_meta)
    if chunk_indices is not None:
        NT = len(chunk_indices) // 2
    else:
        NT = (T + chunk_size - 1) // chunk_size

    if verbose:
        print(
            f"\n=== {case_name} ===\n"
            f"  pt: {pt_path.name} phase={meta.get('phase')} "
            f"B={B} Hk={Hk} Hv={Hv} T={T} K={K} V={V} cs={chunk_size} "
            f"varlen={cu_seqlens is not None} NT={NT}",
            flush=True,
        )

    t0 = time.time()
    w_npu, u_npu = torch.ops.npu.npu_recompute_w_u_fwd(
        k.npu(),
        v.npu(),
        beta.npu(),
        A.npu(),
        chunk_size,
        g=g.npu(),
        gk=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    torch.npu.synchronize()
    npu_elapsed = time.time() - t0

    cpu_w_fp64 = compute_w_golden_fp64(
        k.double(), v.double(), beta.double(), A.double(), g.double(),
        cu_seqlens, chunk_indices, B, Hv, T, K, chunk_size, NT, Hk=Hk,
    )
    cpu_u_fp64 = compute_u_golden_fp64(
        v.double(), beta.double(), A.double(),
        cu_seqlens, chunk_indices, B, Hv, T, chunk_size, NT,
    )

    print("  [w] ct.dual(npu, cpu_fp64, gpu)", flush=True)
    ct.dual(w_npu.cpu(), cpu_w_fp64, gpu_w)
    print("  [u] ct.dual(npu, cpu_fp64, gpu)", flush=True)
    ct.dual(u_npu.cpu(), cpu_u_fp64, gpu_u)

    return {
        "case": case_name,
        "status": "pass",
        "pt": str(pt_path),
        "phase": meta.get("phase"),
        "npu_elapsed_s": round(npu_elapsed, 4),
        "shapes": {"B": B, "Hk": Hk, "Hv": Hv, "T": T, "K": K, "V": V, "chunk_size": chunk_size},
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="recompute_wu NPU vs GPU dual benchmark from GPU dumps")
    p.add_argument("--dump-root", type=Path, required=True, help="GPU dump root (e.g. ./GPU_DUMP)")
    p.add_argument("--case", default="", help="single case directory name under dump-root")
    p.add_argument(
        "--cases",
        default="",
        help="comma-separated case names (overrides --phase)",
    )
    p.add_argument(
        "--phase",
        default="all",
        help="filter case dirs: all | prefix:phase_1_ | prefix:gva_",
    )
    p.add_argument(
        "--dump-phase",
        default="bwd",
        choices=("fwd", "bwd", "any"),
        help="which recompute_wu dump to use when fwd+bwd both exist (default: bwd)",
    )
    p.add_argument("--report", type=Path, default=None, help="write JSON report path")
    return p.parse_args()


def _select_cases(dump_root: Path, args: argparse.Namespace) -> list[Path]:
    all_dirs = list_case_dirs(dump_root)
    if args.case:
        d = dump_root / args.case
        if not d.is_dir():
            raise FileNotFoundError(f"case dir not found: {d}")
        return [d]
    if args.cases.strip():
        names = [n.strip() for n in args.cases.split(",") if n.strip()]
        by_name = {p.name: p for p in all_dirs}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"unknown case(s): {', '.join(missing)}")
        return [by_name[n] for n in names]

    phase = args.phase.strip().lower()
    if phase in ("", "all"):
        return all_dirs
    if phase.startswith("prefix:"):
        prefix = phase.split(":", 1)[1]
        return [d for d in all_dirs if d.name.startswith(prefix)]
    raise ValueError(f"unknown --phase {args.phase!r}; use all or prefix:<name_prefix>")


def main() -> int:
    args = _parse_args()
    dump_phase: str | None = None if args.dump_phase == "any" else args.dump_phase
    selected = _select_cases(args.dump_root, args)
    if not selected:
        print("No cases selected.", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    failed = 0
    for case_dir in selected:
        try:
            results.append(run_one_case(case_dir, dump_phase=dump_phase, verbose=True))
        except Exception as e:
            failed += 1
            print(f"\n=== {case_dir.name} FAILED ===\n{e}", flush=True)
            traceback.print_exc()
            results.append({"case": case_dir.name, "status": "fail", "error": str(e)})

    report = {
        "op": OP_NAME,
        "dump_root": str(args.dump_root),
        "dump_phase": args.dump_phase,
        "total": len(results),
        "passed": len(results) - failed,
        "failed": failed,
        "results": results,
    }
    report_path = args.report or (args.dump_root / "recompute_wu_gpu_dump_dual_report.json")
    with Path(report_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"\nDone: {report['passed']}/{report['total']} passed, report -> {report_path}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
