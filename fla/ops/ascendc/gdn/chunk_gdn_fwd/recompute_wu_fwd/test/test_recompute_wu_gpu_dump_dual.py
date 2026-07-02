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

import fla_npu
import torch
import torch_npu

GDN_DIR = Path(__file__).resolve().parents[3]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GDN_DIR))
sys.path.insert(0, str(TEST_DIR))

from gpu_dump_loader import (
    find_op_dump_pt,
    list_case_dirs,
    load_case_meta,
    load_dump_for_npu,
    resolve_seq_meta,
)
from gpu_dump_dual_utils import add_viz_cli_args, dual_then_viz, resolve_viz_dir
from gpu_dump_dual_runner import add_skip_cli_args, run_dual_batch
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


def _check_op_name(pt_path: Path, meta: dict[str, Any]) -> None:
    op = str(meta.get("op") or "")
    if op and op != OP_NAME:
        raise ValueError(f"{pt_path}: expected op={OP_NAME!r}, got {op!r}")


def run_one_pt(
    pt_path: Path,
    *,
    case_meta: dict[str, Any] | None = None,
    label: str | None = None,
    verbose: bool = True,
    enable_viz: bool = True,
    sample_count: int = 200_000,
    viz_dir: Path | None = None,
) -> dict[str, Any]:
    pt_path = pt_path.resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"dump .pt not found: {pt_path}")

    if case_meta is None:
        case_meta = load_case_meta(pt_path.parent)

    inputs, meta, gpu_outputs = load_dump_for_npu(pt_path)
    _check_op_name(pt_path, meta)

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
    cu_seqlens, chunk_indices, chunk_size, _ = resolve_seq_meta(meta, case_meta)
    if chunk_indices is not None:
        NT = len(chunk_indices) // 2
    else:
        NT = (T + chunk_size - 1) // chunk_size

    case_name = label or f"{pt_path.parent.name}/{pt_path.name}"
    if verbose:
        print(
            f"\n=== {case_name} ===\n"
            f"  pt: {pt_path} phase={meta.get('phase')} "
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

    tensor_viz_dir = None
    if enable_viz:
        base_viz_dir = resolve_viz_dir(
            viz_dir=viz_dir,
            pt_path=pt_path,
            case_dir=None,
            default_report_dir=pt_path.parent,
        )
        tensor_viz_dir = base_viz_dir / case_name.replace("/", "_")

    dual_then_viz("w", w_npu, cpu_w_fp64, gpu_w, viz_dir=tensor_viz_dir,
                  sample_count=sample_count, enable_viz=enable_viz)
    dual_then_viz("u", u_npu, cpu_u_fp64, gpu_u, viz_dir=tensor_viz_dir,
                  sample_count=sample_count, enable_viz=enable_viz)

    return {
        "case": case_name,
        "status": "pass",
        "pt": str(pt_path),
        "phase": meta.get("phase"),
        "npu_elapsed_s": round(npu_elapsed, 4),
        "shapes": {"B": B, "Hk": Hk, "Hv": Hv, "T": T, "K": K, "V": V, "chunk_size": chunk_size},
    }


def run_one_case(
    case_dir: Path,
    *,
    dump_phase: str | None = "bwd",
    verbose: bool = True,
    enable_viz: bool = True,
    sample_count: int = 200_000,
    viz_dir: Path | None = None,
) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    pt_path, _raw = find_op_dump_pt(case_dir, OP_NAME, phase=dump_phase)
    return run_one_pt(
        pt_path,
        case_meta=load_case_meta(case_dir),
        label=case_dir.name,
        verbose=verbose,
        enable_viz=enable_viz,
        sample_count=sample_count,
        viz_dir=viz_dir or (case_dir / "viz"),
    )


def _collect_pt_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.pt is not None:
        paths.append(args.pt)
    if args.pts.strip():
        paths.extend(Path(p.strip()) for p in args.pts.split(",") if p.strip())
    return paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="recompute_wu NPU vs GPU dual benchmark from GPU dumps")
    p.add_argument(
        "--dump-root",
        type=Path,
        default=None,
        help="GPU dump root for batch mode (e.g. ./GPU_DUMP); not needed with --pt/--pts",
    )
    p.add_argument(
        "--pt",
        type=Path,
        default=None,
        help="single dump .pt file (absolute or relative path)",
    )
    p.add_argument(
        "--pts",
        default="",
        help="comma-separated dump .pt files",
    )
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
    add_skip_cli_args(p)
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
    pt_paths = _collect_pt_paths(args)
    dump_phase: str | None = None if args.dump_phase == "any" else args.dump_phase
    report_extra: dict[str, Any] = {}
    if pt_paths:
        report_extra["pt_files"] = [str(p) for p in pt_paths]
    else:
        report_extra["dump_phase"] = args.dump_phase
    return run_dual_batch(
        args,
        op_name=OP_NAME,
        report_basename="recompute_wu_gpu_dump_dual_report.json",
        viz_tensor_names=("w", "u"),
        collect_pt_paths=_collect_pt_paths,
        select_cases=_select_cases,
        run_one_pt=run_one_pt,
        run_one_case=run_one_case,
        run_case_kwargs={"dump_phase": dump_phase},
        report_extra=report_extra,
    )


if __name__ == "__main__":
    raise SystemExit(main())
