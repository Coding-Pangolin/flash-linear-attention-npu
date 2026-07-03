#!/usr/bin/env python3
"""fwd_h NPU vs GPU dual benchmark using GPU-collected .pt dumps.

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
PTA_DIR = Path(__file__).resolve().parent / "pta"
sys.path.insert(0, str(GDN_DIR))
sys.path.insert(0, str(PTA_DIR))

from gpu_dump_loader import (  # noqa: E402
    find_op_dump_pt,
    list_case_dirs,
    load_case_meta,
    load_dump_for_npu,
    resolve_fwd_h_chunk_indices,
    resolve_seq_meta,
)
from gpu_dump_dual_utils import add_viz_cli_args, dual_then_viz, resolve_viz_dir
from gpu_dump_dual_runner import add_skip_cli_args, run_dual_batch  # noqa: E402
from test_fwd_h import forward_h_trans_cpu  # noqa: E402

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

OP_NAME = "fwd_h"


def _check_op_name(pt_path: Path, meta: dict[str, Any]) -> None:
    op = str(meta.get("op") or "")
    if op and op != OP_NAME:
        raise ValueError(f"{pt_path}: expected op={OP_NAME!r}, got {op!r}")


def _as_cu_tensor(cu_seqlens: list[int] | None) -> torch.Tensor | None:
    if cu_seqlens is None:
        return None
    return torch.tensor(cu_seqlens, dtype=torch.long)


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
    w = inputs["w"]
    u = inputs["u"]
    g = inputs["g"]
    initial_state = inputs.get("initial_state")
    gpu_h = gpu_outputs["h"]
    gpu_v_new = gpu_outputs["v_new"]
    gpu_final_state = gpu_outputs.get("final_state")

    B, Hk, T, K = k.shape
    Hv = u.shape[1]
    V = u.shape[-1]
    cu_seqlens, _chunk_pair, chunk_size, _scale = resolve_seq_meta(meta, case_meta)
    chunk_indices_arg = resolve_fwd_h_chunk_indices(
        cu_seqlens, chunk_size, meta, case_meta,
    )
    output_final_state = bool(
        meta.get("output_final_state", case_meta.get("output_final_state", False))
    )
    if gpu_final_state is not None:
        output_final_state = True

    case_name = label or f"{pt_path.parent.name}/{pt_path.name}"
    nt = (
        len(chunk_indices_arg) // 2
        if chunk_indices_arg is not None
        else (T + chunk_size - 1) // chunk_size
    )
    if verbose:
        print(
            f"\n=== {case_name} ===\n"
            f"  pt: {pt_path} B={B} Hk={Hk} Hv={Hv} T={T} K={K} V={V} cs={chunk_size} "
            f"varlen={cu_seqlens is not None} NT={nt} output_final_state={output_final_state}",
            flush=True,
        )

    t0 = time.time()
    h_npu, v_new_npu, final_state_npu = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k.npu(),
        w.npu(),
        u.npu(),
        g=g.npu(),
        initial_state=initial_state.npu() if initial_state is not None else None,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_arg,
    )
    torch.npu.synchronize()
    npu_elapsed = time.time() - t0

    cu_tensor = _as_cu_tensor(cu_seqlens)
    chunk_indices_tensor = (
        torch.tensor(chunk_indices_arg, dtype=torch.long)
        if chunk_indices_arg is not None
        else None
    )
    h_fp64, v_new_fp64, final_state_fp64 = forward_h_trans_cpu(
        k.double(),
        w.double(),
        u.double(),
        g=g.double(),
        initial_state=initial_state.double() if initial_state is not None else None,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_tensor,
        chunk_indices=chunk_indices_tensor,
        accum_dtype=torch.float64,
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

    dual_then_viz("h", h_npu, h_fp64, gpu_h, viz_dir=tensor_viz_dir,
                  sample_count=sample_count, enable_viz=enable_viz)
    dual_then_viz("v_new", v_new_npu, v_new_fp64, gpu_v_new, viz_dir=tensor_viz_dir,
                  sample_count=sample_count, enable_viz=enable_viz)
    if output_final_state and gpu_final_state is not None:
        dual_then_viz(
            "final_state",
            final_state_npu,
            final_state_fp64,
            gpu_final_state,
            viz_dir=tensor_viz_dir,
            sample_count=sample_count,
            enable_viz=enable_viz,
        )

    return {
        "case": case_name,
        "status": "pass",
        "pt": str(pt_path),
        "npu_elapsed_s": round(npu_elapsed, 4),
        "shapes": {"B": B, "Hk": Hk, "Hv": Hv, "T": T, "K": K, "V": V, "chunk_size": chunk_size},
    }


def run_one_case(
    case_dir: Path,
    *,
    verbose: bool = True,
    enable_viz: bool = True,
    sample_count: int = 200_000,
    viz_dir: Path | None = None,
) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    pt_path, _ = find_op_dump_pt(case_dir, OP_NAME, phase=None)
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
    p = argparse.ArgumentParser(description="fwd_h NPU vs GPU dual benchmark from GPU dumps")
    p.add_argument("--dump-root", type=Path, default=None)
    p.add_argument("--pt", type=Path, default=None)
    p.add_argument("--pts", default="")
    p.add_argument("--case", default="")
    p.add_argument("--cases", default="")
    p.add_argument("--phase", default="all", help="all | prefix:phase_1_ | prefix:gva_")
    p.add_argument("--report", type=Path, default=None)
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
    raise ValueError(f"unknown --phase {args.phase!r}")


def main() -> int:
    args = _parse_args()
    return run_dual_batch(
        args,
        op_name=OP_NAME,
        report_basename="fwd_h_gpu_dump_dual_report.json",
        viz_tensor_names=("h", "v_new"),
        collect_pt_paths=_collect_pt_paths,
        select_cases=_select_cases,
        run_one_pt=run_one_pt,
        run_one_case=run_one_case,
    )


if __name__ == "__main__":
    raise SystemExit(main())
