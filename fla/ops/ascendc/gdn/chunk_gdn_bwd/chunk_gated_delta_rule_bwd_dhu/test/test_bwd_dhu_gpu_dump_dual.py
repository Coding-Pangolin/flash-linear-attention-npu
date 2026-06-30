#!/usr/bin/env python3
"""bwd_dhu NPU vs GPU dual benchmark using GPU-collected .pt dumps.

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

from gpu_dump_loader import (  # noqa: E402
    find_op_dump_pt,
    list_case_dirs,
    load_case_meta,
    load_dump_for_npu,
    resolve_seq_meta,
)
from gpu_dump_dual_utils import add_viz_cli_args, dual_then_viz, resolve_viz_dir  # noqa: E402
from test_chunk_gated_delta_rule_bwd_dhu import (  # noqa: E402
    chunk_gated_delta_rule_bwd_dhu_torch,
    effective_scale,
    scale_for_compute_dtype,
)

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

OP_NAME = "bwd_dhu"


def _check_op_name(pt_path: Path, meta: dict[str, Any]) -> None:
    op = str(meta.get("op") or "")
    if op and op != OP_NAME:
        raise ValueError(f"{pt_path}: expected op={OP_NAME!r}, got {op!r}")


def _resolve_scale(meta: dict[str, Any], case_meta: dict[str, Any], K: int, k_dtype: torch.dtype) -> float:
    scale = meta.get("scale")
    if scale is None:
        scale = case_meta.get("scale")
    if scale is None:
        scale = K ** -0.5
    return scale_for_compute_dtype(effective_scale(float(scale), K), k_dtype)


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

    q = inputs["q"]
    k = inputs["k"]
    w = inputs["w"]
    d_o = inputs["do"]
    dv = inputs["dv"]
    g = inputs.get("g")
    h0 = inputs.get("h0")
    dht = inputs.get("dht")
    gpu_dh = gpu_outputs["dh"]
    gpu_dv2 = gpu_outputs["dv2"]
    gpu_dh0 = gpu_outputs.get("dh0")

    B, Hk, T, K = q.shape
    Hv = w.shape[1]
    V = d_o.shape[-1]
    cu_seqlens, chunk_indices, chunk_size, _ = resolve_seq_meta(meta, case_meta)
    scale = _resolve_scale(meta, case_meta, K, k.dtype)

    case_name = label or f"{pt_path.parent.name}/{pt_path.name}"
    if verbose:
        NT = len(chunk_indices) // 2 if chunk_indices else (T + chunk_size - 1) // chunk_size
        print(
            f"\n=== {case_name} ===\n"
            f"  pt: {pt_path} B={B} Hk={Hk} Hv={Hv} T={T} K={K} V={V} cs={chunk_size} "
            f"scale={scale:.6g} varlen={cu_seqlens is not None} NT={NT} "
            f"h0={h0 is not None} dht={dht is not None}",
            flush=True,
        )

    t0 = time.time()
    dh_npu, dh0_npu, dv2_npu = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q.npu(),
        k.npu(),
        w.npu(),
        d_o.npu(),
        dv.npu(),
        scale=scale,
        chunk_size=chunk_size,
        g=g.npu() if g is not None else None,
        gK=None,
        h0=h0.npu() if h0 is not None else None,
        dht=dht.npu() if dht is not None else None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    torch.npu.synchronize()
    npu_elapsed = time.time() - t0

    dh_fp64, dh0_fp64, dv2_fp64 = chunk_gated_delta_rule_bwd_dhu_torch(
        q.double(),
        k.double(),
        w.double(),
        d_o.double(),
        dv.double(),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        g=g.double() if g is not None else None,
        scale=scale,
        chunk_size=chunk_size,
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

    dual_then_viz("dh", dh_npu, dh_fp64, gpu_dh, viz_dir=tensor_viz_dir,
                  sample_count=sample_count, enable_viz=enable_viz)
    dual_then_viz("dv2", dv2_npu, dv2_fp64, gpu_dv2, viz_dir=tensor_viz_dir,
                  sample_count=sample_count, enable_viz=enable_viz)
    if gpu_dh0 is not None and dh0_npu is not None:
        dual_then_viz(
            "dh0",
            dh0_npu,
            dh0_fp64 if dh0_fp64 is not None else dh0_npu,
            gpu_dh0,
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
    p = argparse.ArgumentParser(description="bwd_dhu NPU vs GPU dual benchmark from GPU dumps")
    p.add_argument("--dump-root", type=Path, default=None)
    p.add_argument("--pt", type=Path, default=None)
    p.add_argument("--pts", default="")
    p.add_argument("--case", default="")
    p.add_argument("--cases", default="")
    p.add_argument("--phase", default="all", help="all | prefix:phase_1_ | prefix:gva_")
    p.add_argument("--report", type=Path, default=None)
    add_viz_cli_args(p)
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
    pt_paths = _collect_pt_paths(args)
    enable_viz = not args.no_viz
    sample_count = args.sample_count
    results: list[dict[str, Any]] = []
    failed = 0

    if pt_paths:
        for pt_path in pt_paths:
            try:
                results.append(run_one_pt(
                    pt_path,
                    label=pt_path.name,
                    verbose=True,
                    enable_viz=enable_viz,
                    sample_count=sample_count,
                    viz_dir=args.viz_dir,
                ))
            except Exception as e:
                failed += 1
                print(f"\n=== {pt_path.name} FAILED ===\n{e}", flush=True)
                traceback.print_exc()
                results.append({"case": pt_path.name, "status": "fail", "pt": str(pt_path), "error": str(e)})
        default_report_dir = pt_paths[0].resolve().parent
    else:
        if args.dump_root is None:
            print("ERROR: provide --dump-root or --pt/--pts", file=sys.stderr)
            return 2
        selected = _select_cases(args.dump_root, args)
        if not selected:
            print("No cases selected.", file=sys.stderr)
            return 1
        for case_dir in selected:
            try:
                results.append(run_one_case(
                    case_dir,
                    verbose=True,
                    enable_viz=enable_viz,
                    sample_count=sample_count,
                    viz_dir=args.viz_dir,
                ))
            except Exception as e:
                failed += 1
                print(f"\n=== {case_dir.name} FAILED ===\n{e}", flush=True)
                traceback.print_exc()
                results.append({"case": case_dir.name, "status": "fail", "error": str(e)})
        default_report_dir = args.dump_root

    report = {
        "op": OP_NAME,
        "mode": "pt" if pt_paths else "case_dir",
        "total": len(results),
        "passed": len(results) - failed,
        "failed": failed,
        "results": results,
    }
    report_path = args.report or (default_report_dir / "bwd_dhu_gpu_dump_dual_report.json")
    with Path(report_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nDone: {report['passed']}/{report['total']} passed, report -> {report_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
