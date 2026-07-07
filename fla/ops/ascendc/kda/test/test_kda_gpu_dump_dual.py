#!/usr/bin/env python3
"""KDA chunk_kda_fwd NPU vs GPU dual benchmark using GPU-collected .pt dumps.

Compare: ct.dual(npu_out, cpu_fp64_golden, gpu_out_from_dump)
Tensors: o, final_state (g / initial_state outputs are ignored for now).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import fla_npu
import torch
import torch_npu

KDA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(KDA_DIR))
sys.path.insert(0, str(REPO_ROOT))

from gpu_dump_loader import (
    find_op_dump_pt,
    list_case_dirs,
    load_case_meta,
    load_dump_for_kda,
    resolve_seq_meta,
)
from gpu_dump_dual_utils import dual_then_viz, resolve_viz_dir
from gpu_dump_dual_runner import add_skip_cli_args, run_dual_batch
from kda_dump_adapter import (
    is_supported_by_pr152,
    prepare_beta,
    prepare_gk,
    prepare_npu_gate_tensors,
    resolve_scale,
)
from tests.reference.chunk_kda_reference import chunk_kda_forward_reference

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

OP_NAME = "chunk_kda_fwd"
COMPARE_TENSORS = ("o", "final_state")


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

    inputs, meta, gpu_outputs = load_dump_for_kda(pt_path)
    _check_op_name(pt_path, meta)

    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    beta_raw = inputs["beta"]

    ok, skip_reason = is_supported_by_pr152(v)
    case_name = label or f"{pt_path.parent.name}/{pt_path.name}"
    if not ok:
        if verbose:
            print(f"\n=== {case_name} SKIPPED ===\n  reason: {skip_reason}", flush=True)
        return {
            "case": case_name,
            "status": "skipped",
            "pt": str(pt_path),
            "skip_reason": skip_reason,
        }

    cu_seqlens, _chunk_indices, chunk_size, _scale_meta = resolve_seq_meta(meta, case_meta)
    scale = resolve_scale(inputs, meta, case_meta, q)
    gk = prepare_gk(inputs, meta, chunk_size)
    beta = prepare_beta(
        beta_raw,
        use_beta_sigmoid_in_kernel=bool(meta.get("use_beta_sigmoid_in_kernel")),
        allow_neg_eigval=bool(meta.get("allow_neg_eigval")),
        dtype=beta_raw.dtype if beta_raw.is_floating_point() else torch.float32,
    )

    initial_state = inputs.get("initial_state")
    gk, beta, initial_state = prepare_npu_gate_tensors(gk, beta, initial_state)
    output_final_state = bool(meta.get("output_final_state", gpu_outputs.get("final_state") is not None))

    B, T, Hk, K = q.shape
    Hv = v.shape[2]
    V = v.shape[-1]

    if verbose:
        print(
            f"\n=== {case_name} ===\n"
            f"  pt: {pt_path} B={B} Hk={Hk} Hv={Hv} T={T} K={K} V={V} cs={chunk_size} "
            f"varlen={cu_seqlens is not None} dtype={q.dtype} "
            f"gk={gk.dtype} beta={beta.dtype}",
            flush=True,
        )

    npu_kwargs: dict[str, Any] = {
        "output_final_state": output_final_state,
        "return_intermediate": False,
    }
    if initial_state is not None:
        npu_kwargs["initial_state"] = initial_state.npu()
    if cu_seqlens is not None:
        npu_kwargs["cu_seqlens"] = cu_seqlens

    if verbose:
        print("  [npu] running npu_chunk_kda_fwd ...", flush=True)
    t0 = time.time()
    got = torch.ops.npu.npu_chunk_kda_fwd(
        q.npu(),
        k.npu(),
        v.npu(),
        gk.npu(),
        beta.npu(),
        scale,
        chunk_size,
        **npu_kwargs,
    )
    torch.npu.synchronize()
    npu_elapsed = time.time() - t0
    if verbose:
        print(f"  [npu] done in {npu_elapsed:.3f}s", flush=True)

    if isinstance(got, (tuple, list)):
        o_npu = got[0]
        final_state_npu = got[1] if len(got) > 1 else None
    else:
        o_npu = got
        final_state_npu = None

    if verbose:
        print("  [cpu] running fp64 reference ...", flush=True)
    cu_tensor = torch.tensor(cu_seqlens, dtype=torch.int64) if cu_seqlens is not None else None
    ref = chunk_kda_forward_reference(
        q.double(),
        k.double(),
        v.double(),
        gk.double(),
        beta.double(),
        scale=scale,
        chunk_size=chunk_size,
        initial_state=initial_state.double() if initial_state is not None else None,
        output_final_state=output_final_state,
        cu_seqlens=cu_tensor,
    )

    gpu_o = gpu_outputs["o"]
    gpu_final_state = gpu_outputs.get("final_state")

    viz_case_name = str(case_meta.get("case_name") or pt_path.parent.name)
    tensor_viz_dir = None
    if enable_viz:
        base_viz_dir = resolve_viz_dir(
            viz_dir=viz_dir,
            pt_path=pt_path,
            case_dir=None,
            default_report_dir=pt_path.parent,
        )
        tensor_viz_dir = (base_viz_dir / viz_case_name).resolve()
        if verbose:
            print(f"  [viz] output dir: {tensor_viz_dir}", flush=True)

    dual_then_viz(
        "o",
        o_npu,
        ref.o,
        gpu_o,
        viz_dir=tensor_viz_dir,
        sample_count=sample_count,
        enable_viz=enable_viz,
        viz_name_prefix=f"{viz_case_name}_o_npu_vs_fp64",
    )

    if output_final_state and gpu_final_state is not None and final_state_npu is not None:
        dual_then_viz(
            "final_state",
            final_state_npu,
            ref.final_state,
            gpu_final_state,
            viz_dir=tensor_viz_dir,
            sample_count=sample_count,
            enable_viz=enable_viz,
            viz_name_prefix=f"{viz_case_name}_final_state_npu_vs_fp64",
        )
    elif verbose and output_final_state:
        print("  [final_state] skipped (NPU or GPU dump missing final_state)", flush=True)

    result = {
        "case": case_name,
        "status": "pass",
        "pt": str(pt_path),
        "npu_elapsed_s": round(npu_elapsed, 4),
        "shapes": {
            "B": B,
            "Hk": Hk,
            "Hv": Hv,
            "T": T,
            "K": K,
            "V": V,
            "chunk_size": chunk_size,
        },
        "compare_tensors": list(COMPARE_TENSORS),
    }
    sidecar = pt_path.parent / f".kda_dual_result_{pt_path.stem}.json"
    sidecar.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_one_case(
    case_dir: Path,
    *,
    verbose: bool = True,
    enable_viz: bool = True,
    sample_count: int = 200_000,
    viz_dir: Path | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    pt_path, _raw = find_op_dump_pt(case_dir, OP_NAME)
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
    p = argparse.ArgumentParser(description="KDA chunk_kda_fwd NPU vs GPU dual benchmark from GPU dumps")
    p.add_argument(
        "--dump-root",
        type=Path,
        default=None,
        help="GPU dump root for batch mode (e.g. /data/kda_dump/all)",
    )
    p.add_argument("--pt", type=Path, default=None, help="single dump .pt file")
    p.add_argument("--pts", default="", help="comma-separated dump .pt files")
    p.add_argument("--case", default="", help="single case directory name under dump-root")
    p.add_argument("--cases", default="", help="comma-separated case names")
    p.add_argument(
        "--phase",
        default="all",
        help="filter case dirs: all | smoke | prefix:<name_prefix>",
    )
    p.add_argument(
        "--max-t",
        type=int,
        default=None,
        help="skip cases with T greater than this (e.g. 1024 while debugging large-seq crashes)",
    )
    p.add_argument(
        "--isolated",
        action="store_true",
        help="run each case in a subprocess so a kernel segfault does not abort the whole batch",
    )
    p.add_argument("--report", type=Path, default=None, help="write JSON report path")
    add_skip_cli_args(p)
    return p.parse_args()


def _run_isolated_batch(args: argparse.Namespace) -> int:
    """Run batch cases one subprocess each; merge JSON report at the end."""
    if args.dump_root is None:
        print("ERROR: --isolated batch mode requires --dump-root", file=sys.stderr)
        return 2
    selected = _select_cases(args.dump_root, args)
    if not selected:
        print("No cases selected.", file=sys.stderr)
        return 1

    report_path = args.report or (args.dump_root / "kda_gpu_dump_dual_report.json")
    results: list[dict[str, Any]] = []
    failed = 0
    py = str(Path(__file__).resolve())
    base_cmd = [sys.executable, py, "--no-viz"] if args.no_viz else [sys.executable, py]
    if args.sample_count != 200_000:
        base_cmd.extend(["-sc", str(args.sample_count)])
    if args.viz_dir is not None:
        base_cmd.extend(["--viz-dir", str(args.viz_dir)])
    if args.force:
        base_cmd.append("--force")

    for case_dir in selected:
        pt_path = case_dir / f"001_{OP_NAME}.pt"
        if not pt_path.is_file():
            pt_path = find_op_dump_pt(case_dir, OP_NAME)[0]
        cmd = base_cmd + ["--pt", str(pt_path)]
        print(f"\n=== isolated: {case_dir.name} ===", flush=True)
        proc = subprocess.run(cmd, check=False)
        sidecar = pt_path.parent / f".kda_dual_result_{pt_path.stem}.json"
        if proc.returncode == 0 and sidecar.is_file():
            results.append(json.loads(sidecar.read_text(encoding="utf-8")))
            sidecar.unlink(missing_ok=True)
        else:
            failed += 1
            results.append({
                "case": case_dir.name,
                "status": "fail",
                "pt": str(pt_path),
                "error": f"subprocess exit {proc.returncode}",
            })

    from gpu_dump_dual_runner import write_report

    report = write_report(
        report_path,
        op_name=OP_NAME,
        mode="case_dir_isolated",
        results=results,
        extra={"dump_root": str(args.dump_root), "isolated": True},
    )
    print(
        f"\nDone: {report['passed']} passed, {report['skipped']} skipped, "
        f"{report['failed']} failed / {report['total']} total",
        flush=True,
    )
    print(f"report -> {report_path}", flush=True)
    return 1 if failed else 0


def _case_sort_key(case_dir: Path) -> tuple[int, int, str]:
    meta = load_case_meta(case_dir)
    name = case_dir.name
    smoke_rank = 0 if name.startswith("smoke_") else 1
    t = int(meta.get("T") or 0)
    return smoke_rank, t, name


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
        selected = all_dirs
    elif phase == "smoke":
        selected = [d for d in all_dirs if d.name.startswith("smoke_")]
    elif phase.startswith("prefix:"):
        prefix = phase.split(":", 1)[1]
        selected = [d for d in all_dirs if d.name.startswith(prefix)]
    else:
        raise ValueError(f"unknown --phase {args.phase!r}; use all, smoke, or prefix:<name_prefix>")

    if args.max_t is not None:
        kept: list[Path] = []
        for case_dir in selected:
            meta = load_case_meta(case_dir)
            t = int(meta.get("T") or 0)
            if t <= args.max_t:
                kept.append(case_dir)
            else:
                print(f"SKIP {case_dir.name}: T={t} > max_t={args.max_t}", flush=True)
        selected = kept

    return sorted(selected, key=_case_sort_key)


def main() -> int:
    args = _parse_args()
    if args.isolated and not _collect_pt_paths(args):
        return _run_isolated_batch(args)
    pt_paths = _collect_pt_paths(args)
    report_extra: dict[str, Any] = {}
    if pt_paths:
        report_extra["pt_files"] = [str(p) for p in pt_paths]
    else:
        report_extra["dump_root"] = str(args.dump_root) if args.dump_root else None
    return run_dual_batch(
        args,
        op_name=OP_NAME,
        report_basename="kda_gpu_dump_dual_report.json",
        viz_tensor_names=("_o_npu_vs_fp64", "_final_state_npu_vs_fp64"),
        collect_pt_paths=_collect_pt_paths,
        select_cases=_select_cases,
        run_one_pt=run_one_pt,
        run_one_case=run_one_case,
        report_extra=report_extra,
    )


if __name__ == "__main__":
    raise SystemExit(main())
