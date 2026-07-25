#!/usr/bin/env python3
"""Seed-aligned dual: NPU vs CPU (optional GPU dump outputs).

Inputs are built with **CPU RNG** in **BTHD** (same as GPU FLA dump scripts),
then transposed to **BNSD** for NPU. No need to copy input tensors.

Seed rule (must match GPU ``run_intra_sub_chunk_dump_cases.py``):
  seed_i = base_seed + case_index * 9973
where ``case_index`` is the index in the **filtered** case list
(same ``--phase`` / ``--names`` / ``--include-disabled``).

Usage:
  python test_intra_sub_chunk_seed_dual.py --phase smoke
  python test_intra_sub_chunk_seed_dual.py --names smoke_mha_fix,BSND_GVA_V256_28 --seed 0
  # optional 3-way if you still have GPU output dumps:
  python test_intra_sub_chunk_seed_dual.py --phase smoke --gpu-dump-root /data/isub_dump
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import torch

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parents[5]
for p in (str(TEST_DIR), str(REPO_ROOT / "torch_custom" / "fla_npu")):
    if p not in sys.path:
        sys.path.insert(0, p)

from isub_gpu_dump_dual_utils import add_viz_cli_args, dual_then_viz, print_err_stats  # noqa: E402
from isub_gpu_dump_loader import bthd_to_bnsd, find_isub_dump_pt, load_dump_for_npu  # noqa: E402
from isub_seed_case_utils import (  # noqa: E402
    build_intra_sub_chunk_inputs,
    case_seed,
    filter_cases,
    load_cases,
)

# NPU supports cs=128; do not apply GPU-only chunk filter.


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="intra_sub_chunk seed dual (CPU RNG, no input dump)")
    p.add_argument(
        "--cases-file",
        type=Path,
        default=TEST_DIR / "intra_sub_chunk_cases.json",
    )
    p.add_argument("--phase", default="smoke", help="all|smoke|gdn|varlen|gva|prefix:")
    p.add_argument("--names", default="", help="comma-separated case names (overrides phase)")
    p.add_argument("--include-disabled", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="base seed; case i → seed+i*9973")
    p.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", "0")))
    p.add_argument(
        "--cpu-dtype",
        default="fp32",
        choices=("fp32", "fp64"),
        help="local CPU golden dtype (same seed path as GPU)",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--run-cpu",
        dest="run_cpu",
        action="store_true",
        default=True,
        help="run CPU golden (default: on)",
    )
    g.add_argument(
        "--no-cpu",
        dest="run_cpu",
        action="store_false",
        help="skip CPU golden (NPU only)",
    )
    g.add_argument(
        "--cpu-only",
        action="store_true",
        help="only CPU golden (skip NPU; no device init)",
    )
    p.add_argument(
        "--save-cpu",
        action="store_true",
        help="save aqk_cpu/akkd_cpu under out-dir/<case>/cpu_golden.pt (BNSD)",
    )
    p.add_argument(
        "--gpu-dump-root",
        type=Path,
        default=None,
        help="optional: load GPU aqk/akkd from dump for 3-way ct.dual (inputs still from seed)",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="report/viz root")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    add_viz_cli_args(p)
    return p.parse_args()


def _bthd_bundle_to_bnsd(bundle: dict[str, Any]) -> dict[str, Any]:
    """Convert CPU/GPU BTHD bundle → NPU BNSD (tensors on CPU)."""
    q = bthd_to_bnsd(bundle["q"].detach().cpu())
    k = bthd_to_bnsd(bundle["k"].detach().cpu())
    g = bthd_to_bnsd(bundle["g"].detach().cpu())
    beta = bthd_to_bnsd(bundle["beta"].detach().cpu())  # [B,T,HV]→[B,HV,T]
    cu = bundle["cu_seqlens"]
    idx = bundle["chunk_indices"]
    cu_list = None
    idx_list = None
    if cu is not None:
        cu_list = [int(x) for x in cu.detach().cpu().flatten().tolist()]
    if idx is not None:
        idx_list = [int(x) for x in idx.detach().cpu().reshape(-1).tolist()]
    return {
        "q": q,
        "k": k,
        "g": g,
        "beta": beta,
        "scale": float(bundle["scale"]),
        "chunk_size": int(bundle["chunk_size"]),
        "cu_seqlens": cu_list,
        "chunk_indices": idx_list,
        "meta": bundle["meta"],
    }


def _run_cpu_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu: Optional[list[int]],
    idx: Optional[list[int]],
    cpu_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    from test_chunk_kda_fwd_intra_sub_chunk import chunk_kda_fwd_intra_sub_chunk_ref

    cu_t = None if cu is None else torch.tensor(cu, dtype=torch.long)
    idx_t = None if idx is None else torch.tensor(idx, dtype=torch.long)
    return chunk_kda_fwd_intra_sub_chunk_ref(
        q, k, g, beta, scale, chunk_size, cu_t, idx_t, dtype=cpu_dtype
    )


def _try_load_gpu_outputs(dump_root: Path, case_name: str) -> Optional[dict[str, torch.Tensor]]:
    case_dir = dump_root / case_name
    if not case_dir.is_dir():
        return None
    try:
        pt = find_isub_dump_pt(case_dir)
        _ins, _meta, outs = load_dump_for_npu(pt)
    except (OSError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"  [warn] GPU dump load failed for {case_name}: {exc}", flush=True)
        return None
    if "aqk" not in outs or "akkd" not in outs:
        return None
    return {"aqk": outs["aqk"].float(), "akkd": outs["akkd"].float()}


def run_one_case(
    case: dict[str, Any],
    *,
    seed: int,
    cpu_dtype: torch.dtype,
    run_cpu: bool,
    run_npu: bool,
    save_cpu: bool,
    out_dir: Path,
    enable_viz: bool,
    sample_count: int,
    viz_dir: Path | None,
    gpu_dump_root: Path | None,
) -> dict[str, Any]:
    # Same construction as GPU: BTHD on CPU RNG → BNSD for NPU/CPU ref.
    bundle = build_intra_sub_chunk_inputs(
        case, device=torch.device("cpu"), seed=seed, rng_on_cpu=True
    )
    npu_in = _bthd_bundle_to_bnsd(bundle)
    q, k, g, beta = npu_in["q"], npu_in["k"], npu_in["g"], npu_in["beta"]
    scale = npu_in["scale"]
    BT = npu_in["chunk_size"]
    cu, idx = npu_in["cu_seqlens"], npu_in["chunk_indices"]
    name = str(case["name"])

    B, H, T, K = q.shape
    HV = g.shape[1]
    print(
        f"\n=== {name} === seed={seed} BTHD→BNSD "
        f"B={B} H={H} HV={HV} T={T} K={K} BT={BT} varlen={cu is not None} "
        f"run_cpu={run_cpu} run_npu={run_npu}",
        flush=True,
    )

    aqk_cpu_f = akkd_cpu_f = None
    t_cpu = None
    if run_cpu:
        t0 = time.time()
        aqk_cpu, akkd_cpu = _run_cpu_ref(q, k, g, beta, scale, BT, cu, idx, cpu_dtype)
        t_cpu = time.time() - t0
        aqk_cpu_f = aqk_cpu.float()
        akkd_cpu_f = akkd_cpu.float()
        if save_cpu:
            cdir = out_dir / name
            cdir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "op": "chunk_kda_fwd_intra_sub_chunk_cpu",
                    "seed": seed,
                    "layout": "BNSD",
                    "outputs": {"aqk": aqk_cpu_f.cpu(), "akkd": akkd_cpu_f.cpu()},
                    "meta": npu_in["meta"],
                },
                cdir / "cpu_golden.pt",
            )

    aqk_n_c = akkd_n_c = None
    t_npu = None
    if run_npu:
        import fla_npu.ops.ascendc as ascendc_ops  # noqa: F401

        t1 = time.time()
        aqk_n, akkd_n = ascendc_ops.npu_chunk_kda_fwd_intra_sub_chunk(
            q.npu(),
            k.npu(),
            g.npu(),
            beta.npu(),
            scale,
            BT,
            cu_seqlens=cu,
            chunk_indices=idx,
        )
        torch.npu.synchronize()
        t_npu = time.time() - t1
        aqk_n_c = aqk_n.float().cpu()
        akkd_n_c = akkd_n.float().cpu()

    stats: dict[str, Any] = {}
    if run_npu and run_cpu and aqk_n_c is not None and aqk_cpu_f is not None:
        stats["aqk_npu_vs_cpu"] = print_err_stats("aqk npu↔cpu", aqk_n_c, aqk_cpu_f)
        stats["akkd_npu_vs_cpu"] = print_err_stats("akkd npu↔cpu", akkd_n_c, akkd_cpu_f)

    gpu_outs = None
    if run_npu and gpu_dump_root is not None and aqk_n_c is not None:
        gpu_outs = _try_load_gpu_outputs(gpu_dump_root, name)
        if gpu_outs is not None:
            stats["aqk_npu_vs_gpu"] = print_err_stats("aqk npu↔gpu", aqk_n_c, gpu_outs["aqk"])
            stats["akkd_npu_vs_gpu"] = print_err_stats("akkd npu↔gpu", akkd_n_c, gpu_outs["akkd"])

    tensor_viz = None
    if enable_viz and run_npu and run_cpu:
        base = viz_dir or Path("./viz_isub_seed_dual")
        tensor_viz = Path(base) / name

    if gpu_outs is not None and aqk_n_c is not None and aqk_cpu_f is not None:
        dual_then_viz(
            "aqk", aqk_n_c, aqk_cpu_f, gpu_outs["aqk"],
            viz_dir=tensor_viz, sample_count=sample_count, enable_viz=enable_viz,
        )
        dual_then_viz(
            "akkd", akkd_n_c, akkd_cpu_f, gpu_outs["akkd"],
            viz_dir=tensor_viz, sample_count=sample_count, enable_viz=enable_viz,
        )
    elif enable_viz and tensor_viz is not None and aqk_n_c is not None and aqk_cpu_f is not None:
        import ct

        tensor_viz.mkdir(parents=True, exist_ok=True)
        for tname, npu_t, cpu_t in (
            ("aqk", aqk_n_c, aqk_cpu_f),
            ("akkd", akkd_n_c, akkd_cpu_f),
        ):
            print(f"  [{tname}] ct.viz(npu, cpu)", flush=True)
            ct.viz(
                npu_t, cpu_t,
                out_dir=str(tensor_viz),
                name=tname,
                sample_count=sample_count,
            )

    rec: dict[str, Any] = {
        "case": name,
        "status": "pass",
        "seed": seed,
        "run_cpu": run_cpu,
        "run_npu": run_npu,
        "has_gpu_dump": gpu_outs is not None,
        "stats": stats,
        "meta": npu_in["meta"],
    }
    if t_npu is not None:
        rec["t_npu_s"] = round(t_npu, 4)
    if t_cpu is not None:
        rec["t_cpu_s"] = round(t_cpu, 4)
    return rec


def main() -> int:
    args = _parse_args()
    run_cpu = bool(args.run_cpu or args.cpu_only)
    run_npu = not bool(args.cpu_only)

    if run_npu:
        import torch_npu  # noqa: F401

        torch.npu.config.allow_internal_format = False
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.set_device(int(args.device))

    cases = load_cases(args.cases_file)
    names = [n.strip() for n in args.names.split(",") if n.strip()] or None
    selected = filter_cases(
        cases,
        phase=args.phase,
        names=names,
        include_disabled=args.include_disabled,
    )
    print(
        f"cases_file={args.cases_file} phase={args.phase} selected={len(selected)} "
        f"base_seed={args.seed} rng=CPU BTHD→BNSD run_cpu={run_cpu} run_npu={run_npu} "
        f"device=npu:{args.device}"
    )
    if args.dry_run:
        for i, c in enumerate(selected):
            print(f"  [{i}] {c['name']} seed={case_seed(args.seed, i)} cs={c.get('chunk_size')}")
        return 0

    if not selected:
        print("ERROR: no cases", file=sys.stderr)
        return 1
    if not run_cpu and not run_npu:
        print("ERROR: nothing to run", file=sys.stderr)
        return 1

    if run_npu and run_cpu and not args.no_viz:
        try:
            import ct  # noqa: F401
        except ImportError as exc:
            raise SystemExit("pip install ct") from exc

    out_dir = args.out_dir or Path("./isub_seed_dual_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "intra_sub_chunk_seed_dual_report.json"
    cpu_dtype = torch.float64 if args.cpu_dtype == "fp64" else torch.float32

    results: list[dict[str, Any]] = []
    ok = fail = 0
    for i, case in enumerate(selected):
        name = str(case["name"])
        seed_i = case_seed(args.seed, i)
        print(f"[{i+1}/{len(selected)}] RUN {name} seed={seed_i}", flush=True)
        try:
            rec = run_one_case(
                case,
                seed=seed_i,
                cpu_dtype=cpu_dtype,
                run_cpu=run_cpu,
                run_npu=run_npu,
                save_cpu=args.save_cpu,
                out_dir=out_dir,
                enable_viz=not args.no_viz,
                sample_count=args.sample_count,
                viz_dir=args.viz_dir or (out_dir / "viz"),
                gpu_dump_root=args.gpu_dump_root,
            )
            results.append(rec)
            ok += 1
            bits = []
            if "t_npu_s" in rec:
                bits.append(f"npu={rec['t_npu_s']}s")
            if "t_cpu_s" in rec:
                bits.append(f"cpu={rec['t_cpu_s']}s")
            print(f"  PASS {name} {' '.join(bits)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}: {exc}", flush=True)
            traceback.print_exc()
            results.append({
                "case": name,
                "status": "fail",
                "seed": seed_i,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            fail += 1

    report = {
        "mode": "seed_dual",
        "rng": "cpu",
        "gen_layout": "BTHD",
        "npu_layout": "BNSD",
        "base_seed": args.seed,
        "seed_rule": "base + index * 9973",
        "run_cpu": run_cpu,
        "run_npu": run_npu,
        "cpu_dtype": args.cpu_dtype,
        "phase": args.phase,
        "passed": ok,
        "failed": fail,
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] pass={ok} fail={fail} → {report_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
