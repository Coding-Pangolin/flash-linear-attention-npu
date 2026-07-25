#!/usr/bin/env python3
"""NPU vs GPU(+CPU dump) dual for chunk_kda_fwd_intra_sub_chunk.

Dump layout is **BTHD** (GPU FLA); this script converts to **BNSD** for NPU.
CPU golden prefers ``aqk_cpu``/``akkd_cpu`` from the same dump (same inputs).

Usage:
  python test_intra_sub_chunk_gpu_dump_dual.py --dump-root /path/to/isub_dump
  python test_intra_sub_chunk_gpu_dump_dual.py --pt /path/to/001_....pt
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
from isub_gpu_dump_loader import (  # noqa: E402
    OP_NAME,
    find_isub_dump_pt,
    list_case_dirs,
    load_case_meta,
    load_dump_for_npu,
    resolve_seq_for_npu,
)

OP_TAG = "intra_sub_chunk"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="intra_sub_chunk GPU dump dual (NPU vs GPU/CPU)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dump-root", type=Path, help="directory of per-case dump folders")
    g.add_argument("--pt", type=Path, help="single 001_*.pt dump")
    g.add_argument("--case-dir", type=Path, help="single case directory")
    p.add_argument("--names", default="", help="comma-separated case folder names")
    p.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", "0")))
    p.add_argument(
        "--recompute-cpu",
        action="store_true",
        help="ignore dump CPU; recompute local ref on BNSD inputs (fp32)",
    )
    p.add_argument(
        "--cpu-dtype",
        default="fp32",
        choices=("fp32", "fp64"),
        help="only with --recompute-cpu",
    )
    p.add_argument("--force", action="store_true", help="re-run even if report pass exists")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="default: <dump-root>/intra_sub_chunk_gpu_dump_dual_report.json",
    )
    add_viz_cli_args(p)
    return p.parse_args()


def _maybe_recompute_cpu(
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
    from test_chunk_kda_fwd_intra_sub_chunk import (
        chunk_kda_fwd_intra_sub_chunk_ref,
    )

    cu_t = None if cu is None else torch.tensor(cu, dtype=torch.long)
    idx_t = None if idx is None else torch.tensor(idx, dtype=torch.long)
    return chunk_kda_fwd_intra_sub_chunk_ref(
        q, k, g, beta, scale, chunk_size, cu_t, idx_t, dtype=cpu_dtype
    )


def run_one_pt(
    pt_path: Path,
    *,
    case_meta: dict[str, Any] | None = None,
    label: str | None = None,
    enable_viz: bool = True,
    sample_count: int = 200_000,
    viz_dir: Path | None = None,
    recompute_cpu: bool = False,
    cpu_dtype_name: str = "fp32",
) -> dict[str, Any]:
    import fla_npu.ops.ascendc as ascendc_ops  # noqa: F401

    pt_path = pt_path.resolve()
    if case_meta is None:
        case_meta = load_case_meta(pt_path.parent)

    inputs, meta, outputs = load_dump_for_npu(pt_path)
    op = str(meta.get("op") or "")
    if op and op not in (OP_NAME, f"001_{OP_NAME}"):
        # allow step filename ops
        if OP_NAME not in op and "intra_sub_chunk" not in op:
            raise ValueError(f"{pt_path}: unexpected op={op!r}")

    q, k, g, beta = inputs["q"], inputs["k"], inputs["g"], inputs["beta"]
    # shapes after BTHD→BNSD
    B, H, T, K = q.shape
    HV = g.shape[1]
    cu, idx, chunk_size, scale = resolve_seq_for_npu(meta, case_meta)
    if scale <= 0:
        scale = 1.0 / math.sqrt(K)

    gpu_aqk = outputs.get("aqk")
    gpu_akkd = outputs.get("akkd")
    if gpu_aqk is None or gpu_akkd is None:
        raise KeyError(f"{pt_path}: dump missing GPU outputs aqk/akkd")

    cpu_dtype = torch.float64 if cpu_dtype_name == "fp64" else torch.float32
    if recompute_cpu or "aqk_cpu" not in outputs:
        if "aqk_cpu" not in outputs:
            print("  [warn] no aqk_cpu in dump → recompute local CPU ref", flush=True)
        aqk_cpu, akkd_cpu = _maybe_recompute_cpu(
            q, k, g, beta, scale, chunk_size, cu, idx, cpu_dtype
        )
        cpu_src = "recompute"
    else:
        aqk_cpu = outputs["aqk_cpu"]
        akkd_cpu = outputs["akkd_cpu"]
        cpu_src = "dump"

    case_name = label or f"{pt_path.parent.name}/{pt_path.name}"
    print(
        f"\n=== {case_name} ===\n"
        f"  layout dump=BTHD → npu=BNSD | B={B} H={H} HV={HV} T={T} K={K} "
        f"BT={chunk_size} varlen={cu is not None} cpu_src={cpu_src}",
        flush=True,
    )

    t0 = time.time()
    aqk_n, akkd_n = ascendc_ops.npu_chunk_kda_fwd_intra_sub_chunk(
        q.npu(),
        k.npu(),
        g.npu(),
        beta.npu(),
        scale,
        chunk_size,
        cu_seqlens=cu,
        chunk_indices=idx,
    )
    torch.npu.synchronize()
    npu_s = time.time() - t0

    aqk_n_c = aqk_n.float().cpu()
    akkd_n_c = akkd_n.float().cpu()
    aqk_cpu_f = aqk_cpu.float()
    akkd_cpu_f = akkd_cpu.float()
    aqk_gpu_f = gpu_aqk.float()
    akkd_gpu_f = gpu_akkd.float()

    # shape guard
    if tuple(aqk_n_c.shape) != tuple(aqk_gpu_f.shape):
        raise RuntimeError(
            f"aqk shape mismatch npu={tuple(aqk_n_c.shape)} gpu={tuple(aqk_gpu_f.shape)}"
        )

    stats = {
        "aqk_npu_vs_gpu": print_err_stats("aqk npu↔gpu", aqk_n_c, aqk_gpu_f),
        "aqk_npu_vs_cpu": print_err_stats("aqk npu↔cpu", aqk_n_c, aqk_cpu_f),
        "akkd_npu_vs_gpu": print_err_stats("akkd npu↔gpu", akkd_n_c, akkd_gpu_f),
        "akkd_npu_vs_cpu": print_err_stats("akkd npu↔cpu", akkd_n_c, akkd_cpu_f),
    }

    tensor_viz = None
    if enable_viz:
        base = viz_dir or (pt_path.parent / "viz")
        tensor_viz = Path(base) / case_name.replace("/", "_")

    dual_then_viz(
        "aqk", aqk_n_c, aqk_cpu_f, aqk_gpu_f,
        viz_dir=tensor_viz, sample_count=sample_count, enable_viz=enable_viz,
    )
    dual_then_viz(
        "akkd", akkd_n_c, akkd_cpu_f, akkd_gpu_f,
        viz_dir=tensor_viz, sample_count=sample_count, enable_viz=enable_viz,
    )

    return {
        "case": case_name,
        "status": "pass",
        "pt": str(pt_path),
        "npu_s": round(npu_s, 4),
        "cpu_src": cpu_src,
        "shapes": {
            "aqk": list(aqk_n_c.shape),
            "akkd": list(akkd_n_c.shape),
        },
        "stats": stats,
        "meta": {
            "B": B, "H": H, "HV": HV, "T": T, "K": K,
            "chunk_size": chunk_size, "varlen": cu is not None,
            "dump_layout": "BTHD", "npu_layout": "BNSD",
        },
    }


def _collect_pts(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.pt is not None:
        return [(args.pt.parent.name, args.pt.resolve())]
    if args.case_dir is not None:
        d = args.case_dir.resolve()
        return [(d.name, find_isub_dump_pt(d))]

    root = args.dump_root.resolve()
    names = [n.strip() for n in args.names.split(",") if n.strip()] or None
    dirs = list_case_dirs(root)
    if names:
        by = {d.name: d for d in dirs}
        missing = [n for n in names if n not in by]
        if missing:
            raise FileNotFoundError(f"case dirs not found under {root}: {missing}")
        dirs = [by[n] for n in names]
    return [(d.name, find_isub_dump_pt(d)) for d in dirs]


def main() -> int:
    args = _parse_args()

    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(int(args.device))

    try:
        import ct  # noqa: F401
    except ImportError as exc:
        raise SystemExit("pip install ct  # required for ct.dual / ct.viz") from exc

    items = _collect_pts(args)
    if not items:
        print("ERROR: no dump cases found", file=sys.stderr)
        return 1

    if args.dump_root is not None:
        report_path = args.report or (args.dump_root / "intra_sub_chunk_gpu_dump_dual_report.json")
    else:
        report_path = args.report or Path("./intra_sub_chunk_gpu_dump_dual_report.json")

    existing: dict[str, Any] = {}
    if report_path.is_file() and not args.force:
        try:
            existing = {
                r["case"]: r
                for r in json.loads(report_path.read_text(encoding="utf-8")).get("results", [])
                if r.get("case")
            }
        except (OSError, json.JSONDecodeError, KeyError):
            existing = {}

    results: list[dict[str, Any]] = []
    ok = fail = skip = 0
    print(f"cases={len(items)} device=npu:{args.device} report={report_path}", flush=True)

    for i, (name, pt) in enumerate(items):
        case_key = f"{name}/{pt.name}"
        if not args.force and existing.get(case_key, {}).get("status") == "pass":
            print(f"[{i+1}/{len(items)}] SKIP {case_key} (report pass)")
            results.append({**existing[case_key], "status": "skipped"})
            skip += 1
            continue

        print(f"[{i+1}/{len(items)}] RUN  {case_key}", flush=True)
        try:
            rec = run_one_pt(
                pt,
                case_meta=load_case_meta(pt.parent),
                label=case_key,
                enable_viz=not args.no_viz,
                sample_count=args.sample_count,
                viz_dir=args.viz_dir,
                recompute_cpu=args.recompute_cpu,
                cpu_dtype_name=args.cpu_dtype,
            )
            results.append(rec)
            ok += 1
            print(f"  PASS {case_key} npu={rec['npu_s']}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {case_key}: {exc}", flush=True)
            traceback.print_exc()
            results.append({
                "case": case_key,
                "status": "fail",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            fail += 1

    report = {
        "op": OP_TAG,
        "dump_layout": "BTHD",
        "npu_layout": "BNSD",
        "note": "loader transpose(1,2): GPU/CPU dump BTHD → NPU BNSD",
        "total": len(results),
        "passed": ok,
        "skipped": skip,
        "failed": fail,
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] pass={ok} skip={skip} fail={fail} → {report_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
