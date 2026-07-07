"""Shared helpers for KDA GPU dump dual benchmark tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import ct
import torch


def add_viz_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="skip ct.viz after ct.dual",
    )
    parser.add_argument(
        "-sc",
        "--sample-count",
        type=int,
        default=200_000,
        help="ct.viz ordered sample count for large tensors (default: 200000)",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="ct.viz output directory (default: <case_or_pt_dir>/viz)",
    )


def resolve_viz_dir(
    *,
    viz_dir: Path | None,
    pt_path: Path | None,
    case_dir: Path | None,
    default_report_dir: Path,
) -> Path | None:
    if viz_dir is not None:
        return viz_dir
    if pt_path is not None:
        return pt_path.parent / "viz"
    if case_dir is not None:
        return case_dir / "viz"
    return default_report_dir / "viz"


def _viz_file_paths(out_dir: Path, name_prefix: str) -> list[Path]:
    return [
        out_dir / f"{name_prefix}_Standard.png",
        out_dir / f"{name_prefix}_RealPart.png",
        out_dir / f"{name_prefix}_ImagPart.png",
    ]


def _tensor_finite_stats(t: torch.Tensor) -> dict[str, int | float]:
    x = t.detach().cpu().float().reshape(-1)
    total = int(x.numel())
    finite = int(torch.isfinite(x).sum().item())
    nan = int(torch.isnan(x).sum().item())
    inf = int(torch.isinf(x).sum().item())
    return {
        "total": total,
        "finite": finite,
        "nan": nan,
        "inf": inf,
        "finite_ratio": (finite / total) if total else 0.0,
    }


def _require_npu_finite(name: str, t: torch.Tensor) -> None:
    stats = _tensor_finite_stats(t)
    if stats["finite"] == 0:
        raise RuntimeError(
            f"NPU output {name} has no finite values "
            f"(nan={stats['nan']}, inf={stats['inf']}, shape={tuple(t.shape)})"
        )
    if stats["finite_ratio"] < 0.99:
        print(
            f"  [warn] {name} finite_ratio={stats['finite_ratio']:.4f} "
            f"({stats['finite']}/{stats['total']})",
            flush=True,
        )


def dual_then_viz(
    tensor_name: str,
    npu_out: torch.Tensor,
    fp64_golden: torch.Tensor,
    gpu_bench: torch.Tensor,
    *,
    viz_dir: Path | str | None,
    sample_count: int | None = 200_000,
    enable_viz: bool = True,
    level: str = "L1",
    viz_name_prefix: str | None = None,
) -> bool:
    print(f"  [{tensor_name}] ct.dual(npu, cpu_fp64, gpu)", flush=True)
    dual_result = ct.dual(npu_out.cpu(), fp64_golden, gpu_bench, level=level)
    dual_ok = bool(dual_result.get("success"))
    if not dual_ok:
        print(f"  [{tensor_name}] ct.dual FAIL", flush=True)

    if not enable_viz:
        return dual_ok
    if viz_dir is None:
        return dual_ok

    out_dir = Path(viz_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name_prefix = viz_name_prefix or tensor_name

    npu_f = npu_out.detach().cpu().float()
    fp64_f = fp64_golden.detach().cpu().float()
    gpu_f = gpu_bench.detach().cpu().float()
    if npu_f.shape != fp64_f.shape:
        print(
            f"  [{tensor_name}] ct.viz SKIPPED: shape mismatch "
            f"npu={tuple(npu_f.shape)} fp64={tuple(fp64_f.shape)}",
            flush=True,
        )
        return dual_ok

    viz_kwargs: dict = {
        "out_dir": str(out_dir),
        "name": name_prefix,
        "bench": gpu_f,
    }
    if sample_count is not None and sample_count > 0:
        viz_kwargs["sample_count"] = int(sample_count)

    print(
        f"  [{tensor_name}] ct.viz -> {out_dir} "
        f"(prefix={name_prefix}, sample_count={sample_count})",
        flush=True,
    )
    ct.viz(npu_f, fp64_f, **viz_kwargs)

    saved = [p for p in _viz_file_paths(out_dir, name_prefix) if p.is_file()]
    if saved:
        for p in saved:
            print(f"  [{tensor_name}] viz saved: {p}", flush=True)
    else:
        print(
            f"  [{tensor_name}] viz WARNING: no png under {out_dir} "
            f"(expected {name_prefix}_Standard.png; NPU all-NaN yields no plot)",
            flush=True,
        )
    return dual_ok
