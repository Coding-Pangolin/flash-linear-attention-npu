"""Shared helpers for GDN GPU dump dual benchmark tests."""
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
) -> None:
    print(f"  [{tensor_name}] ct.dual(npu, cpu_fp64, gpu)", flush=True)
    ct.dual(npu_out.cpu(), fp64_golden, gpu_bench, level=level)

    if not enable_viz:
        return
    if viz_dir is None:
        return

    out_dir = Path(viz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_kwargs: dict = {
        "out_dir": str(out_dir),
        "name": tensor_name,
        "bench": gpu_bench.cpu(),
    }
    if sample_count is not None and sample_count > 0:
        viz_kwargs["sample_count"] = int(sample_count)

    print(
        f"  [{tensor_name}] ct.viz(npu, cpu_fp64, bench=gpu, sample_count={sample_count})",
        flush=True,
    )
    ct.viz(npu_out.cpu(), fp64_golden, **viz_kwargs)
