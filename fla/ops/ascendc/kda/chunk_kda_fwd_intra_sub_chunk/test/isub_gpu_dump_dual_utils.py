"""ct.dual / ct.viz helpers for intra_sub_chunk GPU dump dual."""
from __future__ import annotations

import argparse
from pathlib import Path

import ct
import torch


def add_viz_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-viz", action="store_true", help="skip ct.viz")
    parser.add_argument(
        "-sc",
        "--sample-count",
        type=int,
        default=200_000,
        help="ct.viz sample_count (default 200000)",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="viz root (default: <case_dir>/viz)",
    )


def dual_then_viz(
    tensor_name: str,
    npu_out: torch.Tensor,
    cpu_bench: torch.Tensor,
    gpu_bench: torch.Tensor,
    *,
    viz_dir: Path | str | None,
    sample_count: int | None = 200_000,
    enable_viz: bool = True,
    level: str = "L1",
) -> None:
    print(f"  [{tensor_name}] ct.dual(npu, cpu_dump, gpu_dump)", flush=True)
    ct.dual(npu_out.detach().cpu(), cpu_bench.detach().cpu(), gpu_bench.detach().cpu(), level=level)

    if not enable_viz or viz_dir is None:
        return
    out_dir = Path(viz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"out_dir": str(out_dir), "name": tensor_name}
    if sample_count is not None and sample_count > 0:
        kwargs["sample_count"] = int(sample_count)
    print(f"  [{tensor_name}] ct.viz(npu, cpu_dump, sample_count={sample_count})", flush=True)
    ct.viz(npu_out.detach().cpu(), cpu_bench.detach().cpu(), **kwargs)


def print_err_stats(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).abs()
    rel = diff / b.float().abs().clamp_min(1.0)
    stats = {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(rel.max()),
    }
    print(
        f"  [{name}] max_abs={stats['max_abs']:.6g} mean_abs={stats['mean_abs']:.6g} "
        f"max_rel={stats['max_rel']:.6g}",
        flush=True,
    )
    return stats
