"""Shared helpers for GDN CPU dual benchmark from gpu/cases.json."""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import ct
import torch

from gdn_case_utils import generate_cu_seqlens, load_cases, parse_dtype

REPO = Path(__file__).resolve().parents[4]
DEFAULT_CASES_JSON = Path(__file__).resolve().parent / "cases.json"

DEFAULT_GPU_UNSUPPORTED_CASES = [
    "gva_fix_3",
    "gva_var_2",
    "gva_var_3",
    "gva_var_5",
    "gva_var_6",
    "phase_1_fix_11",
    "phase_1_fix_12",
    "phase_1_var_5",
    "phase_1_var_6",
]


def round_to_elem_dtype(x: torch.Tensor, elem_dtype: torch.dtype) -> torch.Tensor:
    """Round matmul operands to the NPU element dtype, but keep fp32 accumulation."""
    if elem_dtype == torch.float32:
        return x.float()
    return x.to(elem_dtype).float()


def matmul_npu_aligned(
    a: torch.Tensor,
    b: torch.Tensor,
    elem_dtype: torch.dtype,
) -> torch.Tensor:
    return round_to_elem_dtype(a, elem_dtype) @ round_to_elem_dtype(b, elem_dtype)

SMOKE_CASES: list[dict[str, Any]] = [
    {
        "name": "smoke_gva_fix",
        "description": "CPU dual smoke: GVA fixed (aligned with test.py)",
        "B": 1,
        "query_head": 2,
        "value_head": 4,
        "T": 256,
        "Kdim": 128,
        "Vdim": 256,
        "chunk_size": 64,
        "dtype": "fp16",
        "gtype": "fp32",
        "gate_function": "randn",
        "varlen": False,
    },
    {
        "name": "smoke_gva_var",
        "description": "CPU dual smoke: GVA varlen mean_len=5 T=512 cs=64",
        "B": 1,
        "query_head": 2,
        "value_head": 4,
        "T": 512,
        "Kdim": 128,
        "Vdim": 256,
        "chunk_size": 64,
        "mean_len": 5,
        "dtype": "fp16",
        "gtype": "fp32",
        "gate_function": "randn",
        "varlen": True,
    },
    {
        "name": "smoke_phase1_fix",
        "description": "CPU dual smoke: HK==HV fixed B=2 T=256 cs=64",
        "B": 2,
        "query_head": 8,
        "value_head": 8,
        "T": 256,
        "Kdim": 128,
        "Vdim": 128,
        "chunk_size": 64,
        "dtype": "fp16",
        "gtype": "fp32",
        "gate_function": "randn",
        "varlen": False,
    },
]


# bwd_dhu smoke：与 torch_custom/fla_npu/test/test_npu_bwd_dhu_gva.py 对齐
BWD_DHU_SMOKE_CASES: list[dict[str, Any]] = [
    {
        "name": "smoke_varlen_t256_v256",
        "B": 1, "query_head": 16, "value_head": 32, "T": 256,
        "Kdim": 128, "Vdim": 256, "chunk_size": 64, "mean_len": 5,
        "dtype": "bf16", "gtype": "fp32", "varlen": True,
    },
    {
        "name": "smoke_fixed_t4096_v256",
        "B": 1, "query_head": 16, "value_head": 32, "T": 4096,
        "Kdim": 128, "Vdim": 256, "chunk_size": 64,
        "dtype": "bf16", "gtype": "fp32", "varlen": False,
    },
]


def generate_cu_seqlens_for_case(case: dict[str, Any]) -> torch.LongTensor | None:
    if not case.get("varlen"):
        return None
    T = int(case["T"])
    chunk_size = int(case.get("chunk_size", 64))
    cu_seqlens_len = int(case["mean_len"])
    batchsize = cu_seqlens_len - 1
    if batchsize <= 1:
        return torch.tensor([0, T], dtype=torch.long)
    seg_avg = (T + batchsize - 1) // batchsize
    seg_max = max(chunk_size, seg_avg, min(128, chunk_size * 2))
    return generate_cu_seqlens(
        cu_seqlens_len,
        T,
        seg_min=chunk_size,
        seg_max=seg_max,
    )


def prepare_chunk_indices_list(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    indices: list[int] = []
    for i in range(len(cu_seqlens) - 1):
        length = cu_seqlens[i + 1] - cu_seqlens[i]
        if length <= 0:
            continue
        num_chunks = (length + chunk_size - 1) // chunk_size
        for chunk_id in range(num_chunks):
            indices.extend([i, chunk_id])
    return indices


def chunk_local_cumsum_cpu(
    g_bth: torch.Tensor,
    cu_list: list[int] | None,
    chunk_size: int,
) -> torch.Tensor:
    g = g_bth.float().clone()
    if cu_list is None:
        bos, eos = 0, g.shape[1]
        for chunk_start in range(0, eos - bos, chunk_size):
            s = bos + chunk_start
            e = min(s + chunk_size, eos)
            g[:, s:e, :] = g[:, s:e, :].cumsum(dim=0)
        return g
    for i in range(len(cu_list) - 1):
        bos, eos = cu_list[i], cu_list[i + 1]
        seg_len = eos - bos
        for chunk_start in range(0, seg_len, chunk_size):
            s = bos + chunk_start
            e = min(s + chunk_size, eos)
            g[:, s:e, :] = g[:, s:e, :].cumsum(dim=0)
    return g


def resolve_cases(
    *,
    cases_json: Path,
    case_names: list[str] | None,
    smoke: bool,
    smoke_cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if smoke:
        return list(smoke_cases if smoke_cases is not None else SMOKE_CASES)
    names = case_names or DEFAULT_GPU_UNSUPPORTED_CASES
    by_name = {c["name"]: c for c in load_cases(cases_json)}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"unknown case(s): {', '.join(missing)}")
    return [by_name[n] for n in names]


def default_out_dir(op: str, *, smoke: bool) -> Path:
    tag = "smoke" if smoke else "casesjson"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO / "fla/ops/ascendc/gdn/dual_benchmark_logs" / op / f"cpu_dual_{tag}" / ts


def dual_check(
    name: str,
    npu_out: torch.Tensor,
    ref_fp64: torch.Tensor,
    ref_npu: torch.Tensor,
    *,
    level: str = "L1",
) -> tuple[bool, dict[str, Any]]:
    print(f"================== {name} (dual: fp64 gt / npu-aligned bench) ==================", flush=True)
    result = ct.dual(
        npu_out.cpu().float(),
        ref_fp64.cpu().float(),
        ref_npu.cpu().float(),
        level=level,
    )
    ok = bool(result.get("success"))
    print(
        f"[{name}] dual {'PASS' if ok else 'FAIL'}: "
        f"checks={result.get('checks')} ratios={result.get('ratios')}",
        flush=True,
    )
    return ok, result


def dual_then_viz_cpu(
    tensor_name: str,
    npu_out: torch.Tensor,
    fp64_golden: torch.Tensor,
    npu_bench: torch.Tensor,
    *,
    case_name: str,
    viz_dir: Path | None,
    sample_count: int,
    enable_viz: bool,
    level: str = "L1",
    spatial: bool = False,
) -> tuple[bool, dict[str, Any]]:
    ok, result = dual_check(tensor_name, npu_out, fp64_golden, npu_bench, level=level)
    if enable_viz and viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
        ct.viz(
            npu_out.cpu().float(),
            fp64_golden.cpu().float(),
            bench=npu_bench.cpu().float(),
            out_dir=str(viz_dir),
            name=f"{case_name}_{tensor_name}_npu_vs_fp64",
            diff_thd=0.001,
            spatial=spatial,
            sample_count=sample_count,
        )
    return ok, result


def write_case_report(out_dir: Path, record: dict[str, Any]) -> None:
    case_log = out_dir / f"{record['case']}.json"
    case_log.write_text(json.dumps(record, indent=2), encoding="utf-8")


def write_batch_report(
    out_dir: Path,
    *,
    op: str,
    results: list[dict[str, Any]],
    device_id: int,
    cases_json: Path,
    dual_level: str,
    smoke: bool,
) -> Path:
    report = {
        "op": op,
        "benchmark": "cpu_dual",
        "smoke": smoke,
        "timestamp": out_dir.name,
        "device_id": device_id,
        "cases_json": str(cases_json),
        "dual_level": dual_level,
        "summary": {
            "total": len(results),
            "pass": sum(1 for r in results if r["status"] == "pass"),
            "fail": sum(1 for r in results if r["status"] == "fail"),
            "error": sum(1 for r in results if r["status"] == "error"),
        },
        "results": results,
    }
    report_path = out_dir / "cpu_dual_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path
