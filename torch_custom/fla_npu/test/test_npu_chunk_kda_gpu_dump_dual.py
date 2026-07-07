# Copyright (c) 2026 Tianjin University, Ltd.
"""KDA chunk_kda_fwd GPU dump 三标杆：读 GPU dump → NPU 执行 → ct.dual(npu, fp64, gpu)。

用法:
  TEST_DEVICE_ID=6 python test_npu_chunk_kda_gpu_dump_dual.py /path/to/KDA_DUMP
  TEST_DEVICE_ID=6 python test_npu_chunk_kda_gpu_dump_dual.py /path/to/KDA_DUMP --case smoke_mha_fix
  TEST_DEVICE_ID=6 python test_npu_chunk_kda_gpu_dump_dual.py /path/to/KDA_DUMP --phase smoke

每个 case 目录需含 ``001_chunk_kda_fwd.pt``（GPU ``feat/kda-gpu-dump`` 采集）。
比对 ``o`` / ``final_state``；``gk`` 用 ``npu_kda_gate_cumsum`` 从 dump 原始 ``g`` 计算。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import ct
import fla_npu  # noqa: F401
import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover
    torch_npu = None

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.reference.chunk_kda_reference import chunk_kda_forward_reference  # noqa: E402

# 与 test_npu_chunk_kda.py 一致：不要设 allow_internal_format=False。
# 大 shape bf16 case 在 False 下 kernel 输出会全 NaN（日志里会有 internal format warning）。

OP_PT = "001_chunk_kda_fwd.pt"
RCP_LN2 = 1.4426950408889634


def _device() -> torch.device:
    dev_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device(f"npu:{dev_id}")
    return torch.device("cpu")


def _kda_gate_cumsum_reference(
    g: torch.Tensor,
    chunk_size: int,
    *,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """CPU fp64 参考用，与 test_npu_chunk_kda.py 一致。"""
    g_float = g.to(torch.float32)
    if use_gate_in_kernel:
        if not safe_gate:
            raise ValueError("only safe_gate raw-g path is supported for GPU dumps")
        x = g_float
        if dt_bias is not None:
            bias = dt_bias.reshape(g.shape[-2], g.shape[-1]).to(torch.float32)
            x = x + bias[None, None, :, :] if g.dim() == 4 else x + bias[None, :, :]
        a = torch.exp(A_log.to(torch.float32))
        x = x * a[None, None, :, None] if g.dim() == 4 else x * a[None, :, None]
        gate = float(lower_bound) * torch.sigmoid(x)
    else:
        gate = g_float

    out = torch.empty_like(gate, dtype=torch.float32)
    if g.dim() == 4:
        for b in range(g.shape[0]):
            for start in range(0, g.shape[1], chunk_size):
                end = min(start + chunk_size, g.shape[1])
                out[b, start:end] = torch.cumsum(gate[b, start:end] * RCP_LN2, dim=0)
    else:
        for start in range(0, g.shape[0], chunk_size):
            end = min(start + chunk_size, g.shape[0])
            out[start:end] = torch.cumsum(gate[start:end] * RCP_LN2, dim=0)
    return out


def _load_dump(pt_path: Path) -> tuple[dict, dict, dict]:
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    inputs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in (data.get("inputs") or {}).items()}
    outputs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in (data.get("outputs") or {}).items()}
    meta = dict(data.get("meta") or {})
    return inputs, meta, outputs


def _merge_meta(meta: dict, case_meta: dict) -> dict:
    merged = dict(case_meta)
    merged.update(meta)
    return merged


def _int_list(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().tolist()]
    return [int(x) for x in value]


def _cu_list(meta: dict, case_meta: dict) -> list[int] | None:
    for src in (meta, case_meta):
        cu = src.get("cu_seqlens")
        if cu is None:
            continue
        if isinstance(cu, torch.Tensor):
            cu = [int(x) for x in cu.tolist()]
        else:
            cu = [int(x) for x in cu]
        if cu:
            return cu
    return None


def _prepare_beta(beta_raw: torch.Tensor, meta: dict, *, dtype: torch.dtype) -> torch.Tensor:
    if not meta.get("use_beta_sigmoid_in_kernel", True):
        out = beta_raw.float()
    else:
        scale = 2.0 if meta.get("allow_neg_eigval") else 1.0
        out = torch.sigmoid(beta_raw.float()) * scale
    return out.to(dtype)


def _finite_stats(t: torch.Tensor) -> str:
    x = t.detach().float().view(-1)
    total = x.numel()
    finite = int(torch.isfinite(x).sum())
    nan = int(torch.isnan(x).sum())
    inf = int(torch.isinf(x).sum())
    return f"finite {finite}/{total} (nan={nan}, inf={inf})"


def _prepare_gk_npu(
    device: torch.device,
    inputs: dict,
    meta: dict,
    chunk_size: int,
    cu_seqlens: list[int] | None,
) -> torch.Tensor:
    g = inputs["g"].to(device)
    kwargs: dict = {}
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens
    if meta.get("use_gate_in_kernel", True):
        kwargs["A_log"] = inputs["A_log"].to(device)
        if inputs.get("dt_bias") is not None:
            kwargs["dt_bias"] = inputs["dt_bias"].to(device)
        kwargs["use_gate_in_kernel"] = True
        kwargs["safe_gate"] = bool(meta.get("safe_gate", True))
        kwargs["lower_bound"] = float(meta.get("lower_bound", -5.0))
    return torch.ops.npu.npu_kda_gate_cumsum(g, chunk_size, **kwargs)


def _prepare_gk_cpu(inputs: dict, meta: dict, chunk_size: int) -> torch.Tensor:
    return _kda_gate_cumsum_reference(
        inputs["g"],
        chunk_size,
        A_log=inputs.get("A_log"),
        dt_bias=inputs.get("dt_bias"),
        use_gate_in_kernel=bool(meta.get("use_gate_in_kernel", True)),
        safe_gate=bool(meta.get("safe_gate", True)),
        lower_bound=float(meta.get("lower_bound", -5.0)),
    )


def _dual(name: str, npu_out: torch.Tensor, ref_fp64: torch.Tensor, gpu_out: torch.Tensor, level: str = "L1") -> bool:
    print(f"  [dual] {name}", flush=True)
    result = ct.dual(
        npu_out.detach().cpu().float(),
        ref_fp64.detach().cpu().float(),
        gpu_out.detach().cpu().float(),
        level=level,
    )
    ok = bool(result.get("success"))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{name}] {tag} checks={result.get('checks')} ratios={result.get('ratios')}", flush=True)
    return ok


def _list_cases(dump_root: Path, phase: str) -> list[Path]:
    dirs = sorted(
        p for p in dump_root.iterdir()
        if p.is_dir() and (p / OP_PT).is_file()
    )
    phase = phase.strip().lower()
    if phase in ("", "all"):
        return dirs
    if phase == "smoke":
        return [d for d in dirs if d.name.startswith("smoke_")]
    if phase.startswith("prefix:"):
        prefix = phase.split(":", 1)[1]
        return [d for d in dirs if d.name.startswith(prefix)]
    raise ValueError(f"unknown phase {phase!r}")


def run_case(case_dir: Path, device: torch.device, *, level: str = "L1") -> str:
    if device.type == "cpu":
        print(f"SKIP {case_dir.name}: NPU not available", flush=True)
        return "skip"

    pt_path = case_dir / OP_PT
    case_name = case_dir.name
    inputs, meta, gpu_outputs = _load_dump(pt_path)

    case_meta_path = case_dir / "case_meta.json"
    case_meta = {}
    if case_meta_path.is_file():
        case_meta = json.loads(case_meta_path.read_text(encoding="utf-8"))
    meta = _merge_meta(meta, case_meta)

    v = inputs["v"]
    if v.shape[-1] == 256:
        print(f"SKIP {case_name}: Vdim=256 not in current NPU scope", flush=True)
        return "skip"

    q = inputs["q"].contiguous().to(device)
    k = inputs["k"].contiguous().to(device)
    v = v.contiguous().to(device)
    chunk_size = int(meta.get("chunk_size") or 64)
    scale_val = meta.get("scale", inputs.get("scale"))
    if isinstance(scale_val, torch.Tensor):
        scale_val = float(scale_val.item())
    scale = float(scale_val if scale_val is not None else (q.shape[-1] ** -0.5))
    cu_seqlens = _cu_list(meta, case_meta)
    chunk_indices = _int_list(meta.get("chunk_indices") or inputs.get("chunk_indices"))

    print(
        f"\n=== {case_name} ===\n"
        f"  pt={pt_path} B={q.shape[0]} Hk={q.shape[2]} Hv={v.shape[2]} "
        f"T={q.shape[1]} K={q.shape[3]} V={v.shape[3]} cs={chunk_size} "
        f"varlen={cu_seqlens is not None} dtype={q.dtype} "
        f"use_gate={meta.get('use_gate_in_kernel')} safe_gate={meta.get('safe_gate')}",
        flush=True,
    )

    gk = _prepare_gk_npu(device, inputs, meta, chunk_size, cu_seqlens)
    torch.npu.synchronize()
    print(f"  [npu] gk {_finite_stats(gk)} min={float(gk.float().min()):.4g} max={float(gk.float().max()):.4g}", flush=True)
    if not torch.isfinite(gk).all():
        raise RuntimeError(f"gk contains non-finite values for {case_name}")

    beta = _prepare_beta(inputs["beta"], meta, dtype=q.dtype).contiguous().to(device)
    initial_state = inputs.get("initial_state")
    if initial_state is not None:
        initial_state = initial_state.contiguous().to(device).float()

    fwd_kw: dict = {"output_final_state": True, "return_intermediate": False}
    if initial_state is not None:
        fwd_kw["initial_state"] = initial_state
    if chunk_indices is not None:
        fwd_kw["chunk_indices"] = chunk_indices
    elif cu_seqlens is not None:
        fwd_kw["cu_seqlens"] = cu_seqlens

    got = torch.ops.npu.npu_chunk_kda_fwd(q, k, v, gk, beta, scale, chunk_size, **fwd_kw)
    torch.npu.synchronize()
    o_npu, final_state_npu = got[0], got[1]
    print(f"  [npu] o {_finite_stats(o_npu)}", flush=True)
    if not torch.isfinite(o_npu).all():
        raise RuntimeError(f"NPU output o has non-finite values for {case_name}")

    gk_cpu = _prepare_gk_cpu(inputs, meta, chunk_size)
    beta_cpu = _prepare_beta(inputs["beta"], meta, dtype=torch.float32)
    cu_tensor = torch.tensor(cu_seqlens, dtype=torch.int64) if cu_seqlens else None
    ref = chunk_kda_forward_reference(
        q.detach().cpu().double(),
        k.detach().cpu().double(),
        v.detach().cpu().double(),
        gk_cpu.double(),
        beta_cpu.double(),
        scale=scale,
        chunk_size=chunk_size,
        initial_state=initial_state.detach().cpu().double() if initial_state is not None else None,
        output_final_state=True,
        cu_seqlens=cu_tensor,
    )

    ok = _dual(f"{case_name}/o", o_npu, ref.o, gpu_outputs["o"], level=level)
    gpu_fs = gpu_outputs.get("final_state")
    if gpu_fs is not None and final_state_npu is not None and final_state_npu.numel() > 0:
        ok = _dual(f"{case_name}/final_state", final_state_npu, ref.final_state, gpu_fs, level=level) and ok

    return "pass" if ok else "fail"


def main() -> int:
    p = argparse.ArgumentParser(description="KDA GPU dump dual benchmark (single-file)")
    p.add_argument("dump_root", type=Path, help="GPU dump root, e.g. /path/to/KDA_DUMP")
    p.add_argument("--case", default="", help="single case dir name")
    p.add_argument("--cases", default="", help="comma-separated case names")
    p.add_argument("--phase", default="all", help="all | smoke | prefix:<name>")
    p.add_argument("--level", default="L1", help="ct.dual level")
    args = p.parse_args()

    dump_root = args.dump_root.resolve()
    if not dump_root.is_dir():
        print(f"ERROR: dump root not found: {dump_root}", file=sys.stderr)
        return 2

    if args.case:
        case_dirs = [dump_root / args.case]
    elif args.cases.strip():
        case_dirs = [dump_root / n.strip() for n in args.cases.split(",") if n.strip()]
    else:
        case_dirs = _list_cases(dump_root, args.phase)

    device = _device()
    stats = {"pass": 0, "fail": 0, "skip": 0}
    for case_dir in case_dirs:
        if not (case_dir / OP_PT).is_file():
            print(f"ERROR: missing {case_dir / OP_PT}", file=sys.stderr)
            stats["fail"] += 1
            continue
        try:
            status = run_case(case_dir, device, level=args.level)
        except Exception as exc:
            print(f"=== {case_dir.name} ERROR ===\n{exc}", flush=True)
            stats["fail"] += 1
            continue
        stats[status] = stats.get(status, 0) + 1

    print(
        f"\nDone: pass={stats['pass']} fail={stats['fail']} skip={stats['skip']} "
        f"/ {sum(stats.values())} total",
        flush=True,
    )
    return 1 if stats["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
