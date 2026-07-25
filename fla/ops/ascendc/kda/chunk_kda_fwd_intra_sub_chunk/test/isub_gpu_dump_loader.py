"""Load intra_sub_chunk GPU dump .pt and convert BTHD → NPU BNSD.

Dump (GPU FLA branch ``20260724_222750_intra-sub-chunk-gpu-dump``):
  storage layout **BTHD** for both GPU and CPU tensors in the same file:
    q/k          [B, T, H,  K]
    g/aqk/akkd*  [B, T, HV, ...]
    beta         [B, T, HV]

NPU aclnn (``npu_chunk_kda_fwd_intra_sub_chunk``):
  layout **BNSD**:
    q/k          [B, H,  T, K]
    g/aqk/akkd   [B, HV, T, ...]
    beta         [B, HV, T]

* ``aqk_cpu`` / ``akkd_cpu`` are also stored as BTHD in ``001_*.pt``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

OP_NAME = "chunk_kda_fwd_intra_sub_chunk"
OP_NAME_CPU = "chunk_kda_fwd_intra_sub_chunk_cpu"

# Rank-1/2 tensors that need T↔H(V) swap when ndim >= 3.
_BTHD_NAMES = frozenset({
    "q", "k", "g", "beta", "aqk", "akkd", "aqk_cpu", "akkd_cpu",
})


def bthd_to_bnsd(t: torch.Tensor) -> torch.Tensor:
    """[B, T, H, ...] -> [B, H, T, ...] (also covers beta [B,T,HV]→[B,HV,T])."""
    if not isinstance(t, torch.Tensor) or t.ndim < 3:
        return t
    return t.transpose(1, 2).contiguous()


def to_npu_tensor(name: str, t: Any) -> Any:
    if not isinstance(t, torch.Tensor):
        return t
    if name in _BTHD_NAMES:
        return bthd_to_bnsd(t.detach().cpu())
    return t.detach().cpu()


def to_npu_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not mapping:
        return {}
    return {k: to_npu_tensor(k, v) for k, v in mapping.items() if v is not None}


def chunk_indices_flat(chunk_indices: Any) -> list[int] | None:
    if chunk_indices is None:
        return None
    if isinstance(chunk_indices, torch.Tensor):
        return [int(x) for x in chunk_indices.detach().cpu().reshape(-1).tolist()]
    if isinstance(chunk_indices, (list, tuple)):
        return [int(x) for x in chunk_indices]
    return None


def prepare_pairwise_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    out: list[int] = []
    for seq_id in range(len(cu_seqlens) - 1):
        seg = int(cu_seqlens[seq_id + 1]) - int(cu_seqlens[seq_id])
        n_chunks = (seg + chunk_size - 1) // chunk_size
        for local in range(n_chunks):
            out.extend([seq_id, local])
    return out


def load_case_meta(case_dir: str | Path) -> dict[str, Any]:
    path = Path(case_dir) / "case_meta.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_dump_for_npu(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (inputs_bnsd, meta, outputs_bnsd).

    ``outputs`` keys:
      aqk / akkd       — GPU
      aqk_cpu / akkd_cpu — CPU golden from dump (if present)
    """
    path = Path(path)
    d = torch.load(path, map_location="cpu", weights_only=False)
    op = str(d.get("op") or "")
    meta = dict(d.get("meta") or {})
    meta["op"] = op
    meta.setdefault("dump_layout", (d.get("layout") or {}).get("storage", "BTHD"))

    raw_in = d.get("inputs") or {}
    raw_out = d.get("outputs") or {}

    # 002_*_cpu.pt stores CPU under outputs.aqk/akkd — normalize.
    if op == OP_NAME_CPU:
        if "aqk_cpu" not in raw_out and "aqk" in raw_out:
            raw_out = {
                "aqk_cpu": raw_out["aqk"],
                "akkd_cpu": raw_out["akkd"],
            }

    inputs = to_npu_mapping(raw_in)
    outputs = to_npu_mapping(raw_out)

    # Prefer meta.cu_seqlens list; fall back to tensor in inputs.
    if meta.get("cu_seqlens") is None and isinstance(raw_in.get("cu_seqlens"), torch.Tensor):
        meta["cu_seqlens"] = [int(x) for x in raw_in["cu_seqlens"].flatten().tolist()]
    if "chunk_size" not in meta and raw_in.get("chunk_size") is not None:
        meta["chunk_size"] = int(raw_in["chunk_size"])
    if "scale" not in meta and raw_in.get("scale") is not None:
        meta["scale"] = float(raw_in["scale"])

    idx = chunk_indices_flat(raw_in.get("chunk_indices"))
    if idx is not None:
        meta["chunk_indices"] = idx

    return inputs, meta, outputs


def resolve_seq_for_npu(
    meta: dict[str, Any],
    case_meta: dict[str, Any] | None = None,
) -> tuple[Optional[list[int]], Optional[list[int]], int, float]:
    case_meta = case_meta or {}
    cu = meta.get("cu_seqlens")
    if cu is None:
        cu = case_meta.get("cu_seqlens")
    if cu is not None and len(cu) == 0:
        cu = None
    if cu is not None:
        cu = [int(x) for x in cu]

    chunk_size = int(meta.get("chunk_size") or case_meta.get("chunk_size") or 64)
    scale = meta.get("scale")
    if scale is None:
        scale = case_meta.get("scale")
    if scale is None:
        # last resort: 1/sqrt(K) from shapes is caller responsibility
        scale = 0.0
    scale = float(scale)

    idx = meta.get("chunk_indices")
    if idx is None:
        idx = case_meta.get("chunk_indices")
    idx = chunk_indices_flat(idx)
    if cu is not None and idx is None:
        idx = prepare_pairwise_chunk_indices(cu, chunk_size)
    return cu, idx, chunk_size, scale


def find_isub_dump_pt(case_dir: str | Path) -> Path:
    """Prefer 001_chunk_kda_fwd_intra_sub_chunk.pt (has GPU+CPU)."""
    case_dir = Path(case_dir)
    preferred = case_dir / f"001_{OP_NAME}.pt"
    if preferred.is_file():
        return preferred
    cands = sorted(case_dir.glob(f"*_{OP_NAME}.pt"))
    if not cands:
        raise FileNotFoundError(f"no *_{OP_NAME}.pt under {case_dir}")
    return cands[0]


def list_case_dirs(dump_root: str | Path) -> list[Path]:
    root = Path(dump_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dump root not found: {root}")
    dirs: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        try:
            find_isub_dump_pt(p)
        except FileNotFoundError:
            continue
        dirs.append(p)
    return dirs


def detect_dump_layout(meta: dict[str, Any], sample: torch.Tensor | None) -> str:
    layout = str(meta.get("dump_layout") or meta.get("gpu_layout") or "").upper()
    if layout in ("BTHD", "BSND"):
        return "BTHD"
    if layout in ("BNSD", "BHSD"):
        return "BNSD"
    # Heuristic: if q looks like [B,H,T,K] with H<<T typical, still ambiguous;
    # default BTHD (GPU dump convention).
    del sample
    return "BTHD"
