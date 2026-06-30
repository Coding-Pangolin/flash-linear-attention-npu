"""Load GPU GDN dump .pt files and convert GPU layout -> NPU aclnn layout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

_BTH_TO_BHT_NAMES = frozenset({
    "q", "k", "v", "w", "u", "g", "beta", "do", "dv", "dv2", "du",
    "dq", "dk", "dw", "dg", "db", "dg2", "dk2", "v_new", "o", "A",
})
_BNTH_TO_BHNT_NAMES = frozenset({"h", "dh"})
_PASSTHROUGH = frozenset({"initial_state", "final_state", "h0", "dh0", "dht"})


def bth_to_bht(t: torch.Tensor) -> torch.Tensor:
    if t.ndim < 3:
        return t
    return t.transpose(1, 2).contiguous()


def bnth_to_bhnt(t: torch.Tensor) -> torch.Tensor:
    """[B, NT, H, K, V] -> [B, H, NT, K, V]."""
    if t.ndim != 5:
        return t
    return t.permute(0, 2, 1, 3, 4).contiguous()


def to_npu_tensor(name: str, t: Any, *, beta_fp32: bool = True) -> Any:
    if not isinstance(t, torch.Tensor):
        return t
    if name in _PASSTHROUGH:
        out = t
    elif name in _BNTH_TO_BHNT_NAMES:
        out = bnth_to_bhnt(t)
    elif name in _BTH_TO_BHT_NAMES:
        out = bth_to_bht(t)
    else:
        out = t
    out = out.detach().cpu()
    if beta_fp32 and name == "beta" and out.is_floating_point():
        out = out.float()
    return out


def to_npu_mapping(mapping: dict[str, Any] | None, *, beta_fp32: bool = True) -> dict[str, Any]:
    if not mapping:
        return {}
    return {
        k: to_npu_tensor(k, v, beta_fp32=beta_fp32)
        for k, v in mapping.items()
        if v is not None
    }


def chunk_indices_npu_list(chunk_indices: Any) -> list[int] | None:
    if chunk_indices is None:
        return None
    if isinstance(chunk_indices, torch.Tensor):
        if chunk_indices.ndim == 2 and chunk_indices.shape[-1] == 2:
            return [int(x) for x in chunk_indices.detach().cpu().reshape(-1).tolist()]
        return [int(x) for x in chunk_indices.detach().cpu().flatten().tolist()]
    if isinstance(chunk_indices, (list, tuple)):
        return [int(x) for x in chunk_indices]
    return None


def prepare_chunk_offsets_list(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    """fwd_h NPU aclnn uses cumsum chunk offsets, not pairwise chunk_indices."""
    if len(cu_seqlens) < 2:
        return [0]
    offsets = [0]
    for i in range(len(cu_seqlens) - 1):
        seg = cu_seqlens[i + 1] - cu_seqlens[i]
        offsets.append(offsets[-1] + (seg + chunk_size - 1) // chunk_size)
    return offsets


def resolve_seq_meta(
    meta: dict[str, Any],
    case_meta: dict[str, Any],
) -> tuple[list[int] | None, list[int] | None, int, float | None]:
    cu = meta.get("cu_seqlens")
    if cu is None:
        cu = case_meta.get("cu_seqlens")
    if cu is not None and len(cu) == 0:
        cu = None

    chunk_indices = meta.get("chunk_indices_npu")
    if chunk_indices is None:
        chunk_indices = meta.get("chunk_indices")
    if isinstance(chunk_indices, torch.Tensor):
        chunk_indices = chunk_indices_npu_list(chunk_indices)

    chunk_size = int(meta.get("chunk_size") or case_meta.get("chunk_size") or 64)
    scale = meta.get("scale")
    if scale is None:
        scale = case_meta.get("scale")
    return cu, chunk_indices, chunk_size, scale


def load_dump_for_npu(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (inputs_npu_layout, meta, outputs_npu_layout) from a GPU dump .pt file."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    op = str(d["op"])
    meta = dict(d.get("meta") or {})
    meta["op"] = op
    if "inputs_npu" in d:
        inputs = d["inputs_npu"]
        outputs = d.get("outputs_npu") or {}
    else:
        inputs = to_npu_mapping(d.get("inputs") or {})
        outputs = to_npu_mapping(d.get("outputs") or {})
    if "chunk_indices_npu" not in meta and "chunk_indices" in meta:
        meta["chunk_indices_npu"] = meta["chunk_indices"]
    return inputs, meta, outputs


def find_op_dump_pt(
    case_dir: str | Path,
    op_name: str,
    *,
    phase: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    case_dir = Path(case_dir)
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for p in sorted(case_dir.glob(f"*_{op_name}.pt")):
        d = torch.load(p, map_location="cpu", weights_only=False)
        candidates.append((p, d))
    if not candidates:
        raise FileNotFoundError(f"no *_{op_name}.pt under {case_dir}")
    if phase is not None:
        for p, d in candidates:
            if d.get("meta", {}).get("phase") == phase:
                return p, d
    return candidates[-1]


def load_case_meta(case_dir: str | Path) -> dict[str, Any]:
    path = Path(case_dir) / "case_meta.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_case_dirs(dump_root: str | Path) -> list[Path]:
    root = Path(dump_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dump root not found: {root}")
    dirs = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and ((p / "manifest.json").is_file() or (p / "case_meta.json").is_file()):
            dirs.append(p)
    return dirs
