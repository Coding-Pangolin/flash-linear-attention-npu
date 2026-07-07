"""Load GPU KDA dump .pt files (BTHD / BSND layout, no GDN-style transpose)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

_KDA_TENSOR_NAMES = frozenset({
    "q", "k", "v", "g", "beta", "o", "initial_state", "final_state", "A_log", "dt_bias",
})


def to_kda_tensor(name: str, value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    out = value.detach().cpu()
    if name == "beta" and out.is_floating_point():
        out = out.float()
    return out


def to_kda_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not mapping:
        return {}
    return {
        k: to_kda_tensor(k, v)
        for k, v in mapping.items()
        if v is not None
    }


def chunk_indices_list(chunk_indices: Any) -> list[int] | None:
    if chunk_indices is None:
        return None
    if isinstance(chunk_indices, torch.Tensor):
        return [int(x) for x in chunk_indices.detach().cpu().reshape(-1).tolist()]
    if isinstance(chunk_indices, (list, tuple)):
        return [int(x) for x in chunk_indices]
    return None


def cu_seqlens_list(cu_seqlens: Any) -> list[int] | None:
    if cu_seqlens is None:
        return None
    if isinstance(cu_seqlens, torch.Tensor):
        return [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    if isinstance(cu_seqlens, (list, tuple)):
        return [int(x) for x in cu_seqlens]
    return None


def resolve_seq_meta(
    meta: dict[str, Any],
    case_meta: dict[str, Any],
) -> tuple[list[int] | None, list[int] | None, int, float | None]:
    cu = cu_seqlens_list(meta.get("cu_seqlens"))
    if cu is None:
        cu = cu_seqlens_list(case_meta.get("cu_seqlens"))
    if cu is not None and len(cu) == 0:
        cu = None

    chunk_indices = chunk_indices_list(meta.get("chunk_indices_npu"))
    if chunk_indices is None:
        chunk_indices = chunk_indices_list(meta.get("chunk_indices"))

    chunk_size = int(meta.get("chunk_size") or case_meta.get("chunk_size") or 64)
    scale = meta.get("scale")
    if scale is None:
        scale = case_meta.get("scale")
    return cu, chunk_indices, chunk_size, scale


def load_dump_for_kda(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (inputs, meta, outputs) from a GPU KDA dump .pt file."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    op = str(d["op"])
    meta = dict(d.get("meta") or {})
    meta["op"] = op
    if "inputs_npu" in d:
        inputs = d["inputs_npu"]
        outputs = d.get("outputs_npu") or {}
    else:
        inputs = to_kda_mapping(d.get("inputs") or {})
        outputs = to_kda_mapping(d.get("outputs") or {})
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
