"""Prepare GPU KDA dump tensors for NPU chunk_kda_fwd and CPU fp64 reference."""
from __future__ import annotations

from typing import Any

import torch

RCP_LN2 = 1.4426950408889634


def kda_gate_cumsum_reference(
    g: torch.Tensor,
    chunk_size: int,
    *,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """Match GPU safe_gate path and NPU gk semantics (log2 cumulative gate)."""
    g_float = g.to(torch.float32)
    if use_gate_in_kernel:
        if not safe_gate:
            raise ValueError("adapter only covers safe_gate raw-g path for GPU dumps")
        x = g_float
        if dt_bias is not None:
            bias = dt_bias.reshape(g.shape[-2], g.shape[-1]).to(torch.float32)
            if g.dim() == 4:
                x = x + bias[None, None, :, :]
            else:
                x = x + bias[None, :, :]
        a = torch.exp(A_log.to(torch.float32))
        if g.dim() == 4:
            x = x * a[None, None, :, None]
        else:
            x = x * a[None, :, None]
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


def prepare_beta(
    beta_raw: torch.Tensor,
    *,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not use_beta_sigmoid_in_kernel:
        out = beta_raw
    else:
        scale = 2.0 if allow_neg_eigval else 1.0
        out = torch.sigmoid(beta_raw.float()) * scale
    if dtype is not None and out.is_floating_point():
        out = out.to(dtype)
    return out


def prepare_gk(
    inputs: dict[str, Any],
    meta: dict[str, Any],
    chunk_size: int,
) -> torch.Tensor:
    g = inputs["g"]
    return kda_gate_cumsum_reference(
        g,
        chunk_size,
        A_log=inputs.get("A_log"),
        dt_bias=inputs.get("dt_bias"),
        use_gate_in_kernel=bool(meta.get("use_gate_in_kernel")),
        safe_gate=bool(meta.get("safe_gate")),
        lower_bound=float(meta.get("lower_bound") or -5.0),
    )


def resolve_scale(
    inputs: dict[str, Any],
    meta: dict[str, Any],
    case_meta: dict[str, Any] | None,
    q: torch.Tensor,
) -> float:
    for src in (inputs.get("scale"), meta.get("scale"), (case_meta or {}).get("scale")):
        if src is not None:
            return float(src)
    return float(q.shape[-1] ** -0.5)


def is_supported_by_pr152(v: torch.Tensor) -> tuple[bool, str | None]:
    """PR #152 excludes Vdim=256 dedicated templates."""
    if v.ndim == 4 and v.shape[-1] == 256:
        return False, "Vdim=256 not supported by PR #152 KDA forward"
    return True, None
