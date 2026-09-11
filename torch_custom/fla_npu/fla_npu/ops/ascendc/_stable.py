"""Stable-ABI thin backend (Phase 1).

Loads ``libfla_npu_thin.so`` (a plain shared object registered through
``STABLE_TORCH_LIBRARY``) and exposes the same Python call shape as the ctypes
and pybind backends.  This module contains no ABI-sensitive code: the only
tensor objects crossing the boundary are handled by torch's own dispatcher.

The mutation contract (version bump / requires_grad rejection) is *not*
provided by the dispatcher for these ops -- measured in Phase 1 -- so the
dispatch layer keeps applying ``_wrap_mutable_direct_op`` on top of whatever
backend it picks.
"""

from __future__ import annotations

import os


_LIB_ENV = "FLA_NPU_STABLE_LIB"
_loaded_path: str | None = None
_OP_CACHE: dict[str, object] = {}


def _lib_path() -> str:
    path = os.environ.get(_LIB_ENV)
    if not path:
        raise RuntimeError(
            f"{_LIB_ENV} is not set; point it at the built libfla_npu_thin.so")
    return path


def load() -> None:
    """dlopen the stable library through torch (no-op when already loaded)."""

    global _loaded_path
    path = _lib_path()
    if _loaded_path == path:
        return
    import torch

    torch.ops.load_library(path)
    _loaded_path = path


def available() -> bool:
    try:
        load()
        import torch

        return hasattr(torch.ops.fla_npu_thin, "npu_recurrent_gated_delta_rule")
    except Exception:
        return False


def _op(name: str):
    """Cached torch.ops handle: the attribute chain is not free per call."""

    op = _OP_CACHE.get(name)
    if op is None:
        load()
        import torch

        op = getattr(torch.ops.fla_npu_thin, name)
        _OP_CACHE[name] = op
    return op


def stream_probe(device_index: int) -> tuple[int, int]:
    """Return (raw backend stream ptr, stable Stream::id()) for comparison."""

    load()
    import torch

    raw, stream_id = _op("_stream_probe")(int(device_index))
    return int(raw), int(stream_id)


def npu_recurrent_gated_delta_rule(
    query,
    key,
    value,
    state,
    *,
    beta,
    scale=1.0,
    actual_seq_lengths,
    ssm_state_indices,
    num_accepted_tokens=None,
    g=None,
    gk=None,
):
    """Recurrent GDN forward; mutates ``state`` in place like the other paths."""

    load()
    import torch
    import torch_npu

    stream = int(
        torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device()))
    return _op("npu_recurrent_gated_delta_rule")(
        query,
        key,
        value,
        state,
        beta,
        actual_seq_lengths,
        ssm_state_indices,
        num_accepted_tokens,
        g,
        gk,
        float(scale),
        stream,
    )
