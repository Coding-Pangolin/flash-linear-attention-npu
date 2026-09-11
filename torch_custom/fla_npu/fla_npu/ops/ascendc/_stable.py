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

# The stable value conversions have no std::string support, so string enum
# arguments travel as int codes.
_LAYOUT_CODES = {"BSND": 0, "TND": 1}


def _lib_path() -> str:
    path = os.environ.get(_LIB_ENV)
    if path:
        return path
    # Wheels that ship the ABI-free launcher place it next to this module.
    bundled = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "libfla_npu_thin.so")
    if os.path.exists(bundled):
        return bundled
    raise RuntimeError(
        f"{_LIB_ENV} is not set and no bundled libfla_npu_thin.so was found")


def _current_stream_ptr() -> int:
    """Raw NPU stream of the calling thread (same guarded accessor as _thin)."""

    import torch

    try:
        import torch_npu

        raw_stream = getattr(torch_npu._C, "_npu_getCurrentRawStream", None)
        if raw_stream is not None:
            return int(raw_stream(torch.npu.current_device()))
    except Exception:
        pass
    return int(torch.npu.current_stream().npu_stream)


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

    stream = _current_stream_ptr()
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


def npu_recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    *,
    cu_seqlens=None,
    ssm_state_indices=None,
    A_log=None,
    dt_bias=None,
    num_accepted_tokens=None,
    layout="BSND",
    scale=None,
    output_final_state=False,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    state_v_first=False,
):
    """Recurrent KDA forward via the Stable-ABI launcher.

    ``layout`` is mapped to an int code (BSND=0, TND=1) because the stable
    argument conversions do not carry strings.  Mutates ``initial_state`` in
    place when ``inplace_final_state`` is true, exactly like the ctypes path.
    """

    import torch
    import torch_npu

    layout_code = _LAYOUT_CODES.get(str(layout))
    if layout_code is None:
        raise RuntimeError(f"npu_recurrent_kda: unknown layout {layout!r}")

    if not inplace_final_state:
        # ctypes drives the same kernel with a scratch state and returns it,
        # leaving the caller's tensor untouched.  The stable launcher only
        # exposes the inplace form (handing the caller's handle back as a second
        # output trips over shared ownership), so build the scratch here -- which
        # is also what keeps the mutation contract honest: the caller's tensor is
        # genuinely not written, and the dispatch layer's MUTATION_FLAGS entry
        # already skips the version bump for this case.
        if initial_state is None:
            raise RuntimeError(
                "npu_recurrent_kda: inplace_final_state=False requires "
                "initial_state (no shape to build the scratch from)")
        scratch = torch.empty_like(initial_state)
        out, _ = npu_recurrent_kda(
            q, k, v, g, beta, scratch, cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices, A_log=A_log, dt_bias=dt_bias,
            num_accepted_tokens=num_accepted_tokens, layout=layout, scale=scale,
            output_final_state=False, inplace_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval, safe_gate=safe_gate,
            lower_bound=lower_bound, state_v_first=state_v_first)
        return out, (scratch if output_final_state else None)

    scale_value = (128.0 ** -0.5) if scale is None else float(scale)
    lower = -5.0 if lower_bound is None else float(lower_bound)
    stream = _current_stream_ptr()
    out, final_state = _op("npu_recurrent_kda")(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        A_log,
        dt_bias,
        num_accepted_tokens,
        layout_code,
        scale_value,
        bool(output_final_state),
        bool(inplace_final_state),
        bool(use_qk_l2norm_in_kernel),
        bool(use_gate_in_kernel),
        bool(use_beta_sigmoid_in_kernel),
        bool(allow_neg_eigval),
        bool(safe_gate),
        lower,
        bool(state_v_first),
        stream,
    )
    if not output_final_state:
        return out, None
    if inplace_final_state and final_state is None:
        # The launcher returns the inplace result implicitly (the kernel writes
        # the caller's tensor); the stable ABI cannot hand that handle back as a
        # second output without a double ownership release, so mirror ctypes
        # here, which also returns the caller's object.
        final_state = initial_state
    return out, final_state


# ---------------------------------------------------------------------------
# Generic plumbing for the generated wrappers.
# ---------------------------------------------------------------------------
def _host_ints(values):
    """int[] arguments travel as host int64 tensors (no list support in the
    stable conversions)."""

    if values is None:
        return None
    import torch

    return torch.tensor(list(values), dtype=torch.int64, device="cpu")


def _call(name: str, values: dict):
    """Invoke a generated stable op from its user-facing argument values."""

    from . import _stable_generated as generated

    schema = generated._SIG[name]
    enums = generated._ENUM.get(name, {})
    args = []
    for arg_name, kind in schema:
        value = values.get(arg_name)
        if kind == "int_array":
            value = _host_ints(value)
        elif kind == "char_ptr":
            table = enums[arg_name]
            if value is None:
                value = 0
            elif isinstance(value, str):
                value = table[value]
        args.append(value)
    args.append(_current_stream_ptr())
    result = _op(name)(*args)
    if not isinstance(result, tuple):
        result = (result,)
    out = []
    for index, when in generated._RET[name]:
        keep = True if when is None else bool(eval(when, {}, values))
        out.append(result[index] if keep else None)
    return tuple(out) if len(out) > 1 else out[0]


try:  # generated wrappers (optional: present when the codegen step has run)
    from ._stable_generated import *  # noqa: F401,F403
    from . import _stable_generated
except Exception:  # pragma: no cover - generated module is optional
    _stable_generated = None
