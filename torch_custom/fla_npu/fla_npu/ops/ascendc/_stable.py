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
# Lowest torch whose stable runtime symbols the launcher was verified against.
# Keep in sync with STABLE_ABI_MIN_TORCH in scripts/build_wheel.py.
_MIN_TORCH = "2.7.1"
_loaded_path: str | None = None
_OP_CACHE: dict[str, object] = {}
# Cached objects for the hot path.  `torch`/`torch_npu` are plain module
# handles and the raw-stream accessor is a plain function: caching *those* is
# safe.  The stream itself is never cached -- that is what corrupted the vLLM
# run earlier, where a process-global stream pointer followed a different
# thread.
_torch = None
_torch_npu = None
_raw_stream_fn = None
# int[] argument cache: value tuple -> host int64 tensor (see _host_ints).
_INT_CACHE: dict[tuple, object] = {}
_INT_CACHE_MAX = 64

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

    global _raw_stream_fn
    torch = _modules()[0]
    if _raw_stream_fn is not None:
        return int(_raw_stream_fn(torch.npu.current_device()))
    try:
        torch_npu = _modules()[1]
        if not torch_npu:
            raise AttributeError("torch_npu is not importable")
        _raw_stream_fn = getattr(torch_npu._C, "_npu_getCurrentRawStream")
    except Exception:
        _raw_stream_fn = False  # look the slow way from now on
    if not _raw_stream_fn:
        return int(torch.npu.current_stream().npu_stream)
    return int(_raw_stream_fn(torch.npu.current_device()))


def _modules():
    """(torch, torch_npu) once imported; kept out of the per-call path."""

    global _torch, _torch_npu
    if _torch is None:
        import torch as _t

        _torch = _t
    if _torch_npu is None:
        try:
            import torch_npu as _tn

            _torch_npu = _tn
        except Exception:
            _torch_npu = False
    return _torch, _torch_npu


def load() -> None:
    """dlopen the stable library through torch (no-op when already loaded)."""

    global _loaded_path
    # Hot path: once a library is loaded, re-resolving it means an environment
    # lookup plus a filesystem stat on every single operator call (~47us
    # measured).  Only an explicitly different FLA_NPU_STABLE_LIB re-resolves.
    if _loaded_path is not None:
        requested = os.environ.get(_LIB_ENV)
        if not requested or requested == _loaded_path:
            return
    path = _lib_path()
    if _loaded_path == path:
        return
    torch = _modules()[0]
    try:
        torch.ops.load_library(path)
    except Exception as exc:  # symbol resolution happens here, not at dlopen
        # The launcher resolves aoti_torch_* at load; an older torch fails with
        # "undefined symbol" a long way from the cause, so say what is wrong.
        raise RuntimeError(
            f"fla_npu: cannot load the Stable-ABI launcher {path} against "
            f"torch {torch.__version__}. It needs torch >= {_MIN_TORCH} (the "
            f"aoti_torch_* runtime symbols it resolves were added over 2.7.x). "
            f"Original error: {exc}") from exc
    _check_build_stamp(path)
    _loaded_path = path


def _check_build_stamp(path: str) -> None:
    """Refuse a library built from different generated adapters than this glue.

    The C++ adapters and the Python wrappers are both produced from the same
    ``ops_stable_generated.inc``; when only one of the two is refreshed, the
    mismatch shows up either as a dispatcher error deep inside a call or -- when
    only a stack index moved -- as a wrong stream, which is much harder to read.
    The stamp turns that into one clear message.  A library predating the stamp
    reports ``unknown`` and is accepted, so older artifacts keep working.
    """

    try:
        from . import _stable_generated as generated

        expected = generated._GENERATED_HASH
    except Exception:
        return
    try:
        import ctypes

        lib = ctypes.CDLL(path)
        lib.fla_npu_thin_source_hash.restype = ctypes.c_char_p
        actual = lib.fla_npu_thin_source_hash().decode("utf-8", "replace")
    except Exception:
        return
    if actual in ("unknown", expected):
        return
    raise RuntimeError(
        f"{path} was built from different generated adapters "
        f"(library {actual}, Python glue {expected}). Rebuild the launcher "
        f"after re-running tools/op_stable_codegen.py --all: "
        f"python csrc_stable/build_stable.py --out {path} --no-debug-probe")


def available() -> bool:
    try:
        load()
        import torch

        return hasattr(torch.ops.fla_npu_thin, "npu_recurrent_gated_delta_rule")
    except Exception:
        return False


def causal_conv1d_launcher():
    """The internal aclnnCausalConv1d launch op, or None when it is absent."""

    if not available():
        return None
    try:
        from . import _stable_generated as generated

        return getattr(generated, "_causal_conv1d_launch", None)
    except Exception:
        return None


# The conv1d family is three public APIs over one aclnn ABI, so its "stable
# backend" is the *launch* rather than a separate wrapper: the shared
# implementation is re-exported here (same function object, so the Python
# surface cannot drift), and the launch inside it goes through the internal op
# above whenever the launcher is loaded.  See
# docs/architecture/stable-abi-migration.md.
PASSTHROUGH_OPS = ("npu_causal_conv1d", "npu_causal_conv1d_fn",
                   "npu_causal_conv1d_update")

from ._aclnn_ctypes import (  # noqa: E402  (documented above)
    npu_causal_conv1d,
    npu_causal_conv1d_fn,
    npu_causal_conv1d_update,
)


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
    stable conversions).

    Decode-time calls reuse the same length list over and over (a batch of
    identical sequences), and building a tensor costs ~18us, so the result is
    cached by value.  Only list/tuple inputs are cached: a tensor is passed
    through, and anything else is converted without caching.
    """

    if values is None:
        return None
    import torch

    if not isinstance(values, (list, tuple)):
        return torch.tensor(list(values), dtype=torch.int64, device="cpu")
    key = tuple(values)
    cached = _INT_CACHE.get(key)
    if cached is not None:
        return cached
    tensor = torch.tensor(list(values), dtype=torch.int64, device="cpu")
    if len(_INT_CACHE) >= _INT_CACHE_MAX:
        _INT_CACHE.clear()
    _INT_CACHE[key] = tensor
    return tensor


def _char_code(op_name: str, argument: str, value):
    """Map a string argument to the int code the stable schema carries."""

    from . import _stable_generated as generated

    table = generated._ENUM[op_name][argument]
    if value is None:
        return 0
    return table[value] if isinstance(value, str) else value


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
    # Absent optional outputs already arrive as None: the C++ side packs
    # nullopt based on the same `when` rules, so re-evaluating them here (with
    # C++ syntax!) is redundant and fragile.
    return tuple(result) if len(result) > 1 else result[0]


try:  # generated wrappers (optional: present when the codegen step has run)
    from ._stable_generated import *  # noqa: F401,F403
    from . import _stable_generated
except Exception:  # pragma: no cover - generated module is optional
    _stable_generated = None


# ---------------------------------------------------------------------------
# Hand-written wrappers
# ---------------------------------------------------------------------------
#
# Operators whose adapter lives in csrc_stable/src/stable_<op>.cpp get their
# wrapper here rather than in _stable_generated.py: the wrapper carries a real
# signature (so a positional call does no argument binding at run time), maps
# the public argument shape onto the schema's, and nothing else.  Validation
# stays with the operator: an illegal input either reaches aclnn and comes back
# as a status, or is caught by the C++ adapter.  FLA_NPU_THIN_VALIDATE=1 routes
# such a call through the ctypes reference instead, which validates in Python
# and reports a precise message.
#
# These wrappers are appended after the generated import so a hand-written one
# always wins, which is what makes migrating an operator a one-file change on
# each side.


def npu_fast_gelu_custom(self):
    """GELU with the operator's own approximation; mirrors the ctypes shape."""

    return _op("npu_fast_gelu_custom")(self, _current_stream_ptr())


def npu_fast_gelu_custom_backward(grad, self):
    """Backward of :func:`npu_fast_gelu_custom`."""

    return _op("npu_fast_gelu_custom_backward")(
        grad, self, _current_stream_ptr())


def npu_kda_gate_cumsum(g, chunk_size, *, A_log=None, dt_bias=None,
                        cu_seqlens=None, use_gate_in_kernel=False,
                        safe_gate=False, lower_bound=None):
    """KDA gate with the log-cumsum folded in.

    The schema can only carry real values, so the optional-argument defaults the
    ctypes reference applies are applied here too (`lower_bound` defaults to
    -5.0 there, and passing None straight through is not representable).
    """

    return _op("npu_kda_gate_cumsum")(
        g,
        A_log,
        dt_bias,
        _host_ints(cu_seqlens),
        chunk_size,
        False if use_gate_in_kernel is None else bool(use_gate_in_kernel),
        False if safe_gate is None else bool(safe_gate),
        -5.0 if lower_bound is None else float(lower_bound),
        _current_stream_ptr(),
    )


def npu_chunk_kda_bwd_intra(q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg, *,
                            cu_seqlens=None, chunk_indices=None, chunk_size=64,
                            safe_gate=True, layout="BSND"):
    """Safe-gate KDA intra-chunk backward.

    BNSD is the native layout; every other combination (BSND, varlen,
    non-default chunk size, safe_gate off) is handled by the ctypes reference,
    which converts into the native form before calling the same kernel.  That
    domain split is a property of this operator, not of the backend, so it lives
    here rather than in the adapter.
    """

    layout = str(layout)
    if not (layout == "BNSD" and cu_seqlens is None and chunk_indices is None
            and int(chunk_size) == 64 and bool(safe_gate)):
        from fla_npu.ops.ascendc import _aclnn_ctypes as _ct

        return _ct.npu_chunk_kda_bwd_intra(
            q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            chunk_size=chunk_size, safe_gate=safe_gate, layout=layout)
    return _op("npu_chunk_kda_bwd_intra")(
        q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg,
        _host_ints(cu_seqlens),
        _host_ints(chunk_indices),
        chunk_size,
        safe_gate,
        _char_code("npu_chunk_kda_bwd_intra", "layout", layout),
        _current_stream_ptr(),
    )


def npu_chunk_bwd_dv_local(q, k, d_o, g, scale, chunk_size, *, g_gamma=None,
                           A=None, cu_seqlens=None, chunk_indices=None):
    """Local dv contribution of the chunked GDN backward."""

    return _op("npu_chunk_bwd_dv_local")(
        q, k, d_o, g, g_gamma, A,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        scale, chunk_size, _current_stream_ptr(),
    )


def npu_chunk_local_cumsum(g, chunk_size, *, cu_seqlens=None,
                           chunk_indices_out=None, reverse=False, scale=1.0,
                           head_first=True, output_dtype="float32"):
    """Per-chunk cumulative sum of ``g``."""

    return _op("npu_chunk_local_cumsum")(
        g,
        _host_ints(cu_seqlens),
        _host_ints(chunk_indices_out),
        chunk_size,
        reverse,
        scale,
        head_first,
        _char_code("npu_chunk_local_cumsum", "output_dtype", output_dtype),
        _current_stream_ptr(),
    )


def npu_chunk_scaled_dot_kkt(k, g, beta, *, cu_seqlens=None,
                             chunk_indices=None, chunk_size=64):
    """Chunked scaled dot product used to build the WY representation."""

    return _op("npu_chunk_scaled_dot_kkt")(
        k, g, beta,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        chunk_size, _current_stream_ptr(),
    )


def npu_chunk_bwd_dqkwg(q, k, v, g, h, dox, dh, dv, chunk_size, *,
                        cu_seqlens=None, chunk_indices=None, w=None,
                        g_gamma=None, scale=None, use_exp2=None,
                        transpose_state_layout=None):
    """dq / dk / dw / dg of one chunk.

    The three trailing flags are optional in the published signature; the
    ctypes reference supplies ``scale=1.0`` and ``False`` for the booleans, so
    do the same here rather than passing None into a scalar slot.
    """

    return _op("npu_chunk_bwd_dqkwg")(
        q, k, v, g, h, dox, dh, dv,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        w, g_gamma,
        1.0 if scale is None else float(scale),
        chunk_size,
        False if use_exp2 is None else bool(use_exp2),
        False if transpose_state_layout is None
        else bool(transpose_state_layout),
        _current_stream_ptr(),
    )


def npu_prepare_wy_repr_bwd_da(k, v, beta, A, dw, du, g, *, chunk_size,
                               cu_seqlens=None, chunk_indices=None):
    """dA only, for backends that already have the other gradients."""

    return _op("npu_prepare_wy_repr_bwd_da")(
        k, v, beta, A, dw, du, g,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        chunk_size, _current_stream_ptr(),
    )


def npu_prepare_wy_repr_bwd_full(k, v, beta, A, dA, dw, du, g, chunk_size, *,
                                 cu_seqlens=None, chunk_indices=None):
    """dk / dv / dbeta / dg, taking dA as an input."""

    return _op("npu_prepare_wy_repr_bwd_full")(
        k, v, beta, A, dA, dw, du, g,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        chunk_size, _current_stream_ptr(),
    )


def npu_prepare_wy_repr_bwd(k, v, beta, A, dw, du, g, chunk_size, *,
                            cu_seqlens=None, chunk_indices=None):
    """dk / dv / dbeta / dg; produces dA internally."""

    return _op("npu_prepare_wy_repr_bwd")(
        k, v, beta, A, dw, du, g,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        chunk_size, _current_stream_ptr(),
    )


def npu_recompute_w_u_fwd(k, v, beta, A, chunk_size, *, g=None, gk=None,
                          cu_seqlens=None, chunk_indices=None):
    """Recompute w and u for the backward pass."""

    return _op("npu_recompute_w_u_fwd")(
        k, v, beta, A, g, gk,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        chunk_size, _current_stream_ptr(),
    )
