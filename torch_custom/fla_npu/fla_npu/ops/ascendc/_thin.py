# Thin C++ launcher adapters for the hot FLA NPU operators.
#
# Enabled per-process with FLA_NPU_THIN_LAUNCHER=1 after the extension has been
# built (FLA_NPU_BUILD_THIN=1). Signatures mirror the ctypes wrappers in
# _aclnn_ctypes.py so the existing public API and mutation contracts are kept.
from __future__ import annotations

import os

_CURRENT_STREAM_PTR = None
_STREAM_PATCHED = False


def _ensure_stream_tracking() -> None:
    """Track torch.npu.set_stream so thin ops avoid the expensive
    torch.npu.current_stream() object construction on every call."""

    global _STREAM_PATCHED, _CURRENT_STREAM_PTR
    if _STREAM_PATCHED:
        return
    import torch

    original = torch.npu.set_stream

    def tracking_set_stream(stream):
        global _CURRENT_STREAM_PTR
        _CURRENT_STREAM_PTR = int(stream.npu_stream)
        return original(stream)

    torch.npu.set_stream = tracking_set_stream
    _STREAM_PATCHED = True


def _extension() -> "module":
    import fla_npu._C_thin as ext

    lib = os.environ.get("FLA_NPU_OP_API_LIB", "")
    if lib:
        try:
            ext.init(lib)
        except Exception:
            pass
    return ext


def _current_stream_ptr() -> int:
    import torch

    global _CURRENT_STREAM_PTR
    _ensure_stream_tracking()
    if _CURRENT_STREAM_PTR is None:
        _CURRENT_STREAM_PTR = int(torch.npu.current_stream().npu_stream)
    return _CURRENT_STREAM_PTR


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
    if g is None and gk is None:
        raise RuntimeError(
            "npu_recurrent_gated_delta_rule: either g or gk must be provided.")
    ext = _extension()
    return ext.npu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta,
        float(scale),
        actual_seq_lengths,
        ssm_state_indices,
        num_accepted_tokens,
        g,
        gk,
        _current_stream_ptr(),
    )


def npu_kda_gate_cumsum(
    g,
    chunk_size,
    *,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    use_gate_in_kernel=False,
    safe_gate=False,
    lower_bound=None,
):
    ext = _extension()
    cu = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    lb = -5.0 if lower_bound is None else float(lower_bound)
    return ext.npu_kda_gate_cumsum(
        g,
        A_log,
        dt_bias,
        cu,
        int(chunk_size),
        bool(use_gate_in_kernel),
        bool(safe_gate),
        lb,
        _current_stream_ptr(),
    )


def npu_chunk_local_cumsum(g, chunk_size, *, cu_seqlens=None, chunk_indices=None, reverse=False, scale=1.0, head_first=True, output_dtype="float32"):
    ext = _extension()
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    return ext.npu_chunk_local_cumsum(
        g,
        cu_seqlens,
        chunk_indices,
        int(chunk_size),
        bool(reverse),
        float(scale),
        bool(head_first),
        str(output_dtype),
        _current_stream_ptr(),
    )


def npu_chunk_scaled_dot_kkt(k, g, beta, *, cu_seqlens=None, chunk_indices=None, chunk_size=64):
    ext = _extension()
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    return ext.npu_chunk_scaled_dot_kkt(
        k,
        g,
        beta,
        cu_seqlens,
        chunk_indices,
        int(chunk_size),
        _current_stream_ptr(),
    )


def npu_recompute_w_u_fwd(k, v, beta, A, chunk_size, *, g=None, gk=None, cu_seqlens=None, chunk_indices=None):
    ext = _extension()
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    result = ext.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        g,
        gk,
        cu_seqlens,
        chunk_indices,
        int(chunk_size),
        _current_stream_ptr(),
    )
    return tuple(result)


def npu_prepare_wy_repr_bwd_full(k, v, beta, A, dA, dw, du, g, chunk_size, *, cu_seqlens=None, chunk_indices=None):
    ext = _extension()
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    result = ext.npu_prepare_wy_repr_bwd_full(
        k,
        v,
        beta,
        A,
        dA,
        dw,
        du,
        g,
        cu_seqlens,
        chunk_indices,
        int(chunk_size),
        _current_stream_ptr(),
    )
    return tuple(result)


def npu_prepare_wy_repr_bwd(k, v, beta, A, dw, du, g, chunk_size, *, cu_seqlens=None, chunk_indices=None):
    ext = _extension()
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    result = ext.npu_prepare_wy_repr_bwd(
        k,
        v,
        beta,
        A,
        dw,
        du,
        g,
        cu_seqlens,
        chunk_indices,
        int(chunk_size),
        _current_stream_ptr(),
    )
    return tuple(result)


def npu_chunk_bwd_dv_local(q, k, d_o, g, scale, chunk_size, *, g_gamma=None, A=None, cu_seqlens=None, chunk_indices=None):
    ext = _extension()
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    return ext.npu_chunk_bwd_dv_local(
        q,
        k,
        d_o,
        g,
        g_gamma,
        A,
        cu_seqlens,
        chunk_indices,
        float(scale),
        int(chunk_size),
        _current_stream_ptr(),
    )


def npu_prepare_wy_repr_bwd_da(k, v, beta, A, dw, du, g, *, cu_seqlens=None, chunk_indices=None, chunk_size=None):
    ext = _extension()
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    return ext.npu_prepare_wy_repr_bwd_da(
        k,
        v,
        beta,
        A,
        dw,
        du,
        g,
        cu_seqlens,
        chunk_indices,
        int(chunk_size),
        _current_stream_ptr(),
    )


def npu_fast_gelu_custom(self):
    ext = _extension()
    return ext.npu_fast_gelu_custom(
        self,
        _current_stream_ptr(),
    )


def npu_fast_gelu_custom_backward(grad, self):
    ext = _extension()
    return ext.npu_fast_gelu_custom_backward(
        grad,
        self,
        _current_stream_ptr(),
    )


def npu_chunk_bwd_dqkwg(q, k, v, g, h, dox, dh, dv, chunk_size, *, cu_seqlens=None, chunk_indices=None, w=None, g_gamma=None, scale=None, use_exp2=None, transpose_state_layout=None):
    ext = _extension()
    scale = (1.0 if scale is None else float(scale))
    use_exp2 = (False if use_exp2 is None else bool(use_exp2))
    transpose_state_layout = (False if transpose_state_layout is None else bool(transpose_state_layout))
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    result = ext.npu_chunk_bwd_dqkwg(
        q,
        k,
        v,
        g,
        h,
        dox,
        dh,
        dv,
        cu_seqlens,
        chunk_indices,
        w,
        g_gamma,
        float(scale),
        int(chunk_size),
        bool(use_exp2),
        bool(transpose_state_layout),
        _current_stream_ptr(),
    )
    return tuple(result)


def npu_chunk_gated_delta_rule_fwd_h(k, w, u, g, *, gk=None, initial_state=None, output_final_state=False, chunk_size=None, cu_seqlens=None, chunk_indices=None, state_v_first=False):
    ext = _extension()
    chunk_size = (64 if chunk_size is None else int(chunk_size))
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    if cu_seqlens and not chunk_indices:
        chunk_indices = []
        for _seq in range(len(cu_seqlens) - 1):
            _len = cu_seqlens[_seq + 1] - cu_seqlens[_seq]
            for _c in range((_len + chunk_size - 1) // chunk_size):
                chunk_indices.extend((_seq, _c))
    result = ext.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g,
        gk,
        initial_state,
        bool(output_final_state),
        int(chunk_size),
        cu_seqlens,
        chunk_indices,
        bool(state_v_first),
        _current_stream_ptr(),
    )
    return (result[0], result[1], (result[2] if output_final_state else None))


def npu_chunk_fwd_h(k, w, u, *, g=None, gk=None, initial_state=None, output_final_state=False, chunk_size=64, save_new_value=True, cu_seqlens=None, chunk_indices=None, use_exp2=False, state_v_first=False):
    ext = _extension()
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    if cu_seqlens and not chunk_indices:
        chunk_indices = []
        for _seq in range(len(cu_seqlens) - 1):
            _len = cu_seqlens[_seq + 1] - cu_seqlens[_seq]
            for _c in range((_len + chunk_size - 1) // chunk_size):
                chunk_indices.extend((_seq, _c))
    result = ext.npu_chunk_fwd_h(
        k,
        w,
        u,
        g,
        gk,
        initial_state,
        bool(output_final_state),
        int(chunk_size),
        bool(save_new_value),
        cu_seqlens,
        chunk_indices,
        bool(use_exp2),
        bool(state_v_first),
        _current_stream_ptr(),
    )
    return (result[0], result[1], (result[2] if output_final_state else None))


def npu_chunk_fwd_o(q, k, v, h, scale, *, g=None, cu_seqlens=None, chunk_indices=None, chunk_size=None, use_exp2=False, transpose_state_layout=False, output_layout="BNSD", g_gamma=None):
    ext = _extension()
    chunk_size = (64 if chunk_size is None else int(chunk_size))
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    return ext.npu_chunk_fwd_o(
        q,
        k,
        v,
        h,
        g,
        cu_seqlens,
        chunk_indices,
        float(scale),
        int(chunk_size),
        bool(use_exp2),
        bool(transpose_state_layout),
        str(output_layout),
        _current_stream_ptr(),
    )


def npu_chunk_gated_delta_rule_bwd_dhu(q, k, w, d_o, dv, scale, chunk_size, *, g=None, gK=None, h0=None, dht=None, cu_seqlens=None, chunk_indices=None, use_exp2=False):
    ext = _extension()
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    result = ext.npu_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        d_o,
        dv,
        g,
        gK,
        h0,
        dht,
        cu_seqlens,
        chunk_indices,
        float(scale),
        int(chunk_size),
        bool(use_exp2),
        _current_stream_ptr(),
    )
    return (result[0], (result[1] if h0 is not None else None), result[2])
