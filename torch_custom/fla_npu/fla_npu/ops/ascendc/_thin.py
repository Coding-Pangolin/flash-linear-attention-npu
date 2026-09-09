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


def npu_recurrent_kda(q, k, v, g, beta, initial_state, *, cu_seqlens=None, ssm_state_indices=None, A_log=None, dt_bias=None, num_accepted_tokens=None, layout="BSND", scale=None, output_final_state=False, inplace_final_state=True, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, safe_gate=False, lower_bound=None, state_v_first=False):
    ext = _extension()
    import torch as _torch
    layout = str(layout)
    if scale is None:
        _kd = k.shape[2] if layout == "TND" else k.shape[3]
        scale = _kd ** -0.5
    inplace_final_state = True if inplace_final_state is None else bool(inplace_final_state)
    if inplace_final_state and initial_state is None:
        raise RuntimeError("npu_recurrent_kda: inplace_final_state=True requires initial_state.")
    if initial_state is None:
        _seq = 1 if layout == "TND" else q.shape[0]
        _hv = v.shape[1] if layout == "TND" else v.shape[2]
        _kk = k.shape[2] if layout == "TND" else k.shape[3]
        _vv = v.shape[2] if layout == "TND" else v.shape[3]
        _tail = (_vv, _kk) if state_v_first else (_kk, _vv)
        initial_state = _torch.zeros((_seq, _hv) + _tail, dtype=_torch.float32, device=v.device)
    lower_bound = (-5.0 if lower_bound is None else float(lower_bound))
    result = ext.npu_recurrent_kda(
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
        str(layout),
        float(scale),
        bool(output_final_state),
        bool(inplace_final_state),
        bool(use_qk_l2norm_in_kernel),
        bool(use_gate_in_kernel),
        bool(use_beta_sigmoid_in_kernel),
        bool(allow_neg_eigval),
        bool(safe_gate),
        float(lower_bound),
        bool(state_v_first),
        _current_stream_ptr(),
    )
    return (result[0], (result[1] if output_final_state else None))


def npu_chunk_gated_delta_rule_fwd_prepare(q, k, v, g, beta, chunk_size, *, a_log=None, dt_bias=None, cu_seqlens=None, chunk_indices=None, allow_neg_eigval=False, use_exp2=False, output_a=True, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False):
    ext = _extension()
    if not (bool(use_qk_l2norm_in_kernel) and not bool(use_gate_in_kernel) and bool(use_exp2) and bool(use_beta_sigmoid_in_kernel) and bool(output_a) and int(chunk_size) == 64 and a_log is None and dt_bias is None):
        from fla_npu.ops.ascendc import _aclnn_ctypes as _ct
        return _ct.npu_chunk_gated_delta_rule_fwd_prepare(q, k, v, g, beta, chunk_size, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, use_gate_in_kernel=use_gate_in_kernel, use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel, allow_neg_eigval=allow_neg_eigval, use_exp2=use_exp2, a_log=a_log, dt_bias=dt_bias, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_a=output_a)
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    result = ext.npu_chunk_gated_delta_rule_fwd_prepare(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        cu_seqlens,
        chunk_indices,
        int(chunk_size),
        bool(allow_neg_eigval),
        bool(use_exp2),
        bool(output_a),
        _current_stream_ptr(),
    )
    return (result[4], result[5], result[6], result[7], result[8], result[0], result[1], result[2], result[3])


def npu_causal_conv1d_bwd(x, y, weight, dy, initial_state, dht, *, query_start_loc=None, activation=0, input_layout="BSND"):
    ext = _extension()
    query_start_loc = [] if query_start_loc is None else [int(v) for v in query_start_loc]
    result = ext.npu_causal_conv1d_bwd(
        x,
        y,
        weight,
        dy,
        initial_state,
        dht,
        query_start_loc,
        int(activation),
        str(input_layout),
        _current_stream_ptr(),
    )
    return tuple(result)


def npu_chunk_kda_fwd(q, k, v, g, beta, scale, chunk_size, *, A_log=None, dt_bias=None, initial_state=None, cu_seqlens=None, chunk_indices=None, layout="BSND", safe_gate=False, lower_bound=None, use_gate_in_kernel=False, state_v_first=False, output_final_state=False, disable_recompute=False, return_intermediate_states=False):
    ext = _extension()
    layout = str(layout)
    if not (bool(disable_recompute) and not bool(output_final_state) and not bool(return_intermediate_states) and layout == "BSND" and cu_seqlens is None and chunk_indices is None):
        from fla_npu.ops.ascendc import _aclnn_ctypes as _ct
        return _ct.npu_chunk_kda_fwd(q, k, v, g, beta, scale, chunk_size=chunk_size, layout=layout, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, safe_gate=safe_gate, lower_bound=lower_bound, use_gate_in_kernel=use_gate_in_kernel, A_log=A_log, dt_bias=dt_bias, disable_recompute=disable_recompute, return_intermediate_states=return_intermediate_states, state_v_first=state_v_first)
    if scale is None:
        scale = float(q.shape[3]) ** -0.5
    lower_bound = -5.0 if lower_bound is None else float(lower_bound)
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    result = ext.npu_chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        cu_seqlens,
        chunk_indices,
        str(layout),
        float(scale),
        int(chunk_size),
        bool(safe_gate),
        float(lower_bound),
        bool(use_gate_in_kernel),
        bool(state_v_first),
        _current_stream_ptr(),
    )
    return (result[0], (result[1] if output_final_state else None), result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], initial_state)


def npu_chunk_kda_bwd_intra(q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg, *, cu_seqlens=None, chunk_indices=None, chunk_size=64, safe_gate=True, layout="BSND"):
    ext = _extension()
    layout = str(layout)
    if not (layout == "BNSD" and cu_seqlens is None and chunk_indices is None and int(chunk_size) == 64 and bool(safe_gate)):
        from fla_npu.ops.ascendc import _aclnn_ctypes as _ct
        return _ct.npu_chunk_kda_bwd_intra(q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=chunk_size, safe_gate=safe_gate, layout=layout)
    cu_seqlens = [] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    chunk_indices = [] if chunk_indices is None else [int(v) for v in chunk_indices]
    result = ext.npu_chunk_kda_bwd_intra(
        q,
        k,
        gk,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        cu_seqlens,
        chunk_indices,
        int(chunk_size),
        bool(safe_gate),
        str(layout),
        _current_stream_ptr(),
    )
    return tuple(result)
