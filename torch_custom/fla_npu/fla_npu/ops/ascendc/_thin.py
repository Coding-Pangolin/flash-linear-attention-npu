# Thin C++ launcher adapters for the hot FLA NPU operators.
#
# Enabled per-process with FLA_NPU_THIN_LAUNCHER=1 after the extension has been
# built (FLA_NPU_BUILD_THIN=1). Signatures mirror the ctypes wrappers in
# _aclnn_ctypes.py so the existing public API and mutation contracts are kept.
from __future__ import annotations

import os


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

    return int(torch.npu.current_stream().npu_stream)


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


def npu_causal_conv1d(
    x,
    weight,
    bias=None,
    conv_states=None,
    *,
    query_start_loc=None,
    cache_indices=None,
    initial_state_mode=None,
    num_accepted_tokens=None,
    activation_mode=0,
    pad_slot_id=-1,
    run_mode=0,
    head_num=0,
):
    ext = _extension()

    def as_list(value):
        return [] if value is None else [int(v) for v in value]

    return ext.npu_causal_conv1d(
        x,
        weight,
        bias,
        conv_states,
        as_list(query_start_loc),
        as_list(cache_indices),
        as_list(initial_state_mode),
        as_list(num_accepted_tokens),
        int(activation_mode),
        int(pad_slot_id),
        int(run_mode),
        int(head_num),
        _current_stream_ptr(),
    )
