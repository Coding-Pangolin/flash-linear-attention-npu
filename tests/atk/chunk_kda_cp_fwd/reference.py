"""Independent CPU KDA oracle; no fla_npu or NPU dependencies.

Semantics: PR #866 at f9cc41fa, recurrent_reference, with explicit FP64 math.
This mathematical oracle is not a bitwise model of chunk BF16 rounding.
"""
from __future__ import annotations

import torch


def reference(q, k, v, g, beta, *, A_log, dt_bias=None, scale=None,
              initial_state=None, epsilon=1e-6):
    """Return BSND output and FP64 final state; preserve CPU autograd."""
    if any(t.device.type != "cpu" for t in (q, k, v, g, beta, A_log)):
        raise ValueError("reference accepts CPU tensors only")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape != g.shape:
        raise ValueError("q/k/v/g must have equal BNSD shapes")
    b, h, tokens, dim = q.shape
    if b != 1 or dim != 128 or tokens < 1 or h < 1 or beta.shape != q.shape[:-1]:
        raise ValueError("expected B=1, H,T>0, K=V=128 and BNS beta")
    if A_log.shape != (h,):
        raise ValueError("A_log must be [H]")
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + epsilon)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + epsilon)
    bias = 0 if dt_bias is None else dt_bias.double().reshape(1, h, 1, dim)
    gate = -5.0 * torch.sigmoid(A_log.double().exp()[None, :, None, None] * (g + bias))
    beta = beta.sigmoid()
    state = (q.new_zeros((b, h, dim, dim)) if initial_state is None
             else initial_state.double())
    if state.shape != (b, h, dim, dim):
        raise ValueError("invalid initial_state shape")
    scale = dim ** -0.5 if scale is None else scale
    outputs = []
    for t in range(tokens):
        decayed = state * gate[:, :, t].exp().unsqueeze(-1)
        key = k[:, :, t]
        error = (v[:, :, t] - (key.unsqueeze(-1) * decayed).sum(-2)) * beta[:, :, t, None]
        state = decayed + key.unsqueeze(-1) * error.unsqueeze(-2)
        outputs.append((q[:, :, t, :, None] * state).sum(-2) * scale)
    return torch.stack(outputs, dim=1), state


def merge_reference(summaries, rank):
    """Exclusive ordered prefix of CPU [P,H,K,V+K] FP32 summaries."""
    if summaries.device.type != "cpu" or summaries.ndim != 4:
        raise ValueError("expected CPU [P,H,K,V+K] summaries")
    p, h, k, width = summaries.shape
    if k != 128 or width != 256 or not 0 <= rank < p:
        raise ValueError("invalid summary shape or rank")
    state = summaries.new_zeros((h, k, 128), dtype=torch.float64)
    for source in range(rank):
        e = summaries[source, ..., :128].double()
        m = summaries[source, ..., 128:].double()
        state = m @ state + e
    return state


def packed_shard_reference(q, k, v, g, beta, *, A_log, cu_seqlens, start, end,
                           dt_bias=None, scale=None):
    """Run global zero-start recurrence and keep output and states for one rank."""
    pieces = []
    states = []
    for left, right in zip(cu_seqlens, cu_seqlens[1:]):
        if left >= end or right <= start:
            continue
        stop = min(end, right)
        inputs = [x[:, :, left:stop] for x in (q, k, v, g, beta)]
        output, state = reference(*inputs, A_log=A_log, dt_bias=dt_bias, scale=scale)
        pieces.append(output[:, max(0, start - left):])
        states.append(state)
    return torch.cat(pieces, dim=1), torch.cat(states, dim=0)
