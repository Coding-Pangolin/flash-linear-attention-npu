"""Host plan and device reference for continuous packed KDA CP forward."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from operator import index


@dataclass(frozen=True)
class PackedShard:
    cu_seqlens: tuple[int, ...]
    first_sequence: int
    prefix_start_rank: int
    sends_summary: bool

    @property
    def sequences(self):
        return len(self.cu_seqlens) - 1


@dataclass(frozen=True)
class PackedKcpPlan:
    global_cu_seqlens: tuple[int, ...]
    shards: tuple[PackedShard, ...]
    heads: int
    head_group_size: int
    rounds: int
    batch_slots: int
    receive_slots: int
    compute_groups: int

    def owner(self, local_sequence, head, rank):
        groups = min(self.compute_groups, self.shards[rank].sequences * self.heads)
        return (local_sequence * self.heads + head) % groups

    def workspace(self, rank):
        """Byte offsets and sizes for the estimated kernel scratch regions."""
        kib = 1024
        full_summary = 128 * kib
        full_state = 64 * kib
        groups = min(self.compute_groups, self.shards[rank].sequences * self.heads)
        sizes = (
            ("send", self.batch_slots * self.head_group_size * full_summary),
            ("receive", self.receive_slots * len(self.shards) * self.head_group_size * full_summary),
            ("initial_state", self.batch_slots * self.head_group_size * full_state),
            ("compute", groups * 768 * kib),
            ("control", 64 * kib),
        )
        offset = 0
        regions = {}
        for name, size in sizes:
            regions[name] = (offset, size)
            offset += size
        return regions, offset


def build_packed_plan(local_tokens, heads, world, compute_groups, *, cu_seqlens=None,
                      head_group_size=None):
    """Apply the v1 auto-policy to global metadata shared by all ranks."""
    if type(world) is not int or world not in (2, 4, 8):
        raise ValueError("packed KDA CP requires CP2, CP4 or CP8")
    if type(local_tokens) is not int or not 0 < local_tokens <= 2**31 - 1:
        raise ValueError("local_tokens must fit positive int32")
    if any(type(n) is not int or n < 1 for n in (heads, compute_groups)):
        raise ValueError("heads and compute_groups must be positive integers")
    total = local_tokens * world
    if cu_seqlens is None:
        cu_seqlens = (0, total)
    elif getattr(cu_seqlens, "ndim", 1) != 1:
        raise ValueError("cu_seqlens must be one-dimensional")
    elif str(getattr(cu_seqlens, "device", "cpu")) != "cpu":
        raise ValueError("cu_seqlens must be on the CPU")
    elif str(getattr(cu_seqlens, "dtype", "int64")) not in ("int32", "int64", "torch.int32", "torch.int64"):
        raise ValueError("cu_seqlens tensors must be int32 or int64")
    try:
        supplied = tuple(cu_seqlens)
        if any(isinstance(x, bool) or str(getattr(x, "dtype", "")) in ("bool", "torch.bool")
               for x in supplied):
            raise TypeError("boolean offsets are invalid")
        boundaries = tuple(index(x) for x in supplied)
    except TypeError as exc:
        raise ValueError("cu_seqlens must contain integer offsets") from exc
    if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != total
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError("global cu_seqlens must strictly increase from 0 to CP * local_tokens")

    shards = []
    for rank in range(world):
        start, end = rank * local_tokens, (rank + 1) * local_tokens
        first = bisect_right(boundaries, start) - 1
        last_exclusive = bisect_left(boundaries, end)
        offsets = (0,) + tuple(t - start for t in boundaries[first + 1:last_exclusive]) + (local_tokens,)
        shards.append(PackedShard(offsets, first, boundaries[first] // local_tokens,
                                  boundaries[last_exclusive] > end))

    half = (heads + 1) // 2
    if head_group_size is not None and (type(head_group_size) is not int
                                        or head_group_size not in (heads, half)):
        raise ValueError("head_group_size must be HV or ceil(HV/2)")
    crosses = any(shard.sends_summary for shard in shards)
    if head_group_size is None:
        blocked = any(shard.sequences == 1 and shard.prefix_start_rank < rank
                      for rank, shard in enumerate(shards))
        head_group_size = half if crosses and blocked and half <= compute_groups < heads else heads
    rounds = (heads + head_group_size - 1) // head_group_size if crosses else 0
    return PackedKcpPlan(boundaries, tuple(shards), heads, head_group_size, rounds,
                         rounds, rounds, compute_groups)


def packed_forward_staged(q, k, v, g, beta, *, A_log, cp_group, plan,
                          dt_bias=None, scale=None, output_final_state=False):
    """Materializing A5 correctness baseline for the planned single-kernel path."""
    import torch
    import torch.distributed as dist
    from ._kcp import KdaCpResult, _ops

    rank, world = dist.get_rank(cp_group), dist.get_world_size(cp_group)
    heads, tokens = q.shape[1:3]
    if len(plan.shards) != world or plan.heads != heads or plan.shards[rank].cu_seqlens[-1] != tokens:
        raise ValueError("local tensor shape and CP group must match the packed plan")
    shard = plan.shards[rank]
    cu = shard.cu_seqlens
    scale = 128 ** -0.5 if scale is None else scale
    prepare, summary_op, merge, fwd_h, finalize = _ops()
    output = []
    states = []
    width = plan.head_group_size if plan.rounds else heads

    for first in range(0, heads, width):
        last = min(first + width, heads)
        valid = last - first
        qa, ka, va, ga, ba = (x[:, first:last].contiguous() for x in (q, k, v, g, beta))
        a = A_log[first:last].clone()
        bias = None if dt_bias is None else dt_bias.reshape(heads, 128)[first:last].flatten().clone()
        gk, aqk, _, w, u, _, kg, qgs, *_ = prepare(
            qa, ka, va, ga, ba, scale, layout="BNSD", cu_seqlens=cu,
            backward_mode="none", safe_gate=True, lower_bound=-5.0,
            use_gate_in_kernel=True, A_log=a, dt_bias=bias, use_exp2=True,
            use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        )
        h0 = torch.zeros((shard.sequences, valid, 128, 128), dtype=torch.float32, device=q.device)
        if plan.rounds:
            send = torch.zeros((width, 128, 256), dtype=torch.float32, device=q.device)
            if shard.sends_summary:
                send[:valid].copy_(summary_op(kg, w, u, gk=gk, cu_seqlens=cu[-2:])[0])
            received = torch.empty((world * width, 128, 256), dtype=send.dtype, device=send.device)
            dist.all_gather_into_tensor(received, send, group=cp_group)
            prefix = rank - shard.prefix_start_rank
            if prefix:
                ordered = received.view(world, width, 128, 256)[:, :valid].contiguous()
                merge(h0[0], ordered, prefix, rank, forward=True, state_v_first=False)

        h, vn, final = fwd_h(kg, w, u, gk=gk, initial_state=h0, cu_seqlens=cu,
                             output_final_state=output_final_state, use_exp2=True)
        output.append(finalize(qgs, aqk, vn, h, cu_seqlens=cu, output_layout="BSND"))
        if output_final_state:
            states.append(final)

    o = output[0] if len(output) == 1 else torch.cat(output, dim=2)
    final = (states[0] if len(states) == 1 else torch.cat(states, dim=1)) if states else None
    return KdaCpResult(o, final)
