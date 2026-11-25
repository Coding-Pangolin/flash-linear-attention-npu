"""Experimental KDA CP forward, framework baseline and registered MC2 boundary.

Compute and native communication use public Stable-ABI operators. The pipeline
backend uses framework HCCL; mc2 uses the registered AllGather/merge kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Any


@dataclass
class KdaCpGroupCache:
    first_head: int
    last_head: int
    tensors: dict[str, Any]
    fused_beta: bool


@dataclass
class KdaCpCache:
    groups: tuple[KdaCpGroupCache, ...]
    cp_group: Any
    rank: int
    world: int
    tokens: int
    heads: int
    scale: float


@dataclass
class KdaCpResult:
    output: Any
    final_state: Any = None
    cache: KdaCpCache | None = None


@dataclass
class KdaCpPlan:
    """Explicit reusable preflight; creation is outside steady-state timing.

    Plans are local to a process/device/group and may not be used concurrently.
    Regular work buffers remain invocation-owned. An IPC experiment may
    hold fixed buffers in its explicitly closed preview state.
    """
    signature: tuple
    cp_group: Any
    rank: int
    world: int
    communication_stream: Any
    device: str
    native_group_name: str | None = None
    packed: Any = None
    device_offsets: Any = None
    native_group_bytes: Any = None
    zero_bias: Any = None
    _ipc_preview: Any = None


def _ops():
    # Import public wrappers lazily, preserving import-without-torch behavior.
    from . import (chunk_kda_fwd_prepare, pre_process_fwd_kernel_merged,
                   merge_fwd_bwd_kernel, chunk_fwd_h, chunk_kda_fwd_finalize)
    return (chunk_kda_fwd_prepare, pre_process_fwd_kernel_merged,
            merge_fwd_bwd_kernel, chunk_fwd_h, chunk_kda_fwd_finalize)


_NATIVE_STATE_RELAY_PLAN = 3


def _uses_chunk32_zero_round(packed):
    # Keep the rollout within the verified task count. The original
    # kernel stalled on A5 with 128 one-token sequences and one head.
    # The legacy path enters MC2 even in zero data rounds, so choose group-wide.
    return packed.rounds == 0 and all(
        shard.sequences * packed.heads <= 64 for shard in packed.shards
    )


def _check_dependencies(mode, backend, world, *, chunk32_zero_round=False):
    """Inspect the configured ELF exports without executing an operator."""
    import os
    from pathlib import Path
    from . import _stable
    from ..._op_abi import elf_exported_symbols
    if chunk32_zero_round:
        required = {"npu_chunk_kda_fwd": "aclnnChunkKdaFwdV2"}
    elif backend == "fused":
        required = {"npu_chunk_kda_cp_fwd": "aclnnChunkKdaCpFwd"}
    else:
        required = {
            "npu_chunk_kda_fwd_prepare": "aclnnChunkKdaFwdPrepare",
            "npu_chunk_fwd_h": "aclnnChunkFwdH",
            "npu_chunk_kda_fwd_finalize": "aclnnChunkKdaFwdFinalize",
        }
    if world > 1 and backend != "fused":
        required["npu_pre_process_fwd_kernel_merged"] = "aclnnPreProcessFwdKernelMerged"
        required["npu_kcp_all_gather_merge" if backend == "mc2" else "npu_merge_fwd_bwd_kernel"] = (
            "aclnnKcpAllGatherMerge" if backend == "mc2" else "aclnnMergeFwdBwdKernel")
    if mode == "save":
        required.update({
            "npu_chunk_kda_bwd_prepare": "aclnnChunkKdaBwdPrepare",
            "npu_pre_process_bwd_kernel_merged": "aclnnPreProcessBwdKernelMerged",
            "npu_merge_fwd_bwd_kernel": "aclnnMergeFwdBwdKernel",
            "npu_chunk_gated_delta_rule_bwd_dhu": "aclnnChunkGatedDeltaRuleBwdDhu",
            "npu_chunk_kda_bwd_finalize": "aclnnChunkKdaBwdFinalize",
        })
    for name in required:
        _stable._op(name)
    context = _stable._layer_context()
    paths = (context["libraries"] if context else
             (os.environ.get("FLA_NPU_OP_API_LIBS") or os.environ.get("FLA_NPU_OP_API_LIB", "")).split(os.pathsep))
    exports = set()
    for path in paths:
        if path:
            exports.update(elf_exported_symbols(Path(path)))
    missing = {symbol + suffix for symbol in required.values() for suffix in ("", "GetWorkspaceSize")} - exports
    if missing:
        raise RuntimeError("install a matching wheel containing KDA CP dependencies: " + ", ".join(sorted(missing)))


def _signature(q, k, v, g, beta, a_log, bias, scale, mode, backend, group_size, final):
    import torch
    errors = []
    shapes = [tuple(x.shape) for x in (q, k, v, g)]
    if q.ndim != 4 or any(s != shapes[0] for s in shapes):
        errors.append("q/k/v/g must have equal BNSD shapes")
    elif q.shape[0] != 1 or q.shape[-1] != 128 or min(q.shape[1:3]) < 1:
        errors.append("expected B=1, H,T>0, K=V=128")
    if tuple(beta.shape) != tuple(q.shape[:-1]):
        errors.append("beta must be BNS")
    heads = q.shape[1] if q.ndim == 4 else 0
    if tuple(a_log.shape) != (heads,):
        errors.append("A_log must be [H]")
    if bias is not None and tuple(bias.shape) not in ((heads, 128), (heads * 128,)):
        errors.append("dt_bias must be [H,128] or [H*128]")
    for name, tensor in zip(("q", "k", "v", "g", "beta", "A_log"),
                            (q, k, v, g, beta, a_log)):
        dtype = torch.bfloat16 if name in ("q", "k", "v") else torch.float32
        if tensor.dtype != dtype:
            errors.append(f"{name} has unsupported dtype")
    tensors = (q, k, v, g, beta, a_log) + (() if bias is None else (bias,))
    if any(x.device != q.device for x in tensors) or q.device.type not in ("npu", "privateuseone"):
        errors.append("all tensors must be on the same NPU")
    if any(not x.is_contiguous() for x in tensors):
        errors.append("all inputs must be contiguous")
    formats = None
    if q.device.type in ("npu", "privateuseone"):
        runtime = sys.modules.get("torch_npu")
        if runtime is None:
            errors.append("initialize the torch_npu runtime before KDA CP")
        else:
            try:
                formats = tuple(int(runtime.get_npu_format(x)) for x in tensors)
                if any(value not in (0, 2, 30, 47) for value in formats):
                    errors.append(f"all inputs must use public NPU formats, got {formats}")
            except Exception as exc:
                errors.append(f"cannot inspect physical NPU formats: {exc}")
    if bias is not None and bias.dtype != torch.float32:
        errors.append("dt_bias must be FP32")
    if any(x.requires_grad for x in tensors):
        errors.append("explicit-forward API requires detached inputs; use saved cache for explicit backward")
    if mode not in ("save", "none"):
        errors.append("backward_mode must be save or none")
    if backend not in ("staged", "pipeline", "mc2", "fused"):
        errors.append("unknown backend")
    if backend == "mc2" and (type(group_size) is not int or not 1 <= group_size <= 16):
        errors.append("native MC2 head_group_size must be 1..16")
    if (backend != "fused" or group_size is not None) and (type(group_size) is not int or group_size < 1):
        errors.append("head_group_size must be a positive integer")
    if not isinstance(scale, (float, int)) or not math.isfinite(scale):
        errors.append("scale must be finite")
    if type(final) is not bool:
        errors.append("output_final_state must be bool")
    signature = (tuple(shapes), tuple(beta.shape), str(q.dtype), str(g.dtype),
                 str(beta.dtype), tuple(a_log.shape), None if bias is None else tuple(bias.shape),
                 float(scale) if isinstance(scale, (float, int)) else repr(scale),
                 mode, backend, group_size, final, formats)
    return signature, errors


def create_kda_cp_plan(q, k, v, g, beta, *, cp_group, A_log, dt_bias=None,
                       scale=None, backward_mode="save", output_final_state=False,
                       backend="staged", head_group_size=None, cu_seqlens=None):
    """All ranks call once before forward; collective errors are shared."""
    import torch
    import torch.distributed as dist
    if not dist.is_initialized() or cp_group is None:
        raise ValueError("initialize HCCL and pass an explicit cp_group")
    if dist.get_backend(cp_group) != "hccl":
        raise ValueError("cp_group must use HCCL")
    rank, world = dist.get_rank(cp_group), dist.get_world_size(cp_group)
    if rank < 0 or world < 1:
        raise ValueError("current process is not a member of cp_group")
    scale = 128 ** -0.5 if scale is None else scale
    if backend != "fused" and head_group_size is None:
        head_group_size = 4
    signature, errors = _signature(q, k, v, g, beta, A_log, dt_bias, scale,
                                   backward_mode, backend, head_group_size, output_final_state)
    if cu_seqlens is not None and backend != "fused":
        errors.append("cu_seqlens is supported by the fused backend")
    packed = None
    if not errors and backend == "fused":
        from ._kcp_packed import build_packed_plan

        if backward_mode != "none":
            errors.append("fused KCP requires backward_mode='none'")
        if "950" not in torch.npu.get_device_name(q.device):
            errors.append("fused KCP requires A5")
        if q.shape[1] > 128:
            errors.append("fused KCP supports at most 128 heads")
        try:
            packed = build_packed_plan(q.shape[2], q.shape[1], world, 27, cu_seqlens=cu_seqlens,
                                       head_group_size=head_group_size)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            head_group_size = packed.head_group_size
            signature, _ = _signature(q, k, v, g, beta, A_log, dt_bias, scale, backward_mode, backend,
                                      head_group_size, output_final_state)
    if not errors and backend in ("pipeline", "mc2"):
        from ._kcp_resources import memory_plan
        width = min(head_group_size, q.shape[1])
        native_workspace = (16 * 2**20 + 3 * width * 128 * 128 * 4) if backend == "mc2" else 0
        resources = memory_plan(q.shape[2], q.shape[1], world, width,
                                training=backward_mode == "save", runtime_bytes=native_workspace,
                                output_final_state=output_final_state, has_bias=dt_bias is not None,
                                external_beta_sigmoid=backward_mode == "save" and "950" in torch.npu.get_device_name(q.device))
        if not resources["within_temporary_budget"]:
            errors.append("materialized temporary buffers exceed 200000000 bytes before communication reservations; reduce head_group_size")
    if backend == "mc2":
        if not 2 <= world <= 8:
            errors.append("native MC2 requires CP2..8")
        if not errors and "910B" not in torch.npu.get_device_name(q.device):
            errors.append("native MC2 is currently gated to A2; A5 communication has not passed")
    chunk32_zero_round = packed is not None and _uses_chunk32_zero_round(packed)
    if not errors:
        try:
            _check_dependencies(backward_mode, backend, world, chunk32_zero_round=chunk32_zero_round)
        except Exception as exc:
            errors.append(str(exc))
    reports = [None] * world
    boundaries = packed.global_cu_seqlens if packed else None
    dist.all_gather_object(reports, ((signature, boundaries), errors), group=cp_group)
    failures = [f"rank {r}: {report[1]}" for r, report in enumerate(reports) if report[1]]
    if any(report[0] != reports[0][0] for report in reports):
        failures.append("CP ranks have mismatched input metadata or options")
    if failures:
        raise ValueError("KDA CP preflight failed: " + "; ".join(failures))
    # torch_npu's communicator-name API takes the global process rank, even
    # when queried on a subgroup. Kernel prefix indices remain group-local.
    needs_native_group = backend in ("mc2", "fused") and not chunk32_zero_round
    name = (cp_group._get_backend(q.device).get_hccl_comm_name(dist.get_rank())
            if needs_native_group else None)
    stream = torch.npu.Stream(device=q.device) if backend in ("pipeline", "mc2") else None
    offsets = (torch.tensor(packed.shards[rank].cu_seqlens, dtype=torch.int32, device=q.device)
               if packed and needs_native_group else None)
    name_bytes = (torch.tensor(list(name.encode("utf-8")), dtype=torch.int64, device="cpu")
                  if packed and needs_native_group else None)
    zero_bias = (torch.zeros((q.shape[1], 128), dtype=torch.float32, device=q.device)
                 if packed and dt_bias is None else None)
    if packed:
        # A reusable plan may be dispatched on a different stream.
        torch.npu.current_stream(q.device).synchronize()
    return KdaCpPlan(signature, cp_group, rank, world, stream, str(q.device), name,
                     packed, offsets, name_bytes, zero_bias)


def chunk_kda_cp_fwd(q, k, v, g, beta, *, cp_group, A_log, dt_bias=None,
                     scale=None, backward_mode="save", output_final_state=False,
                     backend="staged", head_group_size=None, plan=None, cu_seqlens=None):
    import torch
    import torch.distributed as dist
    scale = 128 ** -0.5 if scale is None else scale
    options = dict(cp_group=cp_group, A_log=A_log, dt_bias=dt_bias, scale=scale,
                   backward_mode=backward_mode, output_final_state=output_final_state,
                   backend=backend, head_group_size=head_group_size, cu_seqlens=cu_seqlens)
    if plan is None:
        plan = create_kda_cp_plan(q, k, v, g, beta, **options)
    if backend == "fused" and plan.packed is not None:
        if head_group_size is None:
            head_group_size = plan.packed.head_group_size
        if cu_seqlens is not None:
            from ._kcp_packed import build_packed_plan

            declared = build_packed_plan(q.shape[2], q.shape[1], plan.world, plan.packed.compute_groups,
                                         cu_seqlens=cu_seqlens, head_group_size=head_group_size)
            if declared != plan.packed:
                raise ValueError("cu_seqlens or head_group_size differs from the fused KCP plan")
    elif head_group_size is None:
        head_group_size = 4
    signature, errors = _signature(q, k, v, g, beta, A_log, dt_bias, scale,
                                   backward_mode, backend, head_group_size, output_final_state)
    if (errors or plan.signature != signature or plan.cp_group is not cp_group
            or plan.device != str(q.device)):
        raise ValueError("inputs/options do not match the collectively validated KDA CP plan")
    if dist.get_rank(cp_group) != plan.rank or dist.get_world_size(cp_group) != plan.world:
        raise ValueError("CP group changed after plan creation")
    if backend == "fused":
        from . import _stable

        shard = plan.packed.shards[plan.rank]
        bias = plan.zero_bias if dt_bias is None else dt_bias.reshape(q.shape[1], 128)
        if plan._ipc_preview is not None:
            return plan._ipc_preview.forward(q, k, v, g, beta, A_log, bias, output_final_state)
        if _uses_chunk32_zero_round(plan.packed):
            # NTD consumes head-major inputs as views and writes token-major output.
            output, final, *_ = _stable.npu_chunk_kda_fwd(
                q.squeeze(0), k.squeeze(0), v.squeeze(0), g.squeeze(0), beta.squeeze(0),
                scale, chunk_size=32, layout="NTD", cu_seqlens=shard.cu_seqlens,
                safe_gate=True, use_gate_in_kernel=True, A_log=A_log, dt_bias=bias.reshape(-1),
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
            )
            return KdaCpResult(output.unsqueeze(0), final if output_final_state else None)

        first_source = shard.prefix_start_rank if shard.prefix_start_rank < plan.rank else -1
        rounds = plan.packed.rounds
        # State relay reduces the CP4 critical path on a single continuous
        # sequence. Keep the summary schedule for packed and other CP plans.
        if (plan.world == 4 and rounds == 2 and q.shape[1] == 32 and q.shape[2] >= 512
                and len(plan.packed.global_cu_seqlens) == 2):
            rounds = _NATIVE_STATE_RELAY_PLAN
        output, final = _stable.npu_chunk_kda_cp_fwd(
            q, k, v, g, beta, A_log, bias, plan.device_offsets, plan.native_group_bytes,
            plan.world, plan.rank, rounds, first_source, shard.sends_summary,
            output_final_state, float(scale),
        )
        return KdaCpResult(output, final if output_final_state else None)
    if cu_seqlens is not None:
        raise ValueError("cu_seqlens requires the fused KCP backend")
    prepare, summary_op, merge, fwd_h, finalize = _ops()
    heads, tokens = q.shape[1:3]
    cu = [0, tokens]
    current = torch.npu.current_stream(q.device)
    is_a5 = "950" in torch.npu.get_device_name(q.device)
    fused_beta = not is_a5 or backward_mode == "none"
    group_size = heads if backend == "staged" else min(head_group_size, heads)
    # A single head group has no independent producer to overlap. Avoid four
    # cross-stream event operations in the native-boundary comparison case.
    use_pipeline = backend in ("pipeline", "mc2") and group_size < heads and plan.world > 1
    output = (torch.empty((1, tokens, heads, 128), device=q.device, dtype=q.dtype)
              if group_size < heads else None)
    final_state = (torch.empty((1, heads, 128, 128), device=q.device, dtype=torch.float32)
                   if output_final_state and group_size < heads else None)
    caches = []

    def produce(first, last):
        inputs = [x[:, first:last].contiguous() for x in (q, k, v, g, beta)]
        qa, ka, va, ga, ba = inputs
        # Optional descriptors retain their backing-storage extent. A head
        # slice can be contiguous while still describing the full H allocation;
        # give each group's optional parameter its own exact-size storage.
        aa = A_log[first:last].clone()
        bias = (None if dt_bias is None else
                dt_bias.reshape(heads, 128)[first:last].clone())
        beta_input = ba if fused_beta else ba.sigmoid()
        saved = prepare(qa, ka, va, ga, beta_input, scale, layout="BNSD", cu_seqlens=cu,
                        backward_mode=backward_mode, safe_gate=True, lower_bound=-5.0,
                        use_gate_in_kernel=True, A_log=aa,
                        dt_bias=None if bias is None else bias.flatten(), use_exp2=True,
                        use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=fused_beta)
        if plan.rank + 1 < plan.world:
            summary = summary_op(saved[6], saved[3], saved[4], gk=saved[0], cu_seqlens=cu)[0]
        elif plan.world > 1:
            # Last rank's summary is unused by all exclusive prefixes.
            summary = torch.zeros((last-first, 128, 256), dtype=torch.float32, device=q.device)
        else:
            summary = None
        return dict(first=first, last=last, saved=saved, summary=summary,
                    raw=(va, ga, ba, aa, bias))

    def communicate(work):
        if plan.world == 1:
            work["received"] = None
            return
        summary = work["summary"]
        width = work["last"] - work["first"]
        if backend == "mc2":
            from . import kcp_all_gather_merge
            h0 = torch.zeros((1, width, 128, 128), device=q.device, dtype=torch.float32)
            _, received = kcp_all_gather_merge(summary, h0[0], plan.native_group_name,
                                              plan.world, plan.rank)
            work["h0"], work["received"] = h0, received
            return
        received = torch.empty((plan.world * width, 128, 256), device=q.device, dtype=torch.float32)
        request = dist.all_gather_into_tensor(received, summary, group=cp_group, async_op=True)
        # Establish completion on the active communication stream before
        # recording its event. Enqueue alone is not completion.
        request.wait()
        work["request"] = request
        work["received"] = received.view(plan.world, width, 128, 256)

    def consume(work):
        nonlocal output, final_state
        first, last, saved = work["first"], work["last"], work["saved"]
        gk, aqk, akk, w, u, qg, kg, qgs, qhat, khat, qr, kr, be = saved
        h0 = (work["h0"] if backend == "mc2" else
              torch.zeros((1, last-first, 128, 128), dtype=torch.float32, device=q.device))
        if backend == "mc2" and use_pipeline:
            h0.record_stream(current)
        if plan.rank and backend != "mc2":
            merge(h0[0], work["received"], plan.rank, plan.rank,
                  forward=True, state_v_first=False)
        h, vn, end = fwd_h(kg, w, u, gk=gk, initial_state=h0, cu_seqlens=cu,
                           output_final_state=output_final_state, use_exp2=True)
        local_output = finalize(qgs, aqk, vn, h, cu_seqlens=cu, output_layout="BSND")
        if group_size == heads:
            output = local_output
            if output_final_state:
                final_state = end
        else:
            output[:, :, first:last].copy_(local_output)
            if output_final_state:
                final_state[:, first:last].copy_(end)
        if backward_mode == "save":
            va, ga, ba, aa, bias = work["raw"]
            tensors = dict(gk=gk, aqk=aqk, akk=akk, w=w, qg=qg, kg=kg,
                           qhat=qhat, khat=khat, qr=qr, kr=kr, beta_eff=be,
                           h=h, vn=vn, h0=h0, v=va, g=ga, beta=ba,
                           A_log=aa, dt_bias=bias)
            caches.append(KdaCpGroupCache(first, last, tensors, fused_beta))

    previous = None
    for first in range(0, heads, group_size):
        work = produce(first, min(first + group_size, heads))
        if use_pipeline:
            produced = torch.npu.Event()
            produced.record(current)
            with torch.npu.stream(plan.communication_stream):
                plan.communication_stream.wait_event(produced)
                communicate(work)
                for tensor in (work["summary"], work["received"]):
                    tensor.record_stream(plan.communication_stream)
                ready = torch.npu.Event()
                ready.record(plan.communication_stream)
                work["ready"] = ready
            if previous is not None:
                current.wait_event(previous["ready"])
                previous["received"].record_stream(current)
                consume(previous)
            previous = work
        else:
            communicate(work)
            consume(work)
    if previous is not None:
        current.wait_event(previous["ready"])
        previous["received"].record_stream(current)
        consume(previous)
    cache = (KdaCpCache(tuple(caches), cp_group, plan.rank, plan.world, tokens, heads, scale)
             if backward_mode == "save" else None)
    return KdaCpResult(output, final_state, cache)


def chunk_kda_cp_bwd(do, cache):
    """Explicit CP backward using unchanged staged public operators.

    Shared-parameter gradients are summed once across the explicit CP group.
    This helper does not implement a fused backward or integrate a model's DDP.
    """
    import torch
    import torch.distributed as dist
    from . import (chunk_kda_bwd_prepare, pre_process_bwd_kernel_merged,
                   merge_fwd_bwd_kernel, chunk_gated_delta_rule_bwd_dhu,
                   chunk_kda_bwd_finalize)
    if not isinstance(cache, KdaCpCache):
        raise TypeError("a save-mode KdaCpCache is required")
    errors = []
    expected_device = cache.groups[0].tensors["g"].device
    if not isinstance(do, torch.Tensor):
        errors.append("do must be a Tensor")
    else:
        if tuple(do.shape) != (1, cache.tokens, cache.heads, 128) or do.dtype != torch.bfloat16:
            errors.append("do must be BF16 BSND matching the saved forward")
        if do.device != expected_device:
            errors.append("do must be on the saved forward's NPU")
        if do.requires_grad:
            errors.append("explicit backward requires detached do")
    signature = (cache.tokens, cache.heads, cache.scale, cache.world,
                 tuple((part.first_head, part.last_head) for part in cache.groups))
    reports = [None] * cache.world
    dist.all_gather_object(reports, (signature, errors), group=cache.cp_group)
    failures = [f"rank {r}: {report[1]}" for r, report in enumerate(reports) if report[1]]
    if any(report[0] != reports[0][0] for report in reports):
        failures.append("CP ranks have mismatched saved cache metadata")
    if failures:
        raise ValueError("KDA CP backward preflight failed: " + "; ".join(failures))
    cu = [0, cache.tokens]
    indices = [value for chunk in range((cache.tokens + 63) // 64) for value in (0, chunk)]
    gradients = {name: [] for name in ("dq", "dk", "dv", "dbeta", "dg", "dA_log", "ddt_bias")}
    for part in cache.groups:
        x = part.tensors
        # Prepare/preprocess/Dhu consume BNSD dO, while public output and dO
        # are BSND. Do not reinterpret a token-major view as NTD.
        local_do = do[:, :, part.first_head:part.last_head].transpose(1, 2).contiguous()
        da, dv_local, dq_raw = chunk_kda_bwd_prepare(
            x["aqk"].squeeze(0), x["vn"].squeeze(0), local_do.squeeze(0),
            x["h"].squeeze(0), scale=cache.scale, layout="NTD", cu_seqlens=cu)
        da, dv_local, dq_raw = (t.unsqueeze(0) for t in (da, dv_local, dq_raw))
        width = part.last_head - part.first_head
        if cache.rank:
            summary = pre_process_bwd_kernel_merged(
                x["qg"], x["kg"], x["w"], local_do, dv_local, cache.scale, 64,
                gk=x["gk"].to(torch.bfloat16), cu_seqlens=cu)
        else:
            summary = torch.zeros((width, 128, 256), device=do.device, dtype=torch.float32)
        gathered = torch.empty((cache.world * width, 128, 256), device=do.device, dtype=torch.float32)
        if cache.world > 1:
            dist.all_gather_into_tensor(gathered, summary.contiguous(), group=cache.cp_group)
        else:
            gathered.copy_(summary)
        dht = torch.zeros((1, width, 128, 128), device=do.device, dtype=torch.float32)
        remaining = cache.world - cache.rank - 1
        if remaining:
            merge_fwd_bwd_kernel(dht[0], gathered.view(cache.world, width, 128, 256),
                                 remaining, cache.rank, forward=False, state_v_first=False)
        dh, _, dv_scan = chunk_gated_delta_rule_bwd_dhu(
            x["qg"], x["kg"], x["w"], local_do, dv_local, cache.scale, 64,
            gK=x["gk"], dht=dht.bfloat16(), cu_seqlens=cu,
            chunk_indices=indices, use_exp2=True)
        bias = x["dt_bias"]
        if bias is None:
            bias = torch.zeros((width, 128), device=do.device, dtype=torch.float32)
        result = chunk_kda_bwd_finalize(
            x["qhat"], x["khat"], x["v"], x["gk"], x["g"], x["beta_eff"],
            x["A_log"], bias, x["akk"], x["vn"], x["h"], dh, dv_scan, da, dq_raw,
            scale=cache.scale, q_rstd=x["qr"], k_rstd=x["kr"],
            cu_seqlens=cu, output_layout="BSND", safe_gate=True, lower_bound=-5.0,
            use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=part.fused_beta,
            use_exp2=True)
        dq, dk, dv, db, dg, dA, dbias = result
        if not part.fused_beta:
            effective = x["beta"].sigmoid().transpose(1, 2)
            db = db * effective * (1 - effective)
        for name, tensor in zip(gradients, (dq, dk, dv, db, dg, dA, dbias)):
            gradients[name].append(tensor)
    result = {name: torch.cat(parts, dim=0 if name in ("dA_log", "ddt_bias") else 2)
              for name, parts in gradients.items()}
    if cache.world > 1:
        dist.all_reduce(result["dA_log"], group=cache.cp_group)
        dist.all_reduce(result["ddt_bias"], group=cache.cp_group)
    return result
