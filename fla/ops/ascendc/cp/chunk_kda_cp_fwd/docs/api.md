# ChunkKdaCpFwd contract

Approved basis: the implementation plan accepted by the user on 2026-11-17.
New entrypoint; existing KDA/CP interfaces and ABIs are unchanged.

```python
chunk_kda_cp_fwd(q, k, v, g, beta, *, cp_group, A_log, dt_bias=None,
                 scale=None, backward_mode="save", output_final_state=False,
                 backend="staged", head_group_size=None, plan=None, cu_seqlens=None)
```

Inputs are contiguous, public-format NPU tensors. Q/K/V are BF16 BNSD
`[1,H,T,128]`; raw gate is FP32 of that shape; raw beta is FP32 `[1,H,T]`.
`A_log` is FP32 `[H]`; optional `dt_bias` is FP32 `[H,128]` or `[H*128]`.
H and T are positive, HK=HV, chunk_size=64, state_v_first=False. A2 and A5
share the signature; runtime support is limited to individually tested paths.

CP group is explicit. Its group-rank order is the contiguous global sequence
order. All group members have equal local shapes and options. Default scale is
128**-0.5. The global initial state is zero. Q/K L2 norm uses epsilon=1e-6;
SafeSigmoid gate uses lower_bound=-5 and log2 chunk cumsum; beta is sigmoid.
The A5 path uses external beta sigmoid when required by the backward ABI.

The result has fields `output` (BF16 BSND `[1,T,H,128]`), `final_state`
(FP32 `[1,H,128,128]` or None), and `cache` (training cache or None).
Final state belongs to the end of this rank's shard; only the last rank's state
is the global final state. `save` preserves the existing staged backward's
prepared tensors, BF16 H/V_new, and FP32 h0; `none` exports no training cache.
The entrypoint is an explicit-forward API like PR #866, not an autograd wrapper.
Inputs requiring grad are rejected; explicit backward consumes the saved cache.
The cache retains v/g/beta input views. Keep those inputs alive and unmodified
until backward; this explicit API does not install autograd version checks.

`staged` executes all heads before framework AllGather and merge. `pipeline`
groups heads and schedules framework AllGather on a second stream, aiming to
overlap next-group production; actual overlap depends on the stack and must be
measured. Neither is a native MC2 claim. `mc2` calls the registered
`KcpAllGatherMerge` boundary and currently requires A2, CP2..8 and group size
1..16. A5 native resource allocation has not passed, so its aggregate native
path fails explicitly. With one head group, all backends use the caller stream:
there is no next producer with which to overlap communication.
No backend silently falls back during a measured run. Group size must be a
positive integer; the last group may be smaller. CP=1 needs no data collective.

An explicit collective preflight checks shapes/options on all ranks before
entering the first data collective. Invalid metadata, unsupported device,
non-finite attributes, and mismatched groups fail before kernel work. NPU data
values are checked by the accuracy runner, outside the measured path.
The preflight also checks actual public NPU formats and installed dependency
symbols without executing a kernel. `create_kda_cp_plan` performs this once;
passing its result avoids repeating collective preflight inside steady-state
timing. Plans are process/device/group specific and are not concurrency-safe.

Grouped paths reject a static materialized temporary-buffer estimate above
200000000 bytes. The native estimate adds the library and Merge workspace.
This is a rejection guard, not a measured memory guarantee: HCCL persistent
reservations and actual device peaks must still be included in acceptance.
Required save-mode caches belong to returned outputs and are counted separately;
transient u/qgs, tail-group storage and grouped output copies remain scratch.
The full-head `staged` reference keeps its original materialized footprint.

## Explicit backward

```python
gradients = chunk_kda_cp_bwd(do, result.cache)
```

`do` is BF16 BSND `[1,T,H,128]`. This computes gradients of all ranks' output
losses with zero global terminal-state adjoint; gradients through the optional
`final_state` result are not implemented. Output names are `dq/dk/dv/dg`
(BSND), `dbeta` (BSN), `dA_log` (`[H]`) and `ddt_bias` (`[H,128]`). Reshape
the latter if the caller originally supplied flattened bias; discard it when
there was no bias parameter. Shared-parameter gradients are summed once in the
explicit CP group. A model's TP/DP gradient policy remains the caller's task.
This backward is staged, not a fused backward or an autograd registration.
All ranks call backward with the matching save-mode cache. Its collective
preflight rejects a wrong dO shape/dtype/device or requires-grad on every rank
before entering gradient data collectives. Higher-order derivatives are unsupported.

The staged training interface accepts one sequence per rank. The A5 fused
backend supports continuous packed sequences and forward-only mode. The optional
CPU `cu_seqlens` describes the entire group. The result contains one optional
FP32 terminal state per local segment. Use `head_group_size=H` for one batch
or `ceil(H/2)` for two batches. A plan selects automatically when the argument
is omitted. The original group-size default of 4 still applies to the staged,
pipeline and mc2 paths. The fused backend accepts up to 128 heads and CP2/4/8
metadata. Device acceptance currently covers CP2.

The fused backend supports continued rank slices, BF16 K/V dimension 128 and
zero global initial state. Tail chunks are allowed. Global no-CP comparisons with differently placed
chunk boundaries are numerical comparisons, not bitwise assertions.
