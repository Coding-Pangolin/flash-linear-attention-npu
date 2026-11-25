# KDA CP distributed validation

This directory contains the independent CPU recurrence and ordered-prefix
oracle. Device orchestration is in `examples/cp`: an ordinary single-process
ATK invocation cannot stand in for an initialized multi-rank HCCL experiment.
Formal ATK/mssanitizer integration and full acceptance remain pending; the
following distributed checks are development evidence.

## Contract and oracle

See the operator [API](../../../fla/ops/ascendc/cp/chunk_kda_cp_fwd/docs/api.md)
and [validation record](../../../fla/ops/ascendc/cp/chunk_kda_cp_fwd/docs/validation.md).
`reference.py` uses CPU FP64 token recurrence, independent of chunk kernels,
with raw safe-sigmoid gate, raw beta sigmoid and q/k L2 normalization. It
preserves CPU autograd for all seven input/parameter gradient classes.

Small accuracy tolerance: output `0.005+0.005*abs(reference)`, final state
`0.02+0.02*abs(reference)`, dq/dk/dv `0.008+0.008*abs(reference)`, gate/shared
gradients `0.02+0.02*abs(reference)`. These thresholds are fixed in the runner;
do not relax them after failure. Large cases use all-rank staged parity and
must not be described as CPU-golden validation.

Native communication uses exact gather equality and prefix `atol=1e-6,
rtol=1e-5`. Patterns vary by rank/head/coordinate/invocation; E is asymmetric
and M alternates upper/lower perturbations to expose transpose, right multiply
and reversed composition errors. CP2 alone cannot cover prefix GEMM; use CP4.

## Commands

```bash
python -m unittest discover -s tests -p test_kcp_contract.py -v
python examples/cp/check_kcp_installation.py

# Packed staged development baseline. Set visible devices to a healthy CP pair.
HCCL_OP_EXPANSION_MODE=CCU_SCHED HCCL_BUFFSIZE=32 \
torchrun --standalone --nproc_per_node=2 examples/cp/check_kcp_packed.py \
  --tokens 65 --heads 3 --output outputs/packed_cp2.json

ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_BUFFSIZE=32 \
torchrun --standalone --nproc_per_node=2 examples/cp/check_kcp_native.py \
  --heads 3 --output outputs/native_gate

ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_BUFFSIZE=32 \
torchrun --standalone --nproc_per_node=2 examples/cp/benchmark_kda_cp.py \
  --tokens 129 --heads 3 --group-size 2 --mode save --accuracy --backward \
  --accuracy-only --stress-repeats 8 --backends staged,pipeline,mc2 \
  --output outputs/cp_accuracy

# Two disjoint interleaved CP domains: [0,2] and [1,3].
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 HCCL_BUFFSIZE=32 \
torchrun --standalone --nproc_per_node=4 examples/cp/benchmark_kda_cp.py \
  --cp-groups 2 --tokens 65 --heads 3 --group-size 2 --mode save \
  --accuracy --backward --accuracy-only --stress-repeats 8 \
  --backends staged,pipeline,mc2 --output outputs/subgroups

# CPU-only plan: 48 cases, with explicit persistent-resource reservations.
python scripts/run_kcp_matrix.py --runtime-bytes 16777216 \
  --communication-bytes 67108864
```

For A5 framework checks use `HCCL_OP_EXPANSION_MODE=CCU_SCHED` and omit `mc2`
until its native resource gate passes. Compare an initialization failure with
`examples/cp/check_stock_mc2.py`, which invokes stock AllGatherMatmulV2 without
importing fla_npu. A BF16 stock control does not certify FP32 transport.

The packed runner compares the complete local output and one state per local
sequence segment against the independent FP64 recurrence. It covers both group
sizes, including an odd tail, zero-collective aligned shards, and a CP4 case
whose first and last local fragments belong to different global sequences.
Native compute development checks use `examples/cp/kcp_stage0`.

## Performance and coverage

Core shapes `(T,H)`: `(512,8),(2048,16),(8192,32),(16384,64)`; CP2/4/8;
save/none; A2/A5. Use >=20 warmup, 100 iterations and 5 alternating windows.
Report median of per-window maximum-rank means. Collect 5 separate unmarked
profiles and preserve Task Duration plus the timeline; event makespan and
parallel task duration sums are different metrics. Group widths 1/2/4/8/16
must satisfy the temporary-memory ledger including new HCCL resources.

Completed-window JSONL survives later timeouts. Only aggregate a complete
rank/window/iteration set. Version, shape, precision result, profile result,
memory evidence and keep/discard decision belong in the experiment ledger.

Independent no-CP controls run in separate fresh processes, for example:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_BUFFSIZE=32 \
torchrun --standalone --nproc_per_node=2 examples/cp/benchmark_kda_cp.py \
  --tokens 8192 --heads 32 --group-size 2 --mode none \
  --backends staged,pipeline,mc2 --accuracy-only --no-cp-reference \
  --output outputs/nocp_control
```

The independent staged control computes each rank's global prefix in two-head
batches and compares local output and final state at the same fixed thresholds.
It avoids an all-head global input allocation. Candidate backends also compare
to staged; small `--accuracy` cases compare every backend directly to CPU FP64.
`--vary-parameters --flat-bias --backward --random-do` covers nonzero
head/coordinate-dependent gate parameters, flattened bias and nonuniform
BF16 output gradients without changing the default performance inputs.
