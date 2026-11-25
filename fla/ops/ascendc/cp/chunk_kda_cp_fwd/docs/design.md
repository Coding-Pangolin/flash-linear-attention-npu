# KDA CP whole-forward design

方案设计规则版本：V2。Public semantics are defined only in [api.md](api.md).

## 1. Goal and evidence

Basis: fla-npu 134524a6 (critical files checked unchanged at b797b1aa), CP
example f9cc41fa, PR #833 5ce125df, ops-transformer 87663d62, CannBot 5fd02381,
cannbot-skills ce380702. References are technical evidence, not replacements
for this repository's development gates. Preserve all BF16 rounding points.
In particular, FP32 summary output does not imply FP32 M-chain operands.

First deliver a runnable strong baseline, then a registered MC2 boundary,
then expand fusion only after stage correctness and measured feasibility.
A2 uses AICPU service; A5 compares CCU and AICPU. AIV/URMA is not the default:
KDA already requires Vector work and its Prepare consumes the available UB.

Compare: B0 full-head staged; B1 grouped two-stream framework pipeline; B2 native
AllGather+merge; B3 whole-forward native scheduling. A PTP true-state wavefront
is an algorithmic control because it eliminates M-chain work but adds rank
dependencies. No unmeasured candidate is called optimal.

## 2. Stages, resources and scheduling

Let H be local heads, T local tokens, P CP size, C=64, K=V=128, G head-group
width, L=ceil(H/G). Rank r owns global tokens [r*T,(r+1)*T).

| Stage | Unit | Work and output | Lifetime |
|---|---|---|---|
| P | AIC/AIV | Existing Prepare; group-local gk/Aqk/w/u/kg/qgs and save caches | Until S and F/O finish |
| S | AIC/AIV | Existing local forward summary `[G,128,256]` | Until collective completion |
| G | AICPU or CCU | AllGather summaries in group-rank order | Until M completes |
| M | AIC/AIV | Ordered exclusive prefix, producing FP32 h0 | Until F has loaded it; retain for diagnostics/training |
| F | AIC/AIV | Existing chunk state recurrence, BF16 H/V_new | Until O and optional backward |
| O | AIC or MIX | `O=bf16(qgs @ H + Aqk @ V_new)` | User output |

P's V0/V1/C2/V3/C4/C5/V6/C7 and F's Cube/Vector stages retain their existing
operation order, operand rounding and valid-token masks. M applies earlier
rank summaries only: H=E0; H=M_i @ H+E_i. Rank 0 uses zero h0 but participates
in all collectives. This operation is not SUM AllReduce.

### Framework pipeline baseline (implemented before native work)

Use one caller compute stream and one dedicated communication stream:

```
produce group 0; record produced[0]
comm stream: wait produced[0]; allgather(0); record received[0]
for g=1..L-1:
    produce group g; record produced[g]
    comm stream: wait produced[g]; allgather(g); record received[g]
    compute stream: wait received[g-1]; merge/consume(g-1)
compute stream: wait received[L-1]; merge/consume(L-1)
```

Collectives are posted in the same order on all ranks. Each receive buffer is
separate; do not overwrite a slot protected only by host return. Every tensor
used on the other stream is recorded on that stream. A group's Python objects
remain alive through enqueue of its last consumer; allocator stream tracking
protects the actual asynchronous lifetime. No per-group device synchronize.
The caller's stream waits on the final communication event before return.

The baseline deliberately uses a GPU/NPU event dependency, not busy-wait
producer flags. A producer must never depend on a collective that depends on
that producer. Compute and communication contention is measured, not assumed
absent. The native boundary will first use a bounded compute footprint; no
cross-block wait may require an unscheduled producer.

### Native expansion gate

First verify registered FP32 AllGather with known per-rank patterns. Then reuse
the existing ordered merge without changing its FP32 recurrence. Delayed Commit
must follow the full local summary write/visibility fence; repeat/main/tail
counts and Wait counts must match. Finalize occurs after all participating
cores and AIV writes have drained. A2 cannot use the uninstalled preview host
MC2 initialization APIs as if they were present.

The registered boundary's address map and flags are in
[`kcp_all_gather_merge/docs/design.md`](../../kcp_all_gather_merge/docs/design.md).
The current B3 is whole-forward scheduling of that boundary and the unchanged
local kernels. It is not a single kernel containing Prepare/FwdH/Finalize.
Only after native communication succeeds can further local fusion redefine the absolute UB/L1/L0 address map and flag
allocation against actual compiled components. Existing A5 Prepare occupies
248 KiB UB; its live addresses cannot coexist with an independently allocated
communication buffer. Phase aliasing requires a complete drain, not a flag
rename. Further local kernel fusion is not implemented or advertised before
its own gate passes. A one-group case stays on the caller stream, avoiding
cross-stream events that cannot overlap any useful work.

### Memory ledger

Each summary head is 131072 bytes; each FP32 state head is 65536 bytes.
Group inference Prepare data is approximately T*G*(12*K+2*C) bytes before
alignment. Two in-flight groups at T=8192,G=4 need 104 MiB for these tensors.
Native scratch, runtime workspace, windows and receive buffers are additional.
The planner must reject estimates over 200000000 bytes; actual peak is still
required. Training cache is reported separately and compared under equal
output contracts. Baseline F/O still materialize group-local H/V_new and must
budget those too; savings from a future F/O kernel cannot be assumed now.
The package's `_kcp_resources.py` is the common CPU accounting source for the
runtime guard and experiment planner. It supplies static rejection estimates;
unknown persistent communication allocations cannot be certified by it.
Save mode separates user-owned prepared caches/H/V_new/h0 from scratch; only
u/qgs are transient Prepare outputs. Tail groups and local-to-global output
copies are included. A single group returns the existing Finalize/FwdH outputs
directly, avoiding duplicate result allocation. Source inputs retained as views
are not counted as new cache allocations. Alignment and concurrent opaque
runtime workspaces still require a measured allocation inventory.

### Verification and selection

Core cases: (T,H)=(512,8),(2048,16),(8192,32),(16384,64), P=2/4/8, save/none,
A2/A5. CPU oracle checks small independent recurrence; larger runs compare
all ranks to B0 and head-batched no-CP reference. Saved caches must feed the
existing explicit CP backward before training support is accepted.

Performance: >=20 warmup, 100 iterations/window, >=5 alternating A/B windows;
report per-window max-rank mean and median across windows. Five separate
unmarked profiling runs preserve full task data. Instrumented timestamps are
diagnostic only. Multi-stream duration is joined makespan, not task sums.
Candidates must beat max(3%, measured noise); unsupported/unvalidated native
backends fail explicitly. Lifecycle and kernel correctness gates precede all
performance-based selection.

The benchmark stores per-rank completed-window JSONL checkpoints, raw event
records, maximum-rank window summaries, input seed, versions, source hash and
communication environment. A timeout is an incomplete experiment, never a
partial performance pass. Profile Task Duration separately; do not sum parallel
tasks and call the result end-to-end latency. When using msprof with a venv,
launch a bash wrapper that execs the venv Python: this msprof version resolves
a Python symlink to its base interpreter and otherwise loses the venv.
