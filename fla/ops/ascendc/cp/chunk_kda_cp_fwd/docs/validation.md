# Development evidence — 2026-11-17

Experimental implementation, **not full acceptance**. Native A2 transport,
ordered prefix and whole-forward correctness have development evidence.
No reproducible whole-forward speedup or verified compute/transport overlap
is claimed. `staged` remains the default. Keep this file until acceptance is
complete under `docs/agents/05-算子测试.md`.

## Versions and scope

Base `134524a6aae55bc94101575146e88511856716e5`; isolated branch
`feature/kcp-full-forward`. Critical upstream sources were checked at b797b1aa.
A2 runs use Ascend910B3, torch 2.7.1, torch-npu 2.7.1.post8, CANN 9.2 and a
separate CANN 9.3 weekly environment. A5 development runs use torch 2.11,
torch-npu 2.11.0.post3 and installed CANN 9.2 beta/formal B237 metadata.
Only healthy idle devices are selected; other users' jobs are not stopped.
Server identities, exact environments, wheel hashes, commands and raw records
are retained in private experiment artifacts, not in public examples.

The build selects all ten KDA CP dependencies into one self-contained wheel,
including the Stable launcher and native OPP. It does not overlay a native-only
OPP onto an unrelated wheel. This filtered wheel does not contain every other
FLA operator: the repository-wide wheel check expects recompute_w_u_fwd and
therefore cannot certify that filtered installation as a full-project package.

## Correctness evidence

| Gate | Evidence | Limit |
|---|---|---|
| CPU oracle | Six contract tests pass: independent FP64 recurrence and gradient continuity, noncommuting affine prefix/orientation, tail drain, resource ownership, complete-window statistics | Does not replace device tests |
| A2 CP2 save/none | T65/H2 and T129/H3/G2; all-rank output/state checks; save-mode seven CPU gradients; eight alternating-value invocations | Small numerical calibration cases |
| A2 CP4 native | H16, three invocations, exact FP32 gather, asymmetric E and upper/lower noncommuting M, ordered exclusive prefix on all ranks | CP4 covers actual prefix GEMM; CP8 open |
| A2 CP4 whole forward | T65/H3/G2 save, staged/pipeline/mc2, CPU output/state/seven gradients, eight alternating-value invocations | Full core matrix still separate |
| Non-WORLD domains | Two interleaved CP2 groups [0,2] and [1,3], T65/H3/G2 save, all three backends and CPU gradients | Communicator query uses global rank; prefix uses group-local rank |
| A2 boundaries | CP1 T65/H3 save; CP2 T1/H1 save/none; no bias and no final-state result; explicit backward where saved | CP1 skips communication and summary |
| Collective DFX | Forward mismatched width, one-rank dtype and requires-grad; backward one-rank dO dtype/device/requires-grad are rejected on all ranks | Caller must use the matching cache/validated plan on every rank |
| Native DFX | BF16 summary, wrong h shape, rank outside P, empty communicator name rejected before data communication | Formal ACLNN/GE negative suite still open |
| A5 framework | CCU_SCHED CP2 T65/H3/G2 save with CPU output/state/seven gradients and eight-value stress; none path also passes small CPU check | Native and CP4/8 domains not accepted |
| Nonuniform backward inputs | Final A2 CP2/4 and A5 CP2 wheels pass seven CPU gradient comparisons with nonuniform BF16 dO, nonzero per-head A_log and coordinate-dependent flattened bias | Fixed tolerances; no autograd wrapper implied |
| Large independent controls | Eight core shapes/modes each on A2 CP2/4 and A5 CP2: head-batched no-CP prefixes match staged local output/state; candidates also match staged in the performance matrix | These are separate accuracy processes, not timing runs |

Thresholds are fixed in the distributed runner and documented in
`tests/atk/chunk_kda_cp_fwd/README.md`; no tolerance was relaxed after failure.
The original public Stable wrappers' 41 function signatures are unchanged.
Twenty-two Stable-only runtime regressions pass. ABI ownership/schema and
coverage gates must remain green alongside final wheel installation checks.
The final filtered A2/A5 wheels pass the scoped ten-config installation check,
39 actual registrations, native public/export isolation and ELF Stable-ABI audit.
The host suite reports 91 passed and one locally-unbuilt hash-module skip across 92 tests;
the issue map covers 188 completed issues and the static review has no findings.

## A5 communication blocker

The A5 native CCU kernel compiles. Runtime allocation fails in
`HcclAllocComResourceByTiling` / `rtUbDevQueryInfo` before kernel execution.
The stock CANN BF16 AllGatherMatmulV2 control with `comm_mode="ccu"`, without
fla_npu, reproduces that resource-registration error on the same machine.
A final control disables the FLA .pth hook and removes custom OPP/library
environment paths; its own driver logs again report rtUbDevQueryInfo 107000
and HcclAllocComResourceByTilingV2 error 15.
Changing the allocator's expandable-segment setting did not resolve it.
Another A5 machine fails even earlier during HCCL initialization with
`rankLevelInfos size is zero` / RootInfoDetect. These are external environment
blockers, not proof that FP32 transport is unsupported. AI_CPU framework init
also failed on the tested A5 stack, while framework CCU_SCHED succeeds.
The aggregate `mc2` API rejects A5 rather than silently substituting a backend.

## Candidate ledger and performance interpretation

Every accepted timing window uses 20 warmups, 100 iterations and five
alternating windows; it retains each rank and reports the median of maximum
rank means. Event duration joins the communication stream. Task Duration is
collected separately. Instrumented runs are excluded from timing selection.

| Candidate | Observed development result | Decision |
|---|---|---|
| Grouped G4, old orchestration | A2 CP2 T2048/H16 none: staged 1616.5 us, pipeline 6550.4 us, native 6451.8 us; large window spread/shared-host effects | Reject as a speedup; preserve raw windows |
| Larger T8192/H32/G2, old orchestration | none: staged 6033.4 us, native 34942.9 us | Reject; grouping/dispatch is costly |
| Single-group caller stream | T512/H8/G8 none: staged 1584.2 us, native 1556.9 us; T2048/H16/G16 none: 1902.8/1880.5 us | Below 3% and observed noise; do not promote |
| Save-mode single group | T2048/H16/G16 staged 1917.3 us, native 2058.8 us | No improvement |
| Final candidate | Output adoption, corrected cache ownership, collective guards; final windows/profiles retained independently | Never merge results across revisions |

These earlier figures are rejected pilot evidence, not final-version
performance claims. A previous T16384 run timed out; its incomplete records
are not accepted as a timing result. The current runner preserves per-rank
completed-window JSONL and uses a separate 1800-second benchmark timeout.

The completed core timing slices cover **24 of 48 cases**: eight shapes/modes
each for A2 CP2, A2 CP4 and A5 CP2. Every case includes all-rank staged parity,
20 warmups, 100 iterations and five windows. A2 compares all three backends;
A5 compares staged/pipeline. Examples from these completed slices:

| Platform/domain | T/H/mode/G | Staged us | Pipeline us | MC2 us |
|---|---|---:|---:|---:|---:|
| A2 CP2 | 8192/32/none/2 | 6079.5 | 35225.7 | 35109.5 |
| A2 CP2 | 16384/64/none/1 | 19665.7 | 269258.4 | 268714.3 |
| A2 CP2 | 16384/64/save/4 | 20129.7 | 76149.6 | 76200.2 |
| A2 CP4 | 512/8/none/8 | 1859.4 | 1877.8 | 1744.3 |
| A2 CP4 | 16384/64/none/1 | 19752.7 | 270531.3 | 269175.7 |
| A5 CP2 | 16384/64/none/1 | 10007.3 | 137222.8 | gated |
| A5 CP2 | 16384/64/save/4 | 10741.5 | 37129.5 | gated |

The CP4 small-case improvement is not promoted: staged/native window spreads
are 12.6%/5.6%, larger than a reliable improvement margin. Large grouped cases
are consistently slower. The memory planner chose these group widths with
explicit 32 MiB runtime and 64 MiB communication reservations; those are
planning inputs, not certified actual resource bounds.

One initial unmarked staged profile at T512/H8 reported rank0 Prepare 57.42,
summary 78.44, FwdH 88.11, Finalize 8.47 us mean Task Duration over the last
100 calls. HCOM task duration varied strongly by rank; it includes arrival
skew and cannot be treated as pure link bandwidth. Native kernel residence
overlap with Prepare alone does not establish transport overlap.

Fifteen further unmarked CP4 profiles are complete: staged/pipeline/mc2,
five independent launches each, T512/H8/none/G4. All 60 device reports retain
100 forward iterations (100 tasks per stage for staged, 200 for grouped).
Five-run median Prepare/Summary/FwdH/Finalize task means on rank 0 are
57.39/77.44/87.94/8.38 us for staged and 38.57/71.69/85.03/6.97 us per
group for MC2. Grouping doubles the latter task counts. MC2 boundary residence
varies sharply by rank (five-run medians 225.27 to 1766.63 us). SDMA SQE
intersections with Prepare exist but remain small and asymmetric; these
intervals are not a payload-bandwidth or effective-hiding measurement.
Raw profiles and exact task/timeline analysis are preserved independently.

## Memory and tooling

The shared resource planner separates returned training cache from temporary
u/qgs, communication buffers, tails and grouped result copies. It is static
accounting, not a 200 MB guarantee. A2 CP4/H16 native workspace is 19,922,944
bytes. Exact gather output is 8,388,608 bytes, and the observed driver increase
varies by rank beyond the torch allocator change.

Cold whole-forward diagnostics record own-PID driver usage, torch allocated,
reserved, peak, and unique newly-owned output/cache storage separately. For
A2 CP2 T16384/H64/G1 none, initial torch allocation is 1,346,407,936 bytes,
peak 1,738,199,040, and returned result storage 272,629,760. Do not report
that whole peak as temporary storage or subtract driver/allocator figures
twice. Prepare/summary/FwdH/Finalize have additional runtime workspaces;
new persistent HCCL resources still need attribution before the 200 MB gate.

The first four-rank mssanitizer memcheck attempt was terminated at its
60-second bound after the tool reported missing HCCL built-in kernel JSON
and interaction failures. It produced no accepted memory-check result.
Workers were cleaned up and the selected devices returned idle.

One large A2 no-CP-control attempt exceeded the 60-second experiment bound
after reaching backend comparisons; it is retained as incomplete. Subsequent
verification compares every output element in head batches with unchanged
tolerances. Completed independent baseline controls and the separate complete
candidate-parity matrices supply the large-case evidence; the timeout is not
reclassified as a pass.

## Open acceptance work

- Working A5 native resource registration, then native CP2/4/8 and all-rank
  accuracy/backward validation; matching whole-node idle windows for CP8.
- Complete the remaining 24 core performance/accuracy cases and their
  independent controls once those domains are available.
- Extend the existing five-launch-per-backend profiling beyond T512/CP4;
  finish HCCL payload attribution and new-allocation accounting under 200 MB.
- Formal distributed ATK integration, successful memory/race/synchronization
  tooling and final broad packaging/GE acceptance. Development torchrun checks
  are not described as formal ATK passes.
- Further physical fusion of local Prepare/FwdH/Finalize requires a new
  measured case, address map and correctness gate; no single whole-forward
  Ascend C kernel is advertised by this version.
