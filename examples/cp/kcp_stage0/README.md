# A5 KCP native stage checks

These are direct-launch development checks for CANN on dav-3510. Stage 0 runs
BF16 Q/K normalization, the FP32 log2 gate scan, and beta activation with the
planned packed task mapping and MIX 1:2 structure. It checks lengths 1, 64, 65
and 9 in one launch. The AIC is idle at this stage. Stage 1 uses two AIVs for
BF16 dyadic factors and an AIC for FP32 KK/QK in one MIX kernel with two GM
banks. Stage 2 checks the triangular solve by itself, then checks the local
raw-input Prepare, E/M and nonzero-state forward in one MIX launch. The
local diagnostics share device code with the CP operator.

```sh
source /path/to/cann/set_env.sh
cmake -S examples/cp/kcp_stage0 -B /tmp/kcp-stage-checks \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake"
cmake --build /tmp/kcp-stage-checks -j2
/tmp/kcp-stage-checks/kcp_stage0
/tmp/kcp-stage-checks/kcp_stage1
/tmp/kcp-stage-checks/kcp_stage2
```

Use one healthy visible A5 device. Each program validates the result against
host calculations and exits nonzero on a mismatch. Fixed test buffers include
observability outputs. The Stage 2 state check consumes the same prepared
window from two distinct initial states. The end-to-end CP validation uses
[`check_kcp_packed.py`](../check_kcp_packed.py). See the
[implementation and evidence](../../../fla/ops/ascendc/cp/chunk_kda_cp_fwd/docs/implementation.md).

## Summary32

`kcp_summary32` runs the private A5 K-only preparation and E/M recurrence
without a CP group. It reuses chunk32 normalization, gate factors, rectangular
K/K MMAD, causal mask and triangular inverse. The two 64-token pairs in each
128-token window reuse the original AIV lane mapping. One task handles a
complete local sequence segment and value head.

The payload is row-major FP32 `[local_segments, heads, 128, 256]` with E then
M in each row. Scratch is fixed per MIX group. The runner checks E, M and
`E + M @ H0` against an independent FP64 token recurrence. The default
matrix covers partial chunks, window and slot boundaries, and nonuniform
gates. `--long` adds a sequence of 8193 tokens. Accuracy runs report device
cycles for the first eight 64-token pairs, including K operand packing and
the Cube state update. `--profile` and `--profile-2048` run 32 heads over 513
and 2048 tokens respectively with timing marks disabled.

Using the CANN environment and build directory above:

```sh
cmake --build /tmp/kcp-stage-checks --target kcp_summary32 -j4
/tmp/kcp-stage-checks/kcp_summary32
/tmp/kcp-stage-checks/kcp_summary32 --long
msprof --task-time=on --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization \
  --output=/tmp/kcp-summary32-prof \
  --application="/tmp/kcp-stage-checks/kcp_summary32 --profile"
```

## CP8 IPC entry preview

`ipc_entry.py` builds a private plan for testing one continuous sequence on
eight A5 devices. Launch eight processes on one host with a Gloo group and
bind each logical rank to its physical device before allocating inputs. The
neighbors in logical rank order must be able to import each other's IPC
mailboxes. Gloo exchanges metadata and keys during setup and teardown.

Build the archived A5 IPC variant from the repository root with the CANN
Ascend C toolchain enabled.

```sh
cmake -S fla/ops/ascendc/cp/chunk_kda_cp_fwd/ipc -B build/kcp-ipc
cmake --build build/kcp-ipc --target kcp_ipc_chunk32
```

Pass `build/kcp-ipc/libkcp_ipc_chunk32.so` to the plan constructor and the
plan to the regular forward entry with `backend="fused"` and
`backward_mode="none"`.

```python
options = dict(cp_group=group, A_log=a_log, dt_bias=dt_bias)
plan = create_ipc_preview_plan(q, k, v, g, beta, library=library, **options)
try:
    result = chunk_kda_cp_fwd(
        q, k, v, g, beta, backend="fused", backward_mode="none",
        plan=plan, **options,
    )
    plan._ipc_preview.check_completion()
finally:
    plan._ipc_preview.close()
```

Use the creation stream for all calls. All ranks close the plan collectively
before destroying the bootstrap group. The fixed-size completion buffer lets
the test check all AIV lanes after synchronization. The preview library and the
plan constructor are development artifacts. The regular plan factory and
packaged HCCL dispatch retain their existing interface.

The archived device sources reproduce the `28-affine-resident` configuration.
The A5 CP8 experiment used 32 value heads, K=V=128, and one continuous
sequence. Short FP64 output and final-state checks passed on all eight ranks.
At global lengths 128K and 512K, the maximum-rank forward medians were
4.326 ms and 16.920 ms. The matching PR #866 chunk64 chain with IPC AllGather
took 9.329 ms and 25.133 ms. Those comparisons include different compute and
communication schedules. Native HCCL CP8 did not reach timing on that card
group. The measurements were taken before the sources were archived.
