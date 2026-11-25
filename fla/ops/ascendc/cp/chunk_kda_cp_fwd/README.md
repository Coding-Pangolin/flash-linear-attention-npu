# KDA context-parallel forward

Experimental implementation of the design for issue #912. The public contract is
in [docs/api.md](docs/api.md), scheduling and resource contracts in
[docs/design.md](docs/design.md), and actual validation evidence in
[docs/validation.md](docs/validation.md).

The `staged` and `pipeline` paths are framework baselines. They must not be
reported as an in-kernel MC2 implementation. Native communication is accepted
only after the registered operator has been built and verified on each target.

`backend="fused"` runs raw-input continuous packed CP forward in one user
kernel per A5 rank. CP2 has accuracy evidence for zero, one and two data
AllGather rounds. Use `backward_mode="none"` and the optional CPU
`cu_seqlens` describing the global packed boundaries. The existing `staged`
default and explicit backward remain available. See [implementation.md](docs/implementation.md)
for the measured cases, scheduling and remaining performance work.

```bash
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=chunk_kda_cp_fwd
# Install this OPP and a Stable ABI launcher built from the matching source.
ASCEND_RT_VISIBLE_DEVICES=1,2 HCCL_OP_EXPANSION_MODE=CCU_SCHED HCCL_BUFFSIZE=32 \
torchrun --standalone --nproc_per_node=2 examples/cp/check_kcp_packed.py \
  --backend fused --tokens 129 --heads 32 --output outputs/kcp_fused_cp2.json
```

Build a self-contained wheel with all forward and explicit-backward dependencies
after preparing the CANN/build environment described in the repository README:

```bash
export FLA_NPU_SOC=ascend910b  # ascend950 for A5
export FLA_NPU_OPS=kcp_all_gather_merge,chunk_kda_fwd_prepare,pre_process_fwd_kernel_merged,merge_fwd_bwd_kernel,chunk_fwd_h,chunk_kda_fwd_finalize,chunk_kda_bwd_prepare,pre_process_bwd_kernel_merged,chunk_gated_delta_rule_bwd_dhu,chunk_kda_bwd_finalize
python -m pip wheel -v --no-build-isolation --no-deps . -w dist_kcp
# Install the matching host architecture/tier wheel in a private test environment.
python -m pip install --force-reinstall --no-deps dist_kcp/*.whl
python examples/cp/check_kcp_installation.py
```

The scoped installation check verifies all ten OPP configs, dependency symbols,
39 Stable registrations and the launcher build stamp. It does not replace the
full-project packaging gate for a wheel containing every FLA operator.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_BUFFSIZE=32 \
torchrun --standalone --nproc_per_node=2 examples/cp/benchmark_kda_cp.py \
  --tokens 65 --heads 3 --group-size 2 --mode save --accuracy --backward \
  --accuracy-only --stress-repeats 8 --backends staged,pipeline,mc2 \
  --output outputs/kcp_accuracy
```

Use healthy idle devices and a fresh output directory. A2 native CP2/CP4 and
non-WORLD subgroup validation are recorded; CP8, full ATK/sanitizer acceptance,
the complete 48-case matrix and final resource accounting remain open.

For a separate large accuracy control, use `--accuracy-only --no-cp-reference`
with `staged` first in `--backends`. This gathers a small head batch at a time
and compares the CP baseline to ordinary no-CP computation on each rank's
global prefix. Candidates also match that baseline. Singleton communicator
and reference allocations are outside performance experiments. Small cases
can additionally use `--accuracy --vary-parameters --flat-bias --backward`.
