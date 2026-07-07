# KDA GPU dump dual benchmark (PR #152 base)

NPU `chunk_kda_fwd` vs GPU golden dumps collected from the GPU repo (`feat/kda-gpu-dump`).

## Prerequisites

1. Build PR #152 KDA forward ops on NPU (`chunk_kda_fwd`, `kda_gate_cumsum`, `kda_layout_swap12`, …).
2. Source CANN + `fla_npu` op_api lib (same as other GDN dual tests).
3. GPU dumps under a root directory, one subdir per case (see `gpu/kda_cases.json` names).

## GPU dump collection (GPU machine)

```bash
cd gpu   # flash-linear-attention, branch feat/kda-gpu-dump
./run_kda_dump_cases.sh --dump-dir /data/kda_dump/all --skip-done
```

Each case dir contains `001_chunk_kda_fwd.pt` and `manifest.json`.

## NPU dual run

```bash
export TEST_DEVICE_ID=6
chmod +x fla/ops/ascendc/kda/test/run_kda_gpu_dump_dual.sh

# all cases
./fla/ops/ascendc/kda/test/run_kda_gpu_dump_dual.sh /data/kda_dump/all

# smoke only
./fla/ops/ascendc/kda/test/run_kda_gpu_dump_dual.sh /data/kda_dump/all --phase smoke --no-viz

# single .pt
./fla/ops/ascendc/kda/test/run_kda_gpu_dump_dual.sh /data/kda_dump/all/smoke_mha_fix/001_chunk_kda_fwd.pt
```

Report: `<dump_root>/kda_gpu_dump_dual_report.json`  
Logs: `<dump_root>/logs/kda_gpu_dump_dual_*.log`

## Compared tensors

| Tensor | Compared |
|--------|----------|
| `o` | yes |
| `final_state` | yes (when present in dump and NPU returns it) |
| `g`, `initial_state` (outputs) | **skipped** for now |

## Input adaptation

GPU dump stores raw `g` / `beta` plus kernel flags. Before NPU call:

- `gk` = CPU reference gate cumsum (`safe_gate` + `A_log` / `dt_bias` when `use_gate_in_kernel`)
- `beta` = sigmoid(raw) when `use_beta_sigmoid_in_kernel`
- `q` / `k` are already post-l2norm in the dump
- varlen: pass `cu_seqlens` only; NPU OpApi auto-builds `chunk_indices`

## Known skips

- `gva_t4096_v256` (Vdim=256) is out of PR #152 scope and auto-skipped.

## Branch

`feat/kda-gpu-dump-dual-pr152` — based on [flashserve/flash-linear-attention-npu#152](https://github.com/flashserve/flash-linear-attention-npu/pull/152).
