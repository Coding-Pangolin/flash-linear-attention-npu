# ChunkKdaFwdIntraSubChunk · PERF_ITER_LOG

Model case: `B=1,T=8192,H=HV=32,K=128,BT=64,bf16`  
Acceptance: Task Duration median ≤ 1.5 ms (msprof, idle card).

| knife | change | Dur_med | Δ | precision | default | notes |
|-------|--------|---------|---|-----------|---------|-------|
| B0 | Tile GEMM + Kg L1B hold; W load ‖ MMAD1 (`USE_SCORE_MMAD1_LOAD_W=1`); lockstep CV | **20.323 ms** | — | full suite PASS | on | wall_ms≈21.2; `aiv_scalar_ratio≈0.47`, `aic_mac_ratio≈0.004` |
| B1 | Vectorize FwdSub (Mul/Add over BC; drop O(i²) GetValue) | **7.589 ms** | **−12.734 ms** | full suite PASS | on | `aiv_scalar_ratio≈0.23`, `aiv_vec_ratio≈0.24`; TaskWait≈5.0ms |
| B2 | FwdSub = Triton Mul+axis0-reduce: `Brcb` broadcast → `Mul` → Add-fold col-reduce (chunk_bwd 列求和); not Cube; not Pattern::RA | **4.610 ms** | **−2.979 ms** | full suite PASS | on | `aiv_scalar_ratio≈0.305`, `aiv_vec_ratio≈0.25`; Pattern::RA on 16×16 was ~34.6ms (reject) |

## B0 msprof snapshot (NPU2, 2026-07-24)

- Task Duration us: med=20322.8, min=20305.9, max=20331.2 (n=13)
- PipeUtilization (one sample): aiv_scalar_ratio=0.468, aiv_vec_ratio=0.019, aic_mac_ratio=0.004, aic_mte2_ratio=0.011
- Prof dir: `/tmp/prof_intra_tile_base/PROF_000001_20260724010153247_00669373RMAQGBON`

## B1 msprof snapshot (NPU2, 2026-07-24)

- Task Duration us: med=7589, min=7577, max=7598 (n=13)
- PipeUtilization: aiv_scalar_ratio=0.227, aiv_vec_ratio=0.241, aic_mac_ratio=0.01
- Prof dir: `/tmp/prof_intra_fwdsub/PROF_000001_20260724010500108_00680493CCJNMIBK`

## B2 msprof snapshot (NPU2, 2026-07-24)

- Task Duration us: med=4610.0, min=4594.9, max=4631.8 (n=25)
- PipeUtilization: aiv_scalar_ratio≈0.305, aiv_vec_ratio≈0.25
- Prof dir: `/tmp/prof_intra_fwdsub_mulfold/PROF_000001_20260724012056702_00761696OLINGLMH`

## Next direction (from B2 image)

1. Cut remaining **AIV scalar** (Prepare gate / diag +I)
2. Shrink **Task Wait** + CV overlap (2-window SetFree prefill; dual AIV heads)
3. Re-check Cube after Vector/Wait drop
