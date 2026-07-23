# ChunkKdaFwdIntraSubChunk · PERF_ITER_LOG

Model case: `B=1,T=8192,H=HV=32,K=128,BT=64,bf16`  
Acceptance: Task Duration median ≤ 1.5 ms (msprof, idle card).

| knife | change | Dur_med | Δ | precision | default | notes |
|-------|--------|---------|---|-----------|---------|-------|
| B0 | Tile GEMM + Kg L1B hold; W load ‖ MMAD1 (`USE_SCORE_MMAD1_LOAD_W=1`); lockstep CV | **20.323 ms** | — | full suite PASS | on | wall_ms≈21.2; `aiv_scalar_ratio≈0.47`, `aic_mac_ratio≈0.004` |

## B0 msprof snapshot (NPU2, 2026-07-24)

- Task Duration us: med=20322.8, min=20305.9, max=20331.2 (n=13)
- PipeUtilization (one sample): aiv_scalar_ratio=0.468, aiv_vec_ratio=0.019, aic_mac_ratio=0.004, aic_mte2_ratio=0.011
- Prof dir: `/tmp/prof_intra_tile_base/PROF_000001_20260724010153247_00669373RMAQGBON`

## Next direction (from B0 image, not PLAN.md)

1. Cut **AIV scalar** (FwdSub / Prepare scalar loops → vector Mul+ReduceSum / fewer GetValue)
2. Then CV overlap (2-window SetFree prefill) once Vector path is cheaper
3. Re-check Cube occupancy after Vector shrinks
