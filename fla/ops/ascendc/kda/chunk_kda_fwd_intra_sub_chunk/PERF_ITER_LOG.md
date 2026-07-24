# ChunkKdaFwdIntraSubChunk · PERF_ITER_LOG

Model case: `B=1,T=8192,H=HV=32,K=128,BT=64,bf16`  
Acceptance: Task Duration median ≤ 1.5 ms (msprof, idle card).

| knife | change | Dur_med | Δ | precision | default | notes |
|-------|--------|---------|---|-----------|---------|-------|
| B0 | Tile GEMM + Kg L1B hold; W load ‖ MMAD1 (`USE_SCORE_MMAD1_LOAD_W=1`); lockstep CV | **20.323 ms** | — | full suite PASS | on | wall_ms≈21.2; `aiv_scalar_ratio≈0.47`, `aic_mac_ratio≈0.004` |
| B1 | Vectorize FwdSub (Mul/Add over BC; drop O(i²) GetValue) | **7.589 ms** | **−12.734 ms** | full suite PASS | on | `aiv_scalar_ratio≈0.23`, `aiv_vec_ratio≈0.24`; TaskWait≈5.0ms |
| B2 | FwdSub = Triton Mul+axis0-reduce: `Brcb` broadcast → `Mul` → Add-fold col-reduce (chunk_bwd 列求和); not Cube; not Pattern::RA | **4.610 ms** | **−2.979 ms** | full suite PASS | on | `aiv_scalar_ratio≈0.305`, `aiv_vec_ratio≈0.25`; Pattern::RA on 16×16 was ~34.6ms (reject) |
| C0 | Vec 2-win dual-issue: B×NT + dual-AIV-by-head + `prefill=2` + `SetS0ReadyJoined` + SetFree Process bookend (not hot-path WaitFree) | wall **~3.04 ms** | wall vs B2 ~4.6 → **~−1.6** | full suite PASS | on | Task Dur msprof TBD this commit; protocol from VEC_2WIN_PIPE (Barrier before 0x2 Set) |

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

## C0 wall snapshot (NPU4, 2026-07-24)

- wall_ms med≈3.043 (warmup=3, iters=10); shape B=1,T=8192,H=32,K=128,BT=64,bf16
- Formal msprof Task Dur to follow in Phase E

## Next direction (from C0)

1. msprof C0 baseline (Task Dur + pipe ratios); confirm GEMM∥Post
2. Cut remaining **AIV scalar** (Prepare mid-row Sub loop / FwdSub diag +I SetValue)
3. Sync hygiene (drop V_S waits that only gate vector→vector) — validate per-site, no blind strip
4. Re-check Cube after Vector/Wait drop; optional simulator T=1024
5. Borrow ideas from sibling 2.37ms path only as **hypotheses**; direction must come from our msprof
