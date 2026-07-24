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

## P0 precision fix (landed, 2026-07-24)

Was: intermittent H=32 `aqk_err≈7–13` / aicore timeout (`fixp_error`).

| change | result |
|--------|--------|
| `USE_SCORE_MMAD1_LOAD_W` **default 0** (serial W load) | stops single-L1A overlap corruption |
| AIC: only `PipeBarrier<PIPE_FIX>` before `SetCubeDone` (no extra `PIPE_ALL`) | avoid fixp trap under back-to-back runs |
| Drop `PIPE_ALL` after Akk Fix (sibling `DROP_PIPE_ALL`) | FIX Wait enough |
| Vec2Win slot comment clarified (`WaitCube⇒bank free`) | protocol unchanged |

**Verify:** full suite PASS (incl. H32 T=4096/8192); same-proc H32×5 all `aqk_max_err=0.0137`.

Double-buffer (L1A[2]) still **deferred** — `SCORE_TILE_DBUF_PLAN.md`.

## P1 L1A dbuf (landed, 2026-07-24)

| knife | change | Dur_med | precision | default | notes |
|-------|--------|---------|-----------|---------|-------|
| P1 | `USE_SCORE_L1A_DBUF=1`: l1A[0]=Qg, l1A[1]=W; MTE2(W)‖MMAD1; Wait W before Fix | wall **2.971 ms** / Task Dur **~2.172–2.183 ms** | suite + H32×5 PASS | on | vs C0 wall ~3.04 → **~−0.07**; forces MMAD1_LOAD_W=0 |

## C1 / C2 Cube Fix‖MTE2 & WIN L1 (2026-07-24)

| knife | change | Dur_med | precision | default | notes |
|-------|--------|---------|-----------|---------|-------|
| C1 | `USE_SCORE_FIX_MTE2_DBUF=1`: Akk Fix ‖ next-tile MTE2; Drain before SetCubeDone | **2.180 ms** | full suite PASS (clean rebuild) | **on** | vs P1 **2.172** Δ≈+0.008（Dur 门禁未过）；sim tick 252571→225474 |
| C2 | `USE_SCORE_WIN_L1_RESIDENT` Prefetch 双头 | — | **FAIL** `aqk_err≈14` | **off** | 削弱 P1/C1；代码保留。理论最优见 `CUBE_OPTIMAL_PIPELINE` **路径 A** |

## Next direction

1. **主差距在 AIV**（Task Dur ~2.18 → 1.5，差 ~0.68 ms）：scalar / Post·Prep / CV Wait，用 msprof 定刀，不要再堆无 Dur 收益的 Cube 微重叠
2. P2 L0[2]：仅当 sim 明确 L1→L0 bubble；sibling 门禁曾失败
3. C2 resident：精度修好前不 default；即便修好也优先验证是否伤 P1/C1
