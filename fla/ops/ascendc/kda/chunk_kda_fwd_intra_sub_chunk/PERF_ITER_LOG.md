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

## V-A Vector barrier hygiene (2026-07-24)

| knife | change | Dur_med | Δ | precision | default | notes |
|-------|--------|---------|---|-----------|---------|-------|
| V-A | Prep/Tril/FwdSub/`Clamp*` 合并冗余 `PipeBarrier<PIPE_V>` | **2.159 ms** | **≈−0.021** vs pathA ~2.18 | full suite PASS | on（无宏） | 裸 `msprof` `/tmp/prof_va_barrier` n=8；**未过 −0.05 门禁**，板端不劣化，作卫生刀保留 |

门禁采集：裸 `msprof`（勿 `msprof … -- python`，`--` 会吞掉 interpreter → App EPERM）。见 `MSPROF_GUIDE.md`。

## V-B / V-C / V-D (2026-07-24)

| knife | change | Dur_med | Δ | precision | default | notes |
|-------|--------|---------|---|-----------|---------|-------|
| V-B v1 | 错模板列广播 / 无 barrier 直乘 | — | — | **hang / aicore timeout** | **off** | 已否决 |
| V-C | `USE_MTE2_MERGE=1`：S0 mid‖qkg；Post cmat‖beta 单次 Wait | **2.158 ms** | **≈−0.001** vs V-A | full suite PASS | **on** | 预期中低收益；未过 −0.05，低风险保留 |
| V-D | `USE_POST_S0_MTE_OVERLAP` 延后 Post MTE3 Wait | wall≈2.84（C+D） | — | suite PASS；**裸 msprof hang** | **off** | 墙钟可跑；profiler 下 HardEvent defer 挂；代码保留 |

V-C 产物：`/tmp/prof_vc` n=8。

## V-B retry ScaleRowsByBeta (2026-07-25)

对齐 `chunk_kda_fwd::ScaleRowsByBeta`：`Mul(..., {1,1,0,rowBlk,rowBlk,1})` + **每趟 PipeBarrier**；去掉 brcd 铺砖。

| 项 | 结果 |
|----|------|
| 精度 | 全量 suite **PASS**（slim=1 一次） |
| 板端裸 msprof | **hang**（timeout 120s，无 op_summary）→ **Dur 门禁未采到** |
| 板端多 iter | 间歇 hang（iter0 OK / iter1 挂；另卡上曾 6 iter 全过）→ **不稳定** |
| 默认 | **`USE_FWDSUB_SLIM=0`**（代码路径保留） |

仿真 `prof_msprof_op_sim_t1024_vb`（slim=1 wheel）vs `l1a` 基线：

| 指标 | l1a (P1) | V-B slim | 变化 |
|------|----------|----------|------|
| Total tick | 252571 | **157806** | −37% |
| UB2UB % | 7.6% | **0.4%** | 目标达成 |
| BAR % | 60.7% | 84.8% | 占比升（每 col Mul 加 barrier；总 cycle 降） |
| vec duration med | ~20.8 µs | ~18.5 µs | 仿真核时下降 |

结论：算法方向正确（sim UB2UB/tick 明显降），但板端稳定性与 msprof 门禁未过，**不 default on**。下一步需查 HardEvent/重复 launch 竞态后再开。

## Sim baseline: V-A+V-C default (2026-07-25)

重跑当前默认（P1+C1+V-A+V-C，`USE_FWDSUB_SLIM=0`）`msprof op simulator`：

| shape | Total tick | vec_med | cube_med | BAR% | UB2UB% | MOVEMASK% | VADD+VMUL% |
|-------|------------|---------|----------|------|--------|-----------|------------|
| T=1024 H=2 | **157825** | 19.56 µs | 14.91 µs | 59.7% | 8.4% | 5.2% | 2.5% |
| T=2048 H=2 | **279561** | 37.44 µs | 33.21 µs | 60.5% | 8.6% | 5.3% | 2.6% |
| T=1024 l1a (旧 P1) | 252571 | 20.6 µs | 15.67 µs | 60.7% | 7.6% | 4.8% | 2.3% |
| T=1024 V-B slim | 157806 | 18.67 µs | 14.43 µs | 84.8% | **0.4%** | 2.6% | 1.7% |

产物：`prof_msprof_op_sim_t1024_va_vc`、`prof_msprof_op_sim_t2048_va_vc`。

结论：

1. **V-A/V-C 几乎没改 Vector 结构**（相对 l1a：UB2UB 绝对 cycle 相同；BAR% 60.7→59.7）。板端 ~2.16 ms 上 V-A/V-C 无感，与 sim 一致。
2. **Total tick 相对 l1a 大降主要来自 Cube C1**，不是 Vector knife。
3. **T=2048 与 T=1024 画像同构**（BAR/UB2UB/MOVEMASK 占比几乎不变），拉长 T 不会暴露新热点。
4. Vector 可优化点排序：**BAR 墙（~60%）> UB2UB/brcd（~8.5%）> MOVEMASK/scalar（~5%+13%）≫ 真 VECTOR 算子（~2.5%）≫ MTE（~5%，V-C 已尽）**。
5. V-B 仍是唯一在 sim 上砍掉 UB2UB 的刀，但每 col `PipeBarrier` 把 BAR cycle 抬高；板端不稳 → 下一步应 **稳 V-B（少 barrier / mask=16）**，而不是继续 V-A/V-C 微合并。

## P1 coarse Mul sync (2026-07-25)

依赖推导：消灭 brcd UB2UB；行广播 `prod[p,c]=akk[p,c]*a[p]`；**两趟 col-tile Mul 连发，Add-fold 前一次 `PipeBarrier`**（不再 per-col barrier）。

| 项 | 结果 |
|----|------|
| 精度 | 全量 suite **PASS**；H32×6 multi-iter **OK**（无 hang） |
| 板端裸 msprof | Task Dur med **2.075 ms**（n=4，min/max 2.071–2.078）；vs V-C **2.158** → **Δ≈−0.083 ms**（过 −0.05 门禁） |
| 默认 | **`USE_FWDSUB_SLIM=1`** |
| 残留风险 | 同次 msprof 末尾曾 `aicore timeout`；复采偶发无 device dump。无 profiler 多 iter 稳定 |

仿真 T=1024 vs VA+VC / per-col slim：

| 指标 | VA+VC (brcd) | per-col slim | **P1 coarse** |
|------|--------------|--------------|---------------|
| Total tick | 157825 | 157806 | **97847** |
| vec_med | 19.56 µs | 18.67 µs | **18.12 µs** |
| UB2UB% | 8.4% | 0.4% | **0.4%** |
| BAR call | 23264 | 23264 | **21472** |
| BAR cyc/call | 308 | 804 | 744（仍高，但次数↓ + UB2UB↓ → 墙钟/tick 降） |

产物：`prof_msprof_op_sim_t1024_p1_coarse`；板端 `/tmp/prof_p1_coarse`。

## Next direction

1. **主差距仍在 AIV**（~2.08 → 1.5）：P2 评估 Add-fold（默认不换 RA/ReduceSum；可探针双缓冲）
2. P3：Prep / tril mask / `+I`（按需）
3. 观察 slim=1 在裸 msprof 下的偶发 timeout；勿与 V-D 同开
4. C2 resident：精度修好前不 default
