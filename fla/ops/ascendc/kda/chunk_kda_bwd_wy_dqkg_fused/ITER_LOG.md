# ChunkKdaBwdWyDqkgFused — 迭代留档

> Shape：`B1 H=HV=32 T8192 K128 V128 BT64 bf16`，`state_v_first=false`  
> 指标：裸 `msprof op` → MIX_AIC **Task Duration med**  
> 门禁：suite 全绿 + Δ ≤ **−0.05 ms** 才 default on  
> 硬目标（stretch）：≤ **0.8 ms**

约定：**每刀有效优化 → 更新本表 + git commit**（不含 `dist/` / opp 产物）。

---

## 1. 年表

| 阶段 | med ms | Δ vs 前 | 宏 default | 备注 |
|------|--------|---------|------------|------|
| C0 早期基线 | ~29–32 | — | — | 单 head / 调度未成组 |
| 双 AIV 半行修好 | ~22.0 | — | — | contig split，无 Set Barrier |
| C1（I1+I2） | **21.48** | −0.52 vs ~22 | `L1_A_RESIDENT` `STAGE1_L0C_ACCUM` | Cube 仍等 Vec |
| V1–V5 / N1–N3 | **7.66** | vs C1 −13.8 | 见下行宏 | Epilog fold 为大头 |
| **I5 窗 Prefill** | **7.26** | **−0.40** | `WIN_SOFT_LEAD=1` `PREFILL=2` | suite 绿；复测 7259 µs |
| I5b Post≻WaitFree | skip | — | — | 推迟 Cube S0(w+2) 会堵 Vec（AIV-bound） |
| I6 Kg∥Gate 交织 | reject | **+0.06** | `KG_GATE_INTERLEAVE=0` | 7333 µs |
| I4a FIX∥MTE2 | parked | — | `FIX_MTE2=0` | model FIXP/L0C ECC；已修 evt=14 + outstanding 状态机 |
| I4b L0 dbuf | parked | — | `L0_AB_DBUF=0` | 与 I4a 同开曾 ECC |
| Epilog kPark 复用 | reject | **+6** | — | 13348 µs；已回滚 |
| VS0 每窗一次 | reject | **+0.12** | `VS0_ONCE=0` | 7375 µs |
| **Epilog state panel** | **done** | **−1.29** | fold 内 | **5.97 ms**；[K,V] 整面板 CopyStrided |
| EXP_GN_PARK | reject | **+0.06** | `=0` | 6029 µs；Exp 挪到 Kg 无净收益 |
| Gate 三路 MTE2 融合 | reject | hang | — | model msprof 挂起；已回滚 |
| **P0 重钉** | **5.91** | — | — | 5905 µs；`results/P0_NEXT_SUMMARY.md` |
| **P1a Gate MTE2 PP** | **5.84** | **−0.06** | `USE_VEC_MTE2_PP=1` | 5844 µs；AllocEventID 双槽 |
| P1b Epilog MTE2 PP | reject | **+0.30** | `USE_VEC_MTE2_PP_EPILOG=0` | 6145 µs；代码保留 |
| **P2 Mask/partial** | **done** | ~0 | Init 预建 mask；DvbPartial 去 GetValue | suite 绿；BAR 修剪无净益已回退 |
| P3a FIX∥MTE2 reopen | parked | ECC | `FIX_MTE2=0` | suite 绿；model 507015 AICORE；状态机保留 |
| P3b L0 dbuf | skip | — | `L0_AB_DBUF=0` | P3a 未绿不试 |
| **P4 soft-pipe** | **opened** | — | — | 见 `P4_SOFTPIPE_PLAN.md` |
| **E0 重钉（Epilog iter）** | **5.94** | — | — | 5942 µs；`results/E0_SUMMARY.md`；仍 AIV/BAR |
| **E1 Mask/Select slim** | **~5.89** | **−0.056 vs E0** | `USE_MASK_SELECT_SLIM=1` | Init `zeroSelBuf_`；少 Duplicate/BAR |
| E2 Epilog store merge | reject | ~0 | `USE_EPILOG_STORE_MERGE=0` | ~5881 µs flat；代码保留 |
| E3 Fold/BAR slim | reject | ~0 | `USE_FOLD_BAR_SLIM=0` | ~5894 µs；代码保留 |
| E4 WIN_SOFT_LEAD_V2 | reject | **+0.37** | `USE_WIN_SOFT_LEAD_V2=0` | 6261 µs；PostS2→S0(next)→S3；代码保留 |
| E5 FIX∥MTE2 | parked | — | `FIX_MTE2=0` `L0_AB_DBUF=0` | AIV 税未明显下降，不重开 |
| **当前** | **~5.89** | vs P1a −0.05 | 下表 | E1 default on；E2–E4 parked |

E0 画像：`aic_cube≈4.2%`；`aiv_scalar≈48.7%`；`aiv_mte2≈20.5%`；`aiv_mte3≈14.3%`；`wait_id10≈2.90ms`。  
Sim（`prof_sim_t1024_p1a`）：BAR 30% / MOVEMASK 12% / MTE_UB_GM 19%。

Epilog-iter（Cursor `wy_dqkg_epilog_iter`）已收：E1 有效；E2–E4 有代码无板端净益。

---

## 2. 下一刀裁决

| 候选 | 裁决 | 原因 |
|------|------|------|
| Gate MTE2∥V ping-pong | **done** | −0.06 ms；default on |
| Epilog MTE2∥V ping-pong | **reject** | +0.30 ms |
| Mask/partial 去 scalar | **done** | Init 预建 + DvbPartial Brcb |
| Stage3 Mask/Select slim | **done** | −0.056 vs E0；default on |
| Epilog store merge | **reject** | flat；宏 0 |
| Fold/BAR slim | **reject** | flat；宏 0 |
| P4 soft-pipe V2 | **reject** | +0.37 ms（prefill=1 调度）；宏 0，代码保留 |
| I4 FIX∥MTE2 | **parked** | model 507015；E5 不重开 |

下一刀（若继续）：算法/切分或更深 soft-pipe 变体（勿盲开 Gate/Cube 装数）。

---

## 3. 已开宏一览（kernel）

```text
USE_L1_A_RESIDENT=1
USE_STAGE1_L0C_ACCUM=1
USE_GATE_REUSE_KG_WS=1
USE_EPILOG_VEC_FOLD=1
USE_DUAL_AIV_MASK=1
USE_EARLY_MASK_PER_HEAD=1
USE_STAGE2_PRELOAD_A=1
USE_STAGE0_DA_L0C_ACCUM=1
USE_GATE_EARLY_SET=1
USE_DUAL_AIV_STORE=1
USE_MASK_SOFT_LEAD=1
USE_WIN_SOFT_LEAD=1
KDA_BWD_PREFILL_WINDOWS=2
USE_VEC_MTE2_PP=1
USE_VEC_MTE2_PP_EPILOG=0
USE_MASK_SELECT_SLIM=1
USE_EPILOG_STORE_MERGE=0
USE_FOLD_BAR_SLIM=0
USE_WIN_SOFT_LEAD_V2=0
USE_KG_GATE_INTERLEAVE=0
USE_VS0_ONCE_PER_WINDOW=0
USE_L0_AB_DBUF=0
USE_FIX_MTE2_OVERLAP=0
```
