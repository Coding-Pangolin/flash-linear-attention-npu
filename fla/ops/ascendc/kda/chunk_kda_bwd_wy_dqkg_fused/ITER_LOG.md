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
| …（早期见 git） | | | | |
| **P1a Gate MTE2 PP** | **5.84** | **−0.06** | `USE_VEC_MTE2_PP=1` | 5844 µs |
| **E1 Mask/Select slim** | **~5.89** | **−0.056 vs E0** | `USE_MASK_SELECT_SLIM=1` | |
| E2–E4 / V2 | reject | flat/+0.37 | 宏 0 | 代码保留 |
| **F0 重钉** | **5.91** | — | — | 5911 µs；`results/F0_SUMMARY.md` |
| **F1 Merge-barrier-only** | **5.48** | **−0.43** | `USE_MERGE_BARRIER_ONLY=1` | 5480 µs；去 Stage0/Kg 非 merge Join |
| F2 UB 表 | — | — | — | BK128 **NO-GO**；BV128 **GO**；`results/UB_PEAK_F2.md` |
| **F3b MAX_BV=128** | **4.74** | **−0.74 vs F1** | `USE_BV128=1` | 4741 µs；nBv=1 |
| F4 MASK_ONCE | reject | −0.016 | `USE_MASK_ONCE=0` | suite 绿；未过 −0.05；代码保留 |
| F5 FIX∥MTE2+关 Preload | parked | ECC | `FIX_MTE2=0` | 仍 507015 L0C；Preload 门控代码保留 |
| **F3a' BK128 owned-arena** | **3.86** | **−0.88 vs F3b** | `USE_BK128=1` | 3861 µs；nBk=1；`results/F3A_SUMMARY.md` |
| **F6 MVP split** | e2e | fused 26.8 / N1 101 / N2 107 | `split_stages` off | suite split 绿；launch 税；`results/F6_MVP_SUMMARY.md` |
| **F6b multi-task banks** | e2e | fused 20.8 / N1 **41.1** / N2 51.4 | off | 3 launch；bit 对齐；`results/F6B_SUMMARY.md` |
| **G0 重钉** | **3.79** | — | — | 3789 µs；`results/G0_SUMMARY.md` |
| G1 SyncPlan 窗级 C_S0/1 | reject | **+0.15** | `USE_SYNC_PLAN_V1=0` | 伤 Prefill/S1∥Gate；代码保留；`results/G1_SUMMARY.md` |
| G2 FOLD_BAR_SLIM | reject | **+0.01** | `USE_FOLD_BAR_SLIM=0` | flat；`results/G2_SUMMARY.md` |
| G4 WY L1 v2 | reject | +0.13→+0.03 | `USE_WY_L1_RESIDENT_V2=0` | 串 Mask 伤并行；ND2NZ 已低；`results/G4_SUMMARY.md` |
| **F6 切分设计** | **done** | — | — | [`SPLIT_KERNEL_PLAN.md`](SPLIT_KERNEL_PLAN.md) |
| **A5 regbase** | **compile** | — | arch35 dual-path | MicroAPI vector；910B suite 绿 + Task Dur **3793 µs**（vs G0 3789，无劣化）；950 板精度/性能见 [`ASCEND950_TEST_GUIDE.md`](ASCEND950_TEST_GUIDE.md) |
| **当前** | **~3.79** Task | vs P1a **−2.0** | 下表 | 下一刀：**G5 Gate fold** / WAIT |

---

## 2. 下一刀裁决

| 候选 | 裁决 | 原因 |
|------|------|------|
| F1 Join 收紧 | **done** | −0.43 ms |
| F3b BV128 | **done** | −0.74 ms |
| F3a BK128 裸改 | **blocked** | UB；见 F3a' |
| F3a' owned-compact | **done** | −0.88 ms；default `USE_BK128=1` |
| F4 mask-once | **reject** | flat；改 G3 |
| F5 FIX ECC | **parked** | 关 Preload 仍 507015 → G7 |
| F6 多 kernel 切分 | **MVP+F6b** | N1 e2e 101→41；仍 2× fused→off → G6 |
| **G1 SyncPlan** | **reject** | 窗级 C_S0/1 **+0.15**；nBk=1 已满足 per-head≤1 Set |
| G2 V-BAR fold | **reject** | FOLD_BAR_SLIM flat +0.01 |
| G3 MASK_TABLE | queued | 挂有效 sync 刀后 |
| **G4 WY L1 v2** | **reject** | 串 Mask +0.13；修调度后 flat；ND2NZ 已 2.8% |
| **G5 Gate fold v2** | **next** | P4 增量审计 |
| P1b BK 批握手 | **skip** | nBk=1 已吸收 |

权威 plan：**[`NEXT_ITER_PLAN_G.md`](NEXT_ITER_PLAN_G.md)** · 策略：[`IMPROVEMENT_STRATEGY.md`](IMPROVEMENT_STRATEGY.md) · 切分：[`SPLIT_KERNEL_PLAN.md`](SPLIT_KERNEL_PLAN.md)  
（F 档归档：[`NEXT_ITER_PLAN.md`](NEXT_ITER_PLAN.md)）

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
USE_MERGE_BARRIER_ONLY=1
USE_BV128=1
USE_BK128=1
USE_MASK_ONCE=0
USE_EPILOG_STORE_MERGE=0
USE_FOLD_BAR_SLIM=0
USE_SYNC_PLAN_V1=0
USE_WY_L1_RESIDENT_V2=0
USE_WIN_SOFT_LEAD_V2=0
USE_KG_GATE_INTERLEAVE=0
USE_VS0_ONCE_PER_WINDOW=0
USE_L0_AB_DBUF=0
USE_FIX_MTE2_OVERLAP=0
```
