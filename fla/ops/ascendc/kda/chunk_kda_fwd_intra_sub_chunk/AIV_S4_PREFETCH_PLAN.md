# Intra-sub AIV S4 + Prefetch（910B · 不融合）

> 基线：**T4 L1-resident ≈ 4.10 ms**（`USE_MCH_L1_RESIDENT=1`）  
> 约束：**不与其他算子融合**；**只优化 910B**（不做 950 L0C→L1）  
> 目标：压 `aiv_scalar≈1.57 ms` / CrossCore 空等；乐观落到 **~3.4–3.8 ms**（不承诺 1.5）  
> 母本衔接：`CEILING_1P5.md` / `TARGET_1P5_ANALYSIS.md` §P1

## 门禁

```text
单变量改码 → 精度 suite 过 → 空闲卡 device1 msprof Task Dur 中位
→ 相对基线 ≥0.05 ms 下降则保留；否则开关回退并 commit 记账
→ 再下一刀
```

## 证据（为何动 AIV / 同步）

| 源 | 数 | 含义 |
|----|----|------|
| T4 `op_summary` | `aiv_scalar≈1.57` ≫ `aiv_vec≈0.60` | 最大单项在 AIV 控制/空等，不在向量体 |
| 同 | `aic_fixpipe≈0.92` + `aic_mte2≈0.95` | 910B Fixpipe→GM→Nd2Nz 硬下限；本 plan **不主攻** |
| 板端历史 | `aic wait solveReady` 大于仿真 WriteSolve | 嫌疑：**WriteSolve 后 Dual-AIV `CrossCoreBarrier` + 收尾** |
| 现码 | 每 sub：`WriteSolve` → **`CrossCoreBarrier`** → 双 AIV `Set(solveReady)` | `0x2` 旗已要求两 AIV 都 Set；**Barrier 对 AIC 可见性可能冗余** |

## 明确不做

- 与上下游算子融合 / PR190 合核  
- Ascend950 / L0C→L1  
- 重开 S2c、S2a、刷 mac  
- S2b steal（除非新 wait 图证明 AIC 在 MCH 后空转）

## 阶段

### R1 — S4a：去掉 WriteSolve 后的 `CrossCoreBarrier`

**假设：** `MIX 1:2` 下 `CrossCoreSetFlag<0x2>` 已是「两 AIV 各自写完半行再通知 AIC」；中间 `CrossCoreBarrier` 只增加 AIV 互等，抬高 `aiv_scalar` / 推迟 `solveReady`。

**改法：**

```text
# 今
PostSubWriteSolve(half)
CrossCoreBarrier
CrossCoreSetFlag(solveReady)   # 两 AIV 都 Set

# R1
PostSubWriteSolve(half)
CrossCoreSetFlag(solveReady)   # 仍两 AIV 都 Set；靠 0x2 汇合
```

宏：`USE_S4_NO_POST_BARRIER=1`（默认开 bring-up）。  
保留 prologue Identity 后 Barrier（首拍 I 可见性，另议）。

**门禁：** 精度绿；Dur med vs ≈4.10。

### R2 — soft-prefetch 下拍 Score `KG`→L1B

**前提：** R1 保留或打平后。  
**假设：** MCH 占用 `MCH_L1_BASE+` 时，Score L1 前缀空闲；在等 `solveDone` / MCH 尾可预取 `slotNext` 的 `PLANE_KG`。

宏：`USE_SCORE_SOFT_PREFETCH=1`。  
预期小；无收益则关。

### R3 — S4b 非对称 Post（仅当 R1 无优且 `aiv_scalar` 仍高）

单 AIV 满行 `WriteSolve+Store`，另一 AIV 只 Prep；去掉半行汇合语义。侵入更大，单变量、可回退。

## 年表

| 阶段 | 状态 | 备注 |
|------|------|------|
| R0 本文档 | **done** | |
| R1 S4a no post-barrier | **done** | Dur **3.802**（基线 4.112）；`aiv_scalar` 1.57→1.37；精度绿 |
| R2 soft-prefetch | pending | |
| R3 S4b asymmetric | pending / 按需 | |

## 开关

- 既有最佳：`USE_SCORE_TILE_MMAD=1`、`USE_MCH_L1_RESIDENT=1`、`USE_S2C_BATCH=0`、`USE_MCH_S2B_STEAL=0`
- 本 plan：`USE_S4_NO_POST_BARRIER`、`USE_SCORE_SOFT_PREFETCH`
