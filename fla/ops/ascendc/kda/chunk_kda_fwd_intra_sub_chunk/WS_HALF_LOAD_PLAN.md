# WriteSolve 半行加载 + 可选阶段非对称（910B · 不融合）

> 基线：**S4a ≈ 3.80 ms**（`USE_S4_NO_POST_BARRIER=1`）
> 约束：不融合；仅 910B；保持双 AIV `SetFlag<0x2>` 语义
> 依据：对称半行分析 — WriteSolve 双核各搬 **整表** 16×16 是主税；经典「Post 单核/Prep 另核」易拉长 `solveReady`

## 门禁

```text
单变量 → 精度 suite 过 → device1 msprof Task Dur 中位
→ 相对基线下降 ≥0.05 ms 则保留；否则开关回退并 commit 记账
→ 再下一刀
```

## 明确不做

- 经典 B4：V0 满行 WriteSolve+Store、V1 只 Prep（默认不作为主路径）
- 重开 S2c / S2a / soft-prefetch（已否或无优）
- 融合 / 950

## 试验顺序

### W1 — WriteSolve 半行加载（对称优化，主刀）

**问题：** 两 AIV 各 `DataCopy` 整表 Aqk/Akk，却只 tril/写半行 → GM→UB 约 2× 冗余。

**改法：**

```text
# 今
DataCopy(aqk, cmat[AQK][0], bc*bc)
DataCopy(akk, cmat[AKK][0], bc*bc)
ApplyTrilScaleBeta(..., rowBegin, rowEnd)
WriteSolveInputs(..., rowBegin, rowEnd)

# W1
DataCopy(aqk[rowBegin*bc], cmat[AQK][rowBegin], half_elems)
DataCopy(akk[rowBegin*bc], cmat[AKK][rowBegin], half_elems)
# tril / WriteSolveInputs 不变（仍半行）
LoadBeta：可保持 valid 全长或只加载本行段（优先最小改动：仍 load valid，成本小）
```

宏：`USE_WS_HALF_LOAD=1`（默认开 bring-up）。

**预期：** 缩短 `solveReady` 臂 / 降 `aiv_mte2`；Dur 相对 3.80 有可见下降。
**风险：** 半行 DataCopy 对齐（`bc_=16` fp32 行=64B，一般 OK）；`aqkBuf_` 未写半区勿被 Store 误读（Store 只读本核行）。

### W2 — Prep(next) 叠 MMAD；ready 仍在 solveReady 后立即 Set

**假设：** `depth=2` 下 next slot 空闲。`PrepareSub(i+1)` 挪到 `WaitDone(i)` 前（叠 MMAD）；**禁止**提前 `Set ready`（否则 AIC 会在 Store/MCH 握手未完时开 MMAD(next) → NaN）。`ready` 在 `solveReady` 后立刻 Set（中间不再夹 Prep）。

**改法：**

```text
# 默认
WaitDone(i) → WriteSolve → solveReady → Prep(i+1)+ready → MCH wait → Store

# W2
Prep(i+1) → WaitDone(i) → WriteSolve → solveReady → ready → MCH wait → Store
```

宏：`USE_PREP_BEFORE_DONE`（实测精度失败，默认 **0**）。

**预期：** 小（去掉 solveReady→ready 之间的 Prep 间隙）。无优则关。
**风险：** 提前 ready → AIC 抢跑 NaN；仅提前 Prep 亦 NaN（未再深挖）。

### W3 — B4 负对照（可选，默认跳过）

仅当用户要求验证历史「非对称 Post」假设时：单变量开、记负结果、默认关。

## 年表

| 阶段                | 状态                      | 备注                                                                                     |
| ------------------- | ------------------------- | ---------------------------------------------------------------------------------------- |
| W0 本文档           | done                      |                                                                                          |
| W1 half-load        | done ·**无优**     | Dur**3.801** ≈ S4a **3.802**；`aiv_mte2` 亦无降 → `USE_WS_HALF_LOAD=0` |
| W2 prep-before-done | done ·**精度失败** | 提前`Set ready` → NaN；仅提前 Prep 亦 NaN → `USE_PREP_BEFORE_DONE=0`               |
| W3 B4 负对照        | skipped 除非点名          |                                                                                          |

### W1 实测（device1，模型 shape，中位）

| 配置         | Task Dur ms | aiv_scalar µs | aiv_mte2 µs |
| ------------ | ----------: | -------------: | -----------: |
| S4a          |       3.802 |           1371 |          647 |
| W1 half-load |       3.801 |           1384 |          658 |

结论：半行加载未缩短 `solveReady` 臂；墙钟仍由 scalar/同步主导，MTE2 半表不是热点。

### W2 实测

1. **Prep(next)+ready 在 WaitDone 前**：首 case `aqk` 出现 NaN（AIC 过早开 MMAD(next)）。
2. **仅 Prep 提前、ready 仍在 solveReady 后**：同样 NaN（疑 UB/流水或 score WS 与在飞 MMAD 冲突，未再深挖）。

→ 开关默认关；保持 Prep‖MCH 原序。

## 开关叠加

- 保持：`USE_SCORE_TILE_MMAD`、`USE_MCH_L1_RESIDENT`、`USE_S4_NO_POST_BARRIER=1`
- 关：`USE_S2C_BATCH`、`USE_SCORE_SOFT_PREFETCH`、`USE_MCH_S2B_STEAL`、`USE_WS_HALF_LOAD`、`USE_PREP_BEFORE_DONE`
- 本 plan 代码保留：半行加载 / Prep-before-done 供日后复测
