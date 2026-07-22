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

### W2 — V1 在 WaitDone 前 Prep(next)（仅当 W1 后 WriteSolve/ready 仍厚）

**假设：** 把 Prep(next) 从「MCH 下」挪到「MMAD(i) 尾 / WaitDone 前」，略提前 `ready(next)`。  
**预期：** 小（Prep 已被 MCH 藏住）。无优则关。

宏：`USE_PREP_BEFORE_DONE=1`。

### W3 — B4 负对照（可选，默认跳过）

仅当用户要求验证历史「非对称 Post」假设时：单变量开、记负结果、默认关。

## 年表

| 阶段 | 状态 | 备注 |
|------|------|------|
| W0 本文档 | done | |
| W1 half-load | done · **无优** | Dur **3.801** ≈ S4a **3.802**；`aiv_mte2` 亦无降 → `USE_WS_HALF_LOAD=0` |
| W2 prep-before-done | pending | WriteSolve/ready 仍厚（`aiv_scalar≈1.38`）→ 继续试 |
| W3 B4 负对照 | skipped 除非点名 | |

### W1 实测（device1，模型 shape，中位）

| 配置 | Task Dur ms | aiv_scalar µs | aiv_mte2 µs |
|------|------------:|--------------:|------------:|
| S4a | 3.802 | 1371 | 647 |
| W1 half-load | 3.801 | 1384 | 658 |

结论：半行加载未缩短 `solveReady` 臂；墙钟仍由 scalar/同步主导，MTE2 半表不是热点。

## 开关叠加

- 保持：`USE_SCORE_TILE_MMAD`、`USE_MCH_L1_RESIDENT`、`USE_S4_NO_POST_BARRIER=1`  
- 关：`USE_S2C_BATCH`、`USE_SCORE_SOFT_PREFETCH`、`USE_MCH_S2B_STEAL`  
- 本 plan：`USE_WS_HALF_LOAD`、`USE_PREP_BEFORE_DONE`
