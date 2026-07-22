# Post-MCH 试验（910B · 不融合）

> 基线：**S4a ≈ 3.802 ms**（`MCH_SHORTEN_PLAN` 已收束）  
> 约束：不融合；仅 910B；单变量门禁（同前：精度 → msprof → ≥0.05 ms 则留）

## 为何换题

M1–M3 证明：在必须 3×Neumann + 每轮 Fixpipe→GM→Nd2Nz 的前提下，**叠流水 / 省 MAC 不够门禁**。  
剩下来仍可能动墙钟的，是 **少一次真 GM 搬运**（降 `aic_mte2` 字节），不是再排事件。

## 试验顺序

### P1 — Y 单次 Nd2Nz（主刀）

**现状：** 每中间 iter 对同一 `yBase` 做 `LoadL1A + LoadL1B`（两套 Nz）。  
**假设：** 只 Nd2Nz 一次到 L1B；L1A 侧用 L1 内转换或可接受的等价路径，省 ~2×Nd2Nz/sub。  
**改法：** `USE_MCH_Y_SINGLE_LOAD=1`（具体实现以穿刺为准：优先找 Catlass/AscendC L1B→L1A 或同缓冲视图；若无可靠 API 则本刀 **skip 并记否**）。  
**预期：** mte2 可见下降；Dur −0.05~0.15 ms。精度不过或无 API → 默认关。

### P2 — 开场 L/Lb 单载（仅当 P1 通路成立）

Prologue `L` 同样双载；复用 P1 机制。独立开关 `USE_MCH_L_SINGLE_LOAD`。

### P3 — 明确不做

- 再碰 `MCH_ITERS` / skip X@I / Fix overlap（已否）  
- 重开 S2c / WS / Prep-before-done  
- 承诺回到 1.5 ms

## 年表

| 阶段 | 状态 | 备注 |
|------|------|------|
| P0 本文档 | done | |
| P1 Y single load | pending | |
| P2 L single load | pending | 依赖 P1 |

## 开关

- 基线保持 S4a 默认开集合  
- 本 plan：`USE_MCH_Y_SINGLE_LOAD`（+ 可选 `USE_MCH_L_SINGLE_LOAD`）
