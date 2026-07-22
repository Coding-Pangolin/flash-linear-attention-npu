# Post-MCH 试验（910B · 不融合）

> 基线演进：S4a **3.802 ms** → P1b **3.705 ms**（本 plan）  
> 约束：不融合；仅 910B；单变量门禁（精度 → msprof → ≥0.05 ms 则留）

## 为何换题

`MCH_SHORTEN`（M1–M3）证明叠流水/省 MAC 不够门禁。  
剩下来能动墙钟的是 **少一次真 GM→L1 Nd2Nz**。

## 试验顺序

### P1 — zN 别名单载（主刀 · **收下**）

**发现：** Catlass 对 `RowMajor→L1A/L1B` 均为 `layout::zN`，A/B 双载内容等价。  
**改法：** `USE_MCH_Y_SINGLE_LOAD=1`：Y / L / I 各 Nd2Nz 一次，`l1*b = l1*` 供 `CopyL1ToL0B`。  
**实测：**

| 配置 | Dur 中位 | aic_mte2 | Δ vs S4a |
|------|----------|----------|----------|
| S4a | 3.802 ms | 949 us | — |
| P1 Y-only | 3.763 ms | 876 us | −0.038（未过 0.05） |
| **P1b Y+L+I** | **3.705 ms** | **797 us** | **−0.097 · 过门禁** |

精度全绿。默认 **开**。

### P2 — 已并入 P1b

原计划独立 `USE_MCH_L_SINGLE_LOAD`；同假设下与 Y/I 一并做完，不再单开。

### P3 — 明确不做

- 再碰 `MCH_ITERS` / skip X@I / Fix overlap  
- 重开 S2c / WS / Prep-before-done  
- 承诺 1.5 ms

## 年表

| 阶段 | 状态 | 备注 |
|------|------|------|
| P0 本文档 | done | |
| P1 zN single | **done · 收下** | `USE_MCH_Y_SINGLE_LOAD=1`；新基线 3.705 ms |

## 下一步（可选）

在 3.705 基线上，若仍抠 MCH GM 税：看 X 中间轮是否也能少载（X 只作 A 侧，本就单载）。更大头仍是 Fixpipe 次数（需融合/950）。

## 开关

- 新开：`USE_MCH_Y_SINGLE_LOAD=1`  
- 其余保持 S4a 默认集合
