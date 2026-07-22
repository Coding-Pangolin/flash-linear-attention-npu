# MCH 缩短试验（910B · 不融合）

> 基线：**S4a ≈ 3.80 ms**（Score Tile + L1-resident MCH + `USE_S4_NO_POST_BARRIER=1`）  
> 约束：不融合；仅 910B（无 L0C→L1）；单变量门禁  
> 依据：仿真 `trace.json` — AIV ~62% 等旗（主因 `SOLVE_DONE`）；板端 `fixpipe+mte2≈1.87 ms` ≈ 半墙钟；CUBE∩VECTOR≈0

## 门禁

```text
单变量改码 → 精度 suite 过 → device1 msprof Task Dur 中位
→ 相对基线下降 ≥0.05 ms 则保留；否则开关回退并 commit 记账
→ 再下一刀
```

## 明确不做 / 已否

- 重开 S2c / soft-prefetch / WS half-load / Prep-before-done / B4  
- 承诺 1.5 ms（见 `CEILING_1P5.md`）  
- 字面「中间 X/Y 不经 GM」（910B 做不到）

## 试验顺序

### M1 — `MCH_ITERS=2`（主刀）

Neumann 少一轮砍 Fixpipe/Nd2Nz。→ **精度失败**（需满 3 轮）。

### M2 — 跳过 `X@I`（同 3 iter）

`iter>0` 省小 MMAD。→ **精度过、墙钟无增益**（mac −17 us，fixpipe/mte2 钉死）。

### M3 — Fixpipe∥M / Fixpipe∥Nd2Nz

Y Fixpipe 与 `X+=X@Y` 重叠；Y Nd2Nz 与 X Fixpipe 重叠。→ **精度过、墙钟平坦**。

## 年表

| 阶段 | 状态 | 备注 |
|------|------|------|
| M0 本文档 | done | |
| M1 iters=2 | done · **精度失败** | `akkd_rel≈520` → `USE_MCH_ITERS_2=0` |
| M2 skip X@I | done · **无增益** | Dur 3.825 ≥ 3.802 → `USE_MCH_SKIP_XI=0` |
| M3 fix∥overlap | done · **无增益** | Dur 3.812 ≈ 3.802 → `USE_MCH_FIX_OVERLAP=0` |

### 实测汇总

| 配置 | 精度 | Dur 中位 | aic_fixpipe | aic_mte2 | aic_mac |
|------|------|----------|-------------|----------|---------|
| S4a | 过 | **3.802 ms** | 922 us | 949 us | 249 us |
| M1 iters=2 | **挂** | — | — | — | — |
| M2 skip X@I | 过 | 3.825 ms | 918 us | 949 us | 232 us |
| M3 fix∥overlap | 过 | 3.812 ms | 912 us | 947 us | — |

## 结论（本 plan 收束）

在 **不融合 + 910B** 下，MCH 热路径的可开关微优化已打完：

1. **不能减 Neumann 轮数**（精度硬约束）→ Fixpipe 次数下限锁死  
2. **省 MAC / 叠 FIX∥MTE2** 都不动墙钟 → 瓶颈是 **串行 GM 往返字节数**，不是气泡  
3. 板端最佳仍 **S4a ≈ 3.80 ms**；继续抠 MCH 编排期望 ≪ 0.05 ms，不值得再开刀  

下一方向须换题（见 `POST_MCH_PLAN.md`）：要么动 **Y 双载 Nd2Nz 次数**（真少一次 GM→L1），要么接受单算子天花板、走融合/950。

## 开关叠加

- 保持：`USE_SCORE_TILE_MMAD`、`USE_MCH_L0_*`、`USE_MCH_L1_RESIDENT`、`USE_S4_NO_POST_BARRIER=1`  
- 关（代码保留）：`USE_S2C_BATCH`、`USE_SCORE_SOFT_PREFETCH`、`USE_WS_HALF_LOAD`、`USE_PREP_BEFORE_DONE`、`USE_MCH_S2B_STEAL`、`USE_MCH_ITERS_2`、`USE_MCH_SKIP_XI`、`USE_MCH_FIX_OVERLAP`
