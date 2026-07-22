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

**假设：** Neumann 少一轮 → 少约 1×(X Fixpipe+Nd2Nz) + 1×(Y Fixpipe+Nd2Nz)，缩短 `SOLVE_DONE` 臂。  
**改法：** 宏 `USE_MCH_ITERS_2=1` → `MCH_ITERS=2`（默认 3 仍可回退）。  
**预期：** Dur 可见下降；`akkd` 相对误差可能变大。  
**风险：** 精度不过 → 默认关。

### M2 — 跳过 / 便宜化 `X@I`（仅当 M1 后 fixpipe 仍厚且精度有余量）

**假设：** 每轮 `L0C=X@I` 只为置初值；若有 L1→L0C 或等价便宜路径可省 2–3 次小 MMAD。  
**预期：** 小（mac 本就不墙钟）。无优则关。

### M3 — Fixpipe∥下一拍 Nd2Nz / 少一次 Y 双载（可选边角）

双缓冲已在；再挤期望 ≤0.1–0.2 ms。默认靠后。

## 年表

| 阶段 | 状态 | 备注 |
|------|------|------|
| M0 本文档 | done | |
| M1 iters=2 | done · **精度失败** | 首 case `akkd_rel≈520` → `USE_MCH_ITERS_2=0`；BC=16 需满 3 轮 |
| M2 skip X@I | pending | 精度无余量砍 iter；可试「同 3 iter、少 X@I」 |
| M3 fix∥nd2 | pending / 按需 | |

### M1 实测

| 配置 | 精度 | Dur |
|------|------|-----|
| S4a `MCH_ITERS=3` | 过 | 3.802 ms |
| M1 `MCH_ITERS=2` | **挂** `akkd_rel≈520` | 未采 |

结论：Neumann 第 3 轮对 BC=16 非可选；不能靠减 iter 砍 Fixpipe。下一步优先 **M2（同 3 轮内减搬运/X@I）**，而非再碰 iter 数。

## 开关叠加

- 保持：`USE_SCORE_TILE_MMAD`、`USE_MCH_L0_*`、`USE_MCH_L1_RESIDENT`、`USE_S4_NO_POST_BARRIER=1`  
- 关：`USE_S2C_BATCH`、`USE_SCORE_SOFT_PREFETCH`、`USE_WS_HALF_LOAD`、`USE_PREP_BEFORE_DONE`、`USE_MCH_S2B_STEAL`  
- 本 plan：`USE_MCH_ITERS_2`
