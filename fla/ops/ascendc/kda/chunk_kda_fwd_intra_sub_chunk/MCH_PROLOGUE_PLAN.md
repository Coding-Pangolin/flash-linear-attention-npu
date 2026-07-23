# MCH Prologue 预取（L@L ∥ 搬 X/I）

> 基线：**P1b ≈ 3.705 ms**（zN 单载 + S4a 集合）  
> 依据：MCH 内无 PR190 式 L0 ping/pong；Prologue 现为「先搬完 L/X/I 再 `L@L`」，X/I 与 `L@L` **无依赖**却未重叠  
> 约束：不融合；910B；单变量门禁

## 门禁

```text
改码 → device7 精度 suite → device1 msprof Task Dur 中位
→ 相对 3.705 下降 ≥0.05 ms 则保留；否则默认关并 commit 记账
```

## 假设

```text
现在:  Load L [,X,I]  →  L@L(Fix→y)  →  Load Y
目标:  Load L → L@L(Fix 不 Wait MTE2) ∥ Load X[,I] → Wait FIX_MTE2 → Load Y
```

- `L@L` 只碰 `l1L`/`l0*`/`yBase`；`X`/`I` 进独立 L1 槽，与 Fix 写 `yBase` 无 RAW。  
- 回灌 `Y` 必须 `Wait FIX_MTE2` 之后。  
- 不做环内 L0 ping/pong（16×16 无 K 循环）。

## 改法

宏：`USE_MCH_PROLOGUE_PREFETCH=1`

1. `MchFixpipeToGm(..., waitMte2=true)` 可关末尾 Wait（与 `MchFixpipeToGmEvt` 同形）。  
2. `MchMatmulL1AccFix` 增加 `waitMte2AfterFix`（默认 true）。  
3. `MchL0AccDual` prologue：只先 Load L → `L@L(waitMte2=false)` → Load X/[I] → `Wait FIX_MTE2(MCH_EVT)` → Load Y。

## 风险

- 事件 ID：`MchFixpipeToGm` 用 `MCH_EVT=2`；Load X 内部也会 `MTE2_MTE1(MCH_EVT)` —— 须保证 Fix 的 `FIX_MTE2` 在 Load Y 前被 Wait 掉，且不与 Load X 的 flag 语义打架。  
- 收益可能 &lt;0.05 ms（Fix 仍厚；prologue 只占 MCH 一小段）。

## 年表

| 阶段 | 状态 | 备注 |
|------|------|------|
| Doc | done | |
| 实现 + 精度 + msprof | done | device7 全过；device1 `prof_msprof_op_prologue` → `/tmp` |
| 门禁结论 | **fail → default off** | med **3.679** vs P1b **3.705**（Δ−0.026 &lt; 0.05） |

## 实测（device1, PipeUtilization）

| 配置 | Task Dur median | aic_mte2 | aic_fixpipe |
|------|-----------------|----------|-------------|
| P1b 基线 `prof_msprof_op_p1_zn_alias` | 3.705 ms | ~797 us | — |
| Prologue ON | **3.679 ms** | 806 us | 919 us |

结论：prologue 重叠真实存在但太短（只盖住一次 `L@L` Fix），被 Fixpipe/MTE2 主路径淹没，达不到门禁。代码保留，宏默认 `0`。

## 开关

- 保持：Score Tile、MCH Dual/L1 resident、S4a、`USE_MCH_Y_SINGLE_LOAD=1`  
- 关：S2c / soft-prefetch / WS / Prep-before-done / MCH_ITERS_2 / SKIP_XI / FIX_OVERLAP / **`USE_MCH_PROLOGUE_PREFETCH`**
