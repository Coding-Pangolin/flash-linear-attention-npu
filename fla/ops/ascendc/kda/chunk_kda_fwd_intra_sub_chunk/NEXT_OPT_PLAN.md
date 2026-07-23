# IntraSubChunk 下一轮优化 Plan（post-P1b）

> 当前保留最佳：**P1b ≈ 3.705 ms**（Score Tile + MCH Dual/L1 resident + S4a + zN single-load）  
> 约束：不融合；910B；单变量门禁（精度 → msprof → ΔDur ≥ 0.05 ms 才默认开）

## 已关实验（代码保留）

| 实验 | 结果 |
|------|------|
| MCH_ITERS_2 / SKIP_XI / FIX_OVERLAP | 精度败或 Dur 平 |
| Prologue `L@L ∥ Load X/I` | 精度 OK，Dur −0.026 → off |

## 方案路线（按期望收益排序）

### P0 — `USE_STORE_AQK_UNDER_MCH`（见 `AIV_MCH_IDLE_PLAN.md`）

**问题：** Prep(i+1) 后 AIV 在 `WaitSolveDone` 空转；`aqk` 不依赖 MCH 结果。  
**改法：** Store 拆成 `StoreAqk ‖ MCH`，`solveDone` 后再 `StoreAkkd`。  
**风险：** CrossCore 时序 / 与 Prep 的 UB 生命周期；须保证 StoreAkkd 仍等 AIC。  
**门禁：** 同单变量协议。

### P1 —（备选）MCH 环内更长重叠

仅当 P0 不够：例如 Fixpipe Y 与下一 iter MTE1 的更激进 EVT（已试 `FIX_OVERLAP` 无效，需新证据再动）。

### 不做

- DEPTH=3 / 双 Set 同 flag（CrossCoreFlag 1:1 不可）
- L0C→L1（非 910B）
- 再砍 Neumann iter（精度败）

## 执行顺序

1. 实现 `USE_STORE_AQK_UNDER_MCH=1`  
2. device7 精度 suite  
3. device1 msprof vs **3.705**（prof 写 `/tmp`，workspace 近满）  
4. ≥0.05 ms 保留；否则默认关 + commit 记账
