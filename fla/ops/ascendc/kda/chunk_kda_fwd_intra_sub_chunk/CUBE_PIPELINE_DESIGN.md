# ChunkKdaFwdIntraSubChunk：压缩事实 + scalar 深挖

> Shape `(1,32,8192,128)` BT=64 bf16 · 空闲卡 `ASCEND_DEVICE_ID=1` · **以 msprof Task Duration 中位为准**  
> 当前最佳：**T4 L1-resident ≈ 4.1 ms**（天花板：`CEILING_1P5.md`）· **1.5 ms 单算子内未达成**  
> 详见 `L0_ACC_MCH_DESIGN.md` / `PHASE_B_DUAL_PLAN.md` / `SCORE_TILE_CROSSCORE_PLAN.md`  
> 交互分析 Canvas：`intra-sub-chunk-scalar-deep-dive.canvas.tsx`

---

## 0. 锁定事实（勿被 host avg / ratio 带偏）

| 阶段 | Task Dur | aiv_scalar **绝对** | aiv_scalar ratio | aic_mac | aic_scalar |
|------|----------|---------------------|------------------|---------|------------|
| 早期 scalar Post | 52.3 ms | ~22 ms | 0.43 | ~0.07 ms | — |
| phased | 15.82 | **4.06** | 0.26 | 0.27 | 3.08 |
| P6 | 8.96 | **4.46** | 0.52 | 0.27 | 2.87 |
| P3 | 8.61 | **4.25** | 0.51 | 0.27 | 2.88 |
| **P3b（最新）** | **8.32** | **4.03** | **0.50** | **0.27** | **2.90** |

精度：P6/P3/P3b 全量 `test_npu_chunk_kda_fwd_intra_sub_chunk.py` **pass**。  
目标：~**5 ms**（缺口 ≈3.3 ms）。

**关键观察：** phased→P3b 墙钟腰斩，`aiv_scalar` **绝对时间几乎不动（~4 ms）** → ratio 升高是**分母效应**，不是又写回了 Get/Set 环。

---

## 1. 分核 / CV（已锁定）

```text
task = B×HV×NT   (~4096)     NC=BT/BC=4 核内循环
MIX_AIC_1_2 · ~20 MixBlock · Dual-AIV 行半区 · DEPTH=2
AIV: prep(0); for i: WaitDone → WriteSolve+barrier+solveReady
         → prep(i+1)+ready ‖ MCH; Adds; Store; barrier
AIC: WaitReady → MMAD → Done → MCH(WaitSolveReady…)
```

- 不要改回 `B×HV×NT×NC` 扁平 Cube（flag×4，更慢）。
- `aic_mac` 恒为 **0.27 ms**；`cube_utilization~96%` 只说明「有 MAC 时很忙」，绝对窗极短。

---

## 2. 为何 scalar 占比仍 ~50%

1. **分母效应**：砍的是 vec/MTE3/旧标量数学；留下的 ~4 ms 同步税不动 → ratio↑。  
2. **同步预算对得上**：~205 chunk/核 × NC4 ≈ **820 sub/核**  
   - AIV scalar 4.03 ms ⇒ **~4.9 µs/sub**  
   - AIC scalar 2.90 ms ⇒ **~3.5 µs/sub**（等 AIV）  
   - MAC ⇒ **~0.32 µs/sub**  
   每 sub：`WaitDone` / `solveReady` / `ready` / 2×AIV barrier / 多对 HardEvent —— 量级覆盖残差。  
3. **Pipe 加总不满 1**：AIV≈0.83、AIC≈0.78 → 另有 17–22% 气泡，部分也进 scalar。  
4. 源码：`SetFlag/WaitFlag` ~60 对、`PipeBarrier` ~50、CrossCore 十余处；`GetPhyAddr` 仅 Select mask 填充。

**结论：** 当前「高 scalar」= **CV/HardEvent 空等 + 控制流**，不是 Post tril/I 数学。

---

## 3. 剩余刀：见效预判（与 5ms plan 对齐）

| 方向 | 预期墙钟 | 5ms plan |
|------|----------|----------|
| HardEvent 合并 | −0.5~−1.5 | S1 |
| Prep 减负 | −0.3~−1.0 | S3 |
| **AIV 先发（Prologue 双 Prep / DEPTH=3）** | −0.2~−0.8 | **S2a** |
| AIC MMAD 偷发 + soft-prefetch | −0.3~−1.5 | S2b |
| 批 MMAD/MCH、双 chunk 打包 | 视缺口 | S2c 可选 |
| 非对称 AIV / 去 Post barrier | ±0.5 | S4 |
| **P5b SolveTri `Mmad_ACC`** | −0.5~−2 | **S5** |
| 再抠 Select/tril/I；扁平 ×NC | 否决 | 不做 |

完整清单与优先级见 plan §1「分析清单 → Plan 覆盖」。

---

## 4. 分核还能挖什么

| 已做对 | 可试 | 不要 |
|--------|------|------|
| `B×HV×NT` + 核内 NC | 同 chunk **合并 solveReady** / 批 MMAD 再 MCH（加 WS、减 flag） | 扁平 ×NC |
| DEPTH=2 prep‖MCH | **非对称 AIV**（Post 单核、另一核只 prep） | 为并行而并行半行若 barrier 更贵 |
| Dual-AIV 行拆 | 一 task 打包 2 chunk（摊前奏，通常小） | 改 ABI/BC |

---

## 5. 验收口径与下一 plan

- 看 **`Task Duration` + `aiv_scalar_time(us)` 绝对值**，少看 ratio。  
- 空闲卡；全量精度不回退。  
- **5ms 冲刺 plan（含 P5b / SolveTri `Mmad_ACC` 求逆核内化详述）**：  
  `/root/.cursor/plans/intra_sub_chunk_sync_tax_e5f6a7b8.plan.md`  
  三线：Sync-Tax → Cube 填饱（双 Prep / MMAD 偷发）→ **MCH 核内 `X+=X@Y`**。

### S1+S3（已合入）

| | Task Dur | aiv_scalar | aic_scalar |
|--|----------|------------|------------|
| P3b | 8.321 | 4.033 | 2.901 |
| S1+S3 | 8.324 | **3.815** | 2.901 |

Prep：`mid` 每 sub 一次、三平面一次 Zero、QG/W/KG 一次 MTE3 burst。

### S2a（已试回退）

prologue 双 Prep：Dur **8.51**（+0.18）。Prep1 推迟 `solveReady0`；WaitSolve > WaitReady。

### S5（已合入：闭式 MCH）

`X = X0(I+Y)(I+Y²)(I+Y⁴)`，Y=L²。砍 3×`X+=TMP` 乒乓 → 2 次 CV 握手。

| | Task Dur | aiv_scalar | aic_scalar | aic_mac |
|--|----------|------------|------------|---------|
| S1+S3 | 8.324 | 3.815 | 2.901 | 0.266 |
| S5 | 7.989 | 3.098 | 2.815 | 0.266 |
| S2b | 7.988 | 3.102 | 2.815 | 0.266 |
| **S5b+S4** | **6.588** | **2.076** | 2.766 | 0.266 |
| **L0 ACC** | **5.325** | **1.621** | **1.016** | 0.308 |
| **Phase B Dual** | **4.654** | **1.613** | **1.018** | **0.248** |
| Dual+S2b（否决） | 4.681 | 1.611 | 1.015 | 0.248 |

S5b：I+Y 三平面一次 burst；Store 直接读 `SOLVE_TMP`；去 post-Store barrier。  
S2b（闭式期）：Y-powers 后偷发 MMAD(i+1)（墙钟持平）。  
L0 ACC：经典 Neumann + 双 L0B 预载 `Mmad_ACC`；单次 CV；Store 读 `SOLVE_X`。  
**Phase B Dual**：`MchL0AccDual` X∥Y 双事件 + L1 回灌；`USE_MCH_L0_DUAL=1`；S2b steal 再评估无收益已关（见 `PHASE_B_DUAL_PLAN.md`）。

### L0 ACC / Phase B（当前热路径）

- 精度：suite all passed（Dual；Dual+S2b 亦绿但 Dur 无优）  
- 空闲卡 device1：Dur **4.65 ms**（相对 S5b 6.59 −29%；相对 Phase A 5.33 −13%）  
- 默认：`USE_MCH_L0_ACC=1` + `USE_MCH_L0_DUAL=1` + `USE_MCH_S2B_STEAL=0`

---

## 6. 已合入（摘要）

`af4ba42` A–C → … → `147bc1e` P3b → `3617784` S1+S3 → `905e502` S5 → `3879410` S2b → `32ecc7c` S5b/S4  

旧文：`PARTITION_CUBE_ANALYSIS.md`、`SCALAR_BOTTLENECK_ANALYSIS.md`（历史）。
