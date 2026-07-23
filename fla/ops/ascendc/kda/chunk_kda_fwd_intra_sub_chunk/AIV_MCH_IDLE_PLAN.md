# AIV 填 MCH 空窗：方向分析与执行方案

> 基线：P1b **≈ 3.705 ms**（S4a 3.802 + zN 单载）  
> 现象（best / `-g` 仿真）：`solveReady` 后 AIV 做完 `Prep(i+1)` 仍早于 `SOLVE_DONE` ≈ **2.5k tick**，其后长时间 `WaitSolveDone`  
> 约束：不融合；910B；单变量门禁（精度 → msprof → ΔDur ≥ 0.05 ms 则留）  
> 关联：`TARGET_1P5_ANALYSIS.md`（Prep 非临界）、`WS_HALF_LOAD_PLAN.md`（W2 Prep-before-done NaN）、`AIV_S4_PREFETCH_PLAN.md`

---

## 1. 问题对齐

### 1.1 当前时序（默认，`USE_S2C_BATCH=0`）

```text
AIV: WaitDone → WriteSolve → solveReady → Prep(i+1)+ready → █ WaitSolveDone █ → Store(aqk+akkd)
AIC:                         └──────────── MCH(i) 很长 ────────────┘
```

- `DEPTH=2`：只保证 `score/cmat/solve` 双槽，**不是**「先发两拍再连跑」。
- `Prep(i+1)` **已经**叠在 MCH 上；空的是 Prep 结束后的等旗。
- `CrossCoreFlag` 为 **1:1**，不能在同一 `ready` 上积压两次。

### 1.2 空窗里「下一步」分别是什么

| 候选 | 依赖 | 与空窗关系 |
|------|------|------------|
| 再 Prep(i+1) | 无 | 已做完，重复无意义 |
| WriteSolve(i+1) | `Done(i+1)` / cmat | MCH(i) 期间 **不能**做 |
| Prep(i+2) | 要空槽 → **DEPTH≥3** | 能填空窗，但见 §2 |
| Store **aqk** | 仅需 WriteSolve 后的 `aqkBuf_` | **与 MCH 无数据依赖** |
| Store **akkd** | `SOLVE_X`（MCH 产出） | **必须**等 `solveDone` |
| 下一 chunk 的 Prep(0) | 末 sub 才有空槽 | 覆盖面窄（1/NC） |

### 1.3 关键判断（为何不先推 DEPTH=3）

仿真已表明：**`READY` 早于 `SOLVE_DONE` ≈ 2.5k** → Prep **很少拖 AIC**。  
因此把更多 Prep 塞进空窗（DEPTH=3 的 Prep(i+2)）多半只是 **把空等从 Wait_A 挪到 Wait_B**，墙钟期望 **≪ 0.05 ms**，还要 +50% workspace。

真正能动墙钟的，是把 **现在卡在 `solveDone` 之后、又拖住下一拍 `solveReady` 的 AIV 工作** 前移进空窗。

`PostSubStore` 今天把两件事绑在一起：

1. `aqkBuf_` → Cast → `aqk_`（**不依赖 MCH**）
2. `solveWs_[X]` → `akkd_`（**依赖 MCH**）

→ **最值得做的一刀：StoreAqk ‖ MCH，solveDone 后只剩 StoreAkkd。**

```text
目标时序:
AIV: solveReady → Prep(i+1)+ready → StoreAqk(i) → WaitSolveDone → StoreAkkd(i)
                                                 └─ 填原空窗 ─┘
```

缩短 `solveDone → WaitDone(next) → WriteSolve → solveReady` 尾部，减轻 AIC 在 `WaitSolveReady` 上的气泡。

---

## 2. 候选对比（优先级）

| 优先级 | 方案 | 期望 ΔDur | 代价 / 风险 | 结论 |
|--------|------|-----------|-------------|------|
| **P0** | **StoreAqk ‖ MCH** | **0.05–0.20 ms**（视 Store 占比） | 低：DEPTH 不变；注意 `vecBuf_` 与 Prep 互斥序；dual-AIV 半行 | **主推** |
| P1 | DEPTH=3 + Prep(i+2)（ready 仍按拍 Set） | 多半 &lt;0.05 ms | WS ×1.5（~0.74→1.1 MB@20 核）；旗序易错 | **P0 后再量，默认不先做** |
| P2 | chunk 末拍 Prep(next task) | 更小（1/NC） | 跨 `ProcessChunk` 状态 | 否决为独立主刀 |
| — | 重开 S2a / W2 Prep-before-done / S2c | 已否或回归 | — | **不做** |
| — | 字面「同一 ready 先发两任务」 | — | Flag 非计数 | **做不到** |

---

## 3. P0 详细方案：StoreAqk ‖ MCH

### 3.1 代码改动（单变量）

宏：`USE_STORE_AQK_UNDER_MCH=1`（默认先开试验，门禁不过则回 0）。

1. 从 `PostSubStore` 拆出：
   - `PostSubStoreAqk(...)`：只 Cast + `DataCopyPad` → `aqk_`（数据来自 `aqkBuf_`，与现逻辑一致）
   - `PostSubStoreAkkd(...)`：只从 `SOLVE_X` 搬 `akkd_`
2. `ProcessChunkAiv` 默认环改为：

```text
WaitDone
WriteSolve
[barrier if needed]
solveReady
if (i+1 < nc): Prep(i+1); ready
#if USE_STORE_AQK_UNDER_MCH
  PostSubStoreAqk(i)          // 叠 MCH；须在 Prep 之后（共用 vecBuf_）
#endif
WaitSolveDone                 // PostSubMchWait
#if USE_STORE_AQK_UNDER_MCH
  PostSubStoreAkkd(i)
#else
  PostSubStore(i)             // 旧路径
#endif
```

3. **禁止**在 Prep 之前 StoreAqk（`vecBuf_` 冲突）。  
4. **禁止**提前 `Set ready`（吸取 W2 NaN）。  
5. `USE_S2C_BATCH` 路径保持旧 Store（或显式 `#if` 禁用本宏），避免与 aqk spill 纠缠。

### 3.2 正确性约束

| 项 | 处理 |
|----|------|
| aqk 生命周期 | WriteSolve 后 `aqkBuf_` 保持至 StoreAqk；中间 Prep 不得覆盖 `aqkBuf_`（Prep 用 `vecBuf_`/`scoreWs_`） |
| dual-AIV | 仍按半行 Split；两核都做本行 StoreAqk |
| empty / valid | 与现 `PostSubStore` 早退一致 |
| 精度 | 全 suite；建议加 P1b 式 ON/OFF 同输入差分（aqk/akkd 应近 bitwise） |

### 3.3 验收

```text
改码 → device7 精度 suite
→ device1 msprof Task Dur 中位 vs 3.705
→ Δ ≤ −0.05 ms 保留并 commit；否则默认关 + 记账
→ 可选：再采一版 T=1024 sim，看 WaitSolveDone 段 AIV 是否出现 StoreAqk
```

### 3.4 期望与上限

- 乐观：Store 里 aqk 约占一半 Cast/MTE3，且该尾部位于临界路径 → **~0.1 ms 量级**。  
- 悲观：墙钟仍钉在 AIC Fixpipe，AIV 尾缩短但不露脸 → **平坦**（与 soft-prefetch 同类）。  
- 即使平坦，也比 DEPTH=3 更值得作为 **第一刀**（代价小、对准空窗语义）。

---

## 4. P1 方案备忘：DEPTH=3 Prep(i+2)（不优先）

仅当 P0 后仿真仍显示 **长 WaitSolveDone 且 AIV 无其它可搬工作** 时考虑。

### 4.1 改动要点

- `SCORE_QUEUE_DEPTH`：kernel + tiling **2→3**（WS ×1.5）。  
- 在 `WaitSolveDone` 前、`Prep(i+1)+ready` 后：若 `i+2 < nc`，`PrepareSub(i+2)` **不要 Set ready**。  
- 下一拍 `solveReady` 后：若 Prep 已做，只 `Set ready`；再启动 `Prep(i+3)` 无 ready，类推。  
- Slot：`(i+2) % 3` 不得与在飞 MMAD/MCH 写口冲突（三槽互异，MCH(i) 已释放 score(i)）。

### 4.2 风险

- 收益可能过不了 0.05 ms 门禁（Prep 非临界）。  
- 旗序/槽位错误 → 精度灾难或死等。  
- W2 类回归：任何「与在飞 MMAD 同槽写 score」必须禁止。

---

## 5. 明确不做

- 重开 S2c / S2a / W2 Prep-before-done / S2b steal（无新证据）  
- 同一 `readyFlag` 连续 Set 两次「先发两任务」  
- 承诺靠填空窗打到 1.5 ms  
- 为 DEPTH=3 先扩 WS 再找活干

---

## 6. 执行年表

| 阶段 | 内容 | 状态 |
|------|------|------|
| D0 | 本文档 | done |
| **D1** | **`USE_STORE_AQK_UNDER_MCH`：拆 Store + 叠 MCH** | **done · 实现保留** |
| D2 | 精度 + msprof 门禁 + commit | **fail → default off**（med **3.674** vs **3.705**，Δ−0.031） |
| D3 | （可选）DEPTH=3 Prep(i+2)，仅 D1 后空窗仍厚且有证据 | pending / 默认跳过 |
| D4 | 文档回写 best Dur / 仿真 | pending |

### D2 实测

| 配置 | Task Dur median | aiv_mte3 | aiv_scalar |
|------|-----------------|----------|------------|
| P1b | 3.705 ms | 489 us | 1374 us |
| StoreAqk‖MCH | **3.674 ms** | 596 us | 1337 us |

墙钟仍钉 AIC Fixpipe/MTE2；AIV 尾前移未露脸到 ≥0.05 ms。

---

## 7. 开关叠加（试验时）

- 保持：Score Tile、MCH L0 Dual、L1 resident、S4a、**P1b zN 单载**  
- 关：S2c、soft-prefetch、WS half-load、Prep-before-done、MCH_ITERS_2、SKIP_XI、FIX_OVERLAP  
- 本 plan：`USE_STORE_AQK_UNDER_MCH`

---

## 8. 一句话结论

**最值得做的不是先上 DEPTH=3 多 Prep，而是把 Store 拆开：用 MCH 空窗打掉不依赖 `SOLVE_X` 的 aqk 回写，让 `solveDone` 后的 AIV 尾只剩 akkd。**  
DEPTH=3 留作有 profile 证据时的备选，避免为「看起来能并行」付 1.5× workspace。
