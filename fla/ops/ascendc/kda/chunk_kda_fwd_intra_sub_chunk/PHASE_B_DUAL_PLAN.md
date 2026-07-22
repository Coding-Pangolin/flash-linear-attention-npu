# Phase B 双缓冲 + S2b Steal 评估计划

> 基线（已合入）：Phase A L0 ACC，空闲卡 Task Dur 中位 **5.325 ms**  
> （`aiv≈1.621` / `aic≈1.016` / `mac≈0.308`；相对 S5b 6.59 −19%）  
> **Phase B Dual 实测：Dur 中位 4.654 ms**（−12.6% vs Phase A；已低于 ~5 ms）  
> S2b steal 评估：**无收益**（4.681，略差），默认 `USE_MCH_S2B_STEAL=0`  
> 参考：`solve_tri_cube.h`::`MCHInvertDiagonal`；设计母本 `L0_ACC_MCH_DESIGN.md` §5.4

---

## 0. 立场（承接刚完成的分析）

| 事实 | 含义 |
|------|------|
| Phase A 已有「双 L0B」 | 只保证 ACC 语义（I/Y 分槽预载），**不是** X∥Y 流水 |
| SolveTri 真双流 | `X_BUF/Y_BUF` + `EVT_X/EVT_Y` + 独立 scratch；迭代内 `X@I ∥ Y@Y` |
| Dual 实测 | Dur **5.325→4.654**；`mac` 0.308→0.248（重叠生效）；aiv/aic 几乎持平 |
| S2b 评估 | Dual 后偷 MMAD(i+1)：精度绿，Dur **4.681**（相对 Dual **无收益**）→ **关** |

**顺序锁定：Phase B Dual → 精度/msprof → 再评估 S2b steal。禁止并行开两刀。**

---

## 1. Phase B 要抄什么 / 不抄什么

### 1.1 必抄（SolveTri）

```text
# 资源
L0A/B/C 各双份：X_BUF=0, Y_BUF=TILE   (TILE = BC*BC*sizeof(float))
HardEvent：EVT_X / EVT_Y 两套（M_MTE1, MTE1_M, M_FIX, FIX_M）
L1 常驻：I, X, Y, L  (+ I 的 L1B 副本)

# 循环 priming（首轮 Wait 能过）
SetFlag M_MTE1(X/Y); SetFlag FIX_M(X/Y)

# 每轮（NUM_ITERS=3，无 MBH）
Wait M_MTE1(X); Load L0A/B[X] ← X,I; Set MTE1_M(X)
Wait M_MTE1(Y); Load L0A/B[Y] ← Y,Y; Set MTE1_M(Y)

Wait FIX_M(X); Wait MTE1_M(X); Mmad(X, initC=true)     # L0C_X = X@I

if iter < 2:
  Wait FIX_M(Y); Wait MTE1_M(Y); Mmad(Y, initC=true)  # L0C_Y = Y@Y
  StoreY → L1_Y (经 SOLVE_Y0 scratch); Set FIX_M(Y)

PipeBarrier<PIPE_M>()
if last: Wait MTE1_M(Y)                                 # Y 只作 ACC 的 B
Mmad_ACC(L0C_X, L0A_X, L0B_Y)                           # += X@Y
StoreX → L1_X / SOLVE_X; Set M_MTE1(X/Y); Set FIX_M(X)

# 收尾 drain unpaired flags
```

要点：

- ACC 时 `L0B[Y]` 仍是**本轮加载的旧 Y**（`Y@Y` 不破坏 L0 源），与 SolveTri 一致。
- 末轮不算 `Y@Y`，但必须 `Wait MTE1_M(Y)` 后再 ACC。
- fp32：**禁止** half 的 `LoadData2D`；继续 Catlass `CopyL1ToL0` / `CopyL0CToGm`。

### 1.2 明确不做

| 项 | 原因 |
|----|------|
| MBH / `-M` / `SLOT_MNEG` | 本算子无 RecursiveMerge |
| 改 Catlass `BlockMmad` | Score 路径不动 |
| 同时开 S2b steal | Resource/L0 与 Dual 事件纠缠；先 Dual 单变量 |
| 为双流而双流、不改 GM 往返 | ROI 差；至少中间 X/Y 回灌 L1 供下轮直 Load |

### 1.3 与 Phase A 的 diff

| | Phase A（当前热路径） | Phase B |
|--|----------------------|---------|
| 开关 | `USE_MCH_L0_ACC=1` | + `USE_MCH_L0_DUAL=1` |
| 迭代结构 | `AccFix(X)` 串行再 `Y@Y` | X∥Y 双事件流 |
| L0A/C | 单份复用 | 双份 |
| 中间写回 | 每步 Fixpipe GM 后整段重载 | Fixpipe→GM 平面后**立即回灌 L1**；尽量不在循环外重复 Load |
| `ComputeMchAic` | `MchL0Acc`; `return false` | `MchL0AccDual`; 仍 `false` 直到 S2b 评估 |

---

## 2. S2b Steal 评估（Dual 绿之后）

### 2.1 旧 S2b 语义（闭式）

Y 幂后、等 AIV `I+Y` 的空窗偷发 `MMAD(i+1)`。ACC 已删 `I+Y`，该空窗**不存在**。

### 2.2 ACC 下候选窗口

```text
AIV: WriteSolve → solveReady → [Store ‖ Prep next]
AIC: WaitSolveReady → MchL0AccDual → solveDone
     ↑ 此处若偷 MMAD：与 Dual 争 L0/L1/Resource → 危险

更稳窗口（候选）：
AIC: MchL0AccDual → solveDone
     if iSub+1 < nc:
       WaitReady(next); ComputeMmad(next); SetDone; return true  # skip next Wait+MMAD
# 与 AIV Store(current) 重叠
```

### 2.3 门禁

| 检查 | 通过条件 |
|------|----------|
| 精度 | suite 不回退 |
| 性能 | Task Dur **严格优于** Dual-only 中位（建议 ≥0.05 ms），否则回退 steal |
| 稳定 | 连续 2 次空闲卡 msprof 中位同结论 |
| 冲突 | Dual 事件 drain 干净；`Resource` 串行构造 + `PIPE_ALL` 在 MMAD↔MCH 边界 |

开关建议：`USE_MCH_S2B_STEAL`（默认 0；评估时置 1）。

---

## 3. 分阶段落地（可执行）

| ID | 内容 | 门禁 | 状态 |
|----|------|------|------|
| **B.0** | 本文 + plan；开关草图 | 文档 | ✅ |
| **B.1** | `MchL0AccDual`：双 L0 + EVT_X/Y + L1 回灌；`USE_MCH_L0_DUAL=1` | 编译过 | ✅ |
| **B.2** | 全量精度 suite | all passed | ✅ |
| **B.3** | 空闲卡 msprof vs **5.325** | Dur **4.654**（保留 Dual） | ✅ |
| **B.4** | `USE_MCH_S2B_STEAL` 试偷发 | Dur 4.681 **无优于 Dual** → 关 | ✅ 否决 |
| **B.5** | 更新设计文档；commit | `f110114` | ✅ |

**降级：**

- B.2 红 → 关 `USE_MCH_L0_DUAL`，热路径回 Phase A。
- B.3 Dur 变差 >0.05 ms → 默认关 Dual，文档记「已试」。
- B.4 steal 无收益或红 → 保持 `return false`。

---

## 4. 开关草图

```cpp
#ifndef USE_MCH_L0_DUAL
#define USE_MCH_L0_DUAL 1   // Phase B bring-up; 0 → MchL0Acc (Phase A)
#endif
#ifndef USE_MCH_S2B_STEAL
#define USE_MCH_S2B_STEAL 0 // 仅 Dual 绿后评估
#endif

// ComputeMchAic:
#if USE_MCH_L0_ACC
#  if USE_MCH_L0_DUAL
  MchL0AccDual(slot);
#  else
  MchL0Acc(slot);
#  endif
  SetFlag(solveDone);
#  if USE_MCH_S2B_STEAL
  return TryStealMmadNext(iSub); // WaitReady+MMAD+SetDone or false
#  else
  return false;
#  endif
#endif
```

---

## 5. 验收

| 项 | 标准 |
|----|------|
| 功能 | `test_npu_chunk_kda_fwd_intra_sub_chunk.py` 全绿 |
| 同步 | WriteSolve 后仍 **1×** solveReady/Done；Store 读 `SOLVE_X` |
| 性能 | 相对 5.325：Dual 不显著回退；有下降更好；steal 须再优 |
| 回退 | `USE_MCH_L0_DUAL=0` ≡ 当前 Phase A；`USE_MCH_L0_ACC=0` ≡ S5b |

---

## 6. 相关文件

| 路径 | 角色 |
|------|------|
| `op_kernel/chunk_kda_fwd_intra_sub_chunk.cpp` | `MchL0Acc` / 新增 `MchL0AccDual` |
| `gdn/.../solve_tri/op_kernel/solve_tri_cube.h` | 双流金标准 |
| `L0_ACC_MCH_DESIGN.md` | Phase A 母本；§5.4 伪代码 |
| `CUBE_PIPELINE_DESIGN.md` | 墙钟年表 |
| `prof_intra_sub_chunk_l0acc/` | Phase A 对照 csv |

---

## 7. 一句话

**先按 SolveTri 骨架上 X∥Y 双事件 + L1 回灌（`USE_MCH_L0_DUAL`）；精度与 msprof 过后再用「solveDone 后偷下一拍 MMAD」评估 S2b；任一刀无收益就关开关保留 Phase A。**
