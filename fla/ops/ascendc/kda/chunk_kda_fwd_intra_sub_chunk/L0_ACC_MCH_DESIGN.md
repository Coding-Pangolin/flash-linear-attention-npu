# Intra-sub-chunk：SolveTri 风格 L0 ACC MCH 详细方案

> 状态：**L0 ACC 热路径已落地**（`USE_MCH_L0_ACC=1`）。  
> 精度 suite 全绿；空闲卡 msprof Task Dur 中位 ≈ **5.33 ms**（S5b 基线 6.59 → −19%）。  
> 关键修复：ACC 必须对标 SolveTri **双 L0B 预载**（I→L0B[0]、Y→L0B[TILE]），禁止在 `X@I` 与 ACC 之间 `M_MTE1` 覆盖同一 L0B 槽（曾导致 `akkd_rel≈2300`）。

参考实现：

- `fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/op_kernel/solve_tri_cube.h`
  - `MCH_InitXY` / `MCHInvertDiagonal`
  - `Mmad_Offset(..., cmatrixInitVal)` / `Mmad_ACC_Offset`（`cmatrixInitVal=false`）
  - `StoreL0CToSlot_*`（L0C → scratch GM → L1 Nd2Nz）

| 项         | 值                                                                 |
| ---------- | ------------------------------------------------------------------ |
| 问题规模   | **BC = 16×16，fp32**（`MAX_BC=16`）                       |
| 目标算子   | `chunk_kda_fwd_intra_sub_chunk` tiling key 1                     |
| Score GEMM | 继续 Catlass**`BlockMmad`**（`ComputeMmad`）             |
| MCH        | op-local Tile / AscendC`Mmad`，**不改 Catlass Block 模板** |
| 不做       | `aclnnSolveTri`、MBH、扁平 `×NC`、改 Score 路径               |
| 性能基线   | S5b：~6.59 ms；`aiv_scalar≈2.08`；`aic_scalar≈2.77`          |
| L0 ACC 实测 | ~**5.33 ms**；`aiv≈1.62`；`aic≈1.02`；`mac≈0.31`（device1） |
| 性能目标   | ~**5 ms**（已接近；可选 Phase B / 恢复 S2b steal）           |

---

## 1. 数学与两条等价路径

### 1.1 本算子要求

对严格下三角块 \(L\)（GM：`cmatWs_[PLANE_AKK]`），求

\[
X = (I - L)^{-1}
\]

AIV `WriteSolveInputs` 已写出：

- \(L\) → `cmat` AKK
- \(X_0 = I - L\) → `SOLVE_X`

与 GPU Triton / CPU golden 的 forward-sub 目标一致（见 `test_chunk_kda_fwd_intra_sub_chunk.py`）。

### 1.2 经典 Neumann（SolveTri / 目标热路径）

\[
Y_0 = L^2,\quad
X \leftarrow X + X Y,\quad
Y \leftarrow Y^2
\quad(\times\ \lceil\log_2 BC\rceil = 3)
\]

初值 \(X_0 = I - L\)。迭代后 \(X \approx (I-L)^{-1}\)。

**L0 ACC 关键语义**（对标 `Mmad_ACC_Offset`）：

```text
L0C = X @ I          # cmatrixInitVal = true
L0C += X @ Y         # cmatrixInitVal = false  ★
Fixpipe → GM / 回灌 L1
```

### 1.3 当前热路径 S5b（闭式，已验证）

利用 \((I-L)^{-1} = (I+L)(I+L^2)(I+L^4)\)（BC=16 时幂到 \(L^8\)）：

```text
Y0 = L@L, Y1 = Y0@Y0, Y2 = Y1@Y1
Pk = I + Yk          # AIV AddIdentityToMchPlanes
X_new = X0 @ P0 @ P1 @ P2   # 落在 SOLVE_TMP
```

代价：每笔 `CubeGemmSolve` = `BlockMmad` **必 Fixpipe 出 GM**；`I+Y` 在 AIV → **多一轮 CV 握手**。

### 1.4 为何切 L0 ACC

| 开销源            | S5b                                   | L0 ACC 目标                       |
| ----------------- | ------------------------------------- | --------------------------------- |
| AIC↔AIV 握手     | WriteSolve 后**2×** Ready/Done | **1×**                     |
| AIV`I+Y` 三平面 | 有                                    | 无                                |
| 中间 GEMM Fixpipe | 6 次 GM 往返量级                      | 每轮 X/Y 各一次；中间乘积留在 L0C |
| `aic_scalar`    | 高于`aiv_scalar`                    | 期望下降                          |

---

## 2. 现状流水 vs 目标流水

### 2.1 现状（S5b）

```text
AIV: PrepareSub → ready
AIC: WaitReady → ComputeMmad → done
AIV: WaitDone → WriteSolve(L,X0) → solveReady
AIC: WaitSolveReady
     Y0=L@L, Y1=Y0², Y2=Y1² → solveDone     [可偷下一拍 MMAD]
AIV: WaitSolveDone → Pk=I+Yk → solveReady   ← 残差 CV
AIC: WaitSolveReady → P0@P1→T@P2→X0@T → TMP → solveDone
AIV: WaitSolveDone → Store(TMP)
```

### 2.2 目标（经典 Neumann + L0 ACC）

```text
AIV prologue: 写常驻 I → SolveOff(0, SOLVE_Y1)（核生命周期一次）
AIV: PrepareSub → ready
AIC: WaitReady → ComputeMmad → done
AIV: WaitDone → WriteSolve(L,X0) → solveReady     # 仅此一次 MCH 握手
AIC: WaitSolveReady → MchL0Acc(slot) → Fixpipe X → solveDone
AIV: WaitSolveDone → Store(SOLVE_X)
# 删除：AddIdentityToMchPlanes / 第二次 solveReady / CubeGemmSolve×6
```

**同步约束（L0.1 起强制）**：先关 S2b「MCH 间隙偷 MMAD」，避免与 `MchL0Acc` 争 `Resource`/L0；精度与 msprof 稳定后再评估是否恢复 steal。

---

## 3. 对标 SolveTri：映射表

| 概念                    | SolveTri (`SolveTriCube`)              | 本算子                                                            |
| ----------------------- | ---------------------------------------- | ----------------------------------------------------------------- |
| 输入矩阵\(A\)（对角块） | L1`SLOT_INPUT`                         | GM`cmat` AKK → L1 `l1L`                                      |
| \(X_0=I-A\)             | `MCH_InitXY`：`I*I` 再 `+= (-I)@A` | AIV 已写`SOLVE_X`；AIC 只加载                                   |
| \(Y=A^2\)               | `MCH_InitXY`：`A@A` → `SLOT_Y`    | `Y = L@L` → `SOLVE_Y0` → L1 `l1Y`                         |
| 单位阵\(I\)             | L1`SLOT_I`（`PrepareConstants`）     | L1`l1I`；GM 源 `SOLVE_Y1(slot0)`                              |
| \(X\leftarrow X+X@Y\)   | `Mmad(X@I,init)` + `Mmad_ACC(X@Y)`   | `MchMatmulL1AccFix(..., doAcc=true)`                            |
| \(Y\leftarrow Y@Y\)     | `Mmad(Y@Y,init)` → scratch_Y → L1    | `doAcc=false` → `SOLVE_Y0` → L1                             |
| 迭代次数                | `NUM_ITERS=3`                          | `MCH_ITERS=3`                                                   |
| L0 双缓冲               | `X_BUF=0`, `Y_BUF=TILE_LEN`          | **Phase A 可单缓冲**；Phase B 再双流                        |
| Fixpipe 中转            | 每核`scratchGM_` / `scratchGM_Y_`    | 复用`SOLVE_X` / `SOLVE_Y0`（或 TMP）                          |
| 数据类型                | L0A/B 多为**half**；L0C float      | **全程 fp32**（必须 Catlass `LoadData3D`/`CopyL1ToL0`） |
| MBH /`-M`             | 有（最后一轮并行）                       | **不做**                                                    |
| 规模                    | 64/128+（多 fractal）                    | **单 fractal 16×16**（`NUM_FRACS=1`）                    |

**工程层选择**：不直接 `#include` SolveTri（half 路径 / 槽位 / MBH 耦合重）。在本 cpp 内用 **Catlass Tile 栈**（同 `chunk_gated_delta_rule_bwd_dhu` 跳过 `BlockMmad`）实现等价 ACC；事件语义对齐 SolveTri。

**严禁**：对 fp32 抄 SolveTri half 的 `LoadData2D`（`repeatTimes=1`）——已实测数值炸毁。

---

## 4. 资源与槽位

### 4.1 GM（已有，不增平面）

| 平面          | 用途（S5b）    | L0 ACC 用途                           |
| ------------- | -------------- | ------------------------------------- |
| `cmat` AKK  | \(L\)          | 入 L1；只读                           |
| `SOLVE_X`   | \(X_0\) / 中间 | 入\(X_0\)；出最终 \(X\)（Store 读此） |
| `SOLVE_Y0`  | \(Y_0=L^2\)    | Y 的 Fixpipe 中转                     |
| `SOLVE_TMP` | 闭式积         | 可选 scratch；Store 不再依赖          |
| `SOLVE_Y1`  | \(Y_1\) / 闭式 | **常驻 I**（slot0 写一次）      |

### 4.2 AIC 片上（`Catlass::Arch::Resource`）

```text
L1:  [I | X | Y | L | Btmp]   各 BC×BC×sizeof(float)
     Btmp：与 L/Y 同内容的独立 B 槽，避免 L1A/L1B 别名
L0A / L0B / L0C: 各 ≥ 1×16×16（Phase B 可双份对齐 SolveTri）
```

`ComputeMmad` 与 `MchL0Acc` **串行**，各自构造 `Resource`，末尾 `PipeBarrier<PIPE_ALL>`（与现多次 `CubeGemmSolve` 一致）。

### 4.3 AIV

- Prologue：一次 `FillIdentity` → `SolveOff(0, SOLVE_Y1, 0, 0)`
- 热路径：`WriteSolve` + `Store(SOLVE_X)`；去掉 `AddIdentityToMchPlanes`

---

## 5. 完整伪代码

### 5.1 SolveTri 原语（对照用，半伪代码）

摘自 `solve_tri_cube.h` 语义压缩版（略 MBH / 多 fractal）：

```text
# ---- 原语 ----
Mmad_Offset(buf, initC):
  Mmad(L0C[buf], L0A[buf], L0B[buf], cmatrixInitVal=initC)

Mmad_ACC_Offset(xBuf, yBuf):
  Mmad(L0C[xBuf], L0A[xBuf], L0B[yBuf], cmatrixInitVal=false)

StoreL0CToSlot(slot, l0cOff, scratch):
  Fixpipe L0C[l0cOff] → scratchGM
  Nd2Nz DataCopy scratchGM → L1[slot]

# ---- 初始化：Y=A², X=I-A ----
MCH_InitXY():
  Load L0A/B[Y] ← SLOT_INPUT;  Mmad_Offset(Y, true); Store → SLOT_Y
  Load L0A/B[X] ← SLOT_I;      Mmad_Offset(X, true)          # = I
  Load L0A[X] ← SLOT_INEG; L0B[X] ← SLOT_INPUT
  Mmad_Offset(X, false)                                      # += (-I)@A → I-A
  Store → SLOT_X

# ---- 主循环 ×3 ----
MCHInvertDiagonal():
  MCH_InitXY()
  for iter in 0..2:
    Load L0A/B[X] ← SLOT_X, SLOT_I
    Load L0A/B[Y] ← SLOT_Y, SLOT_Y
    Mmad_Offset(X, true)                    # L0C_X = X@I
    if iter < 2:
      Mmad_Offset(Y, true); StoreY → SLOT_Y # Y = Y@Y
    PipeBarrier<PIPE_M>()
    Mmad_ACC_Offset(X, Y)                   # L0C_X += X@Y
    StoreX → SLOT_X
```

本算子 **不搬 Init 里的 `I-A` Cube 路径**（AIV 已写好 \(X_0\)）；只搬主循环的 ACC 语义。

### 5.2 本算子：原语层

```text
# GM 已是 RowMajor fp32；进 L1 必须走 Catlass Nd2Nz（CopyGmToL1A）
MchLoadGmToL1(l1, gm, base):
  CopyGmToL1A(l1 ← gm[base], shape=BC×BC)
  Set/Wait MTE2_MTE1

MchFixpipeToGm(l0C, gm, base):
  CopyL0CToGm(gm[base] ← l0C)   # Catlass Fixpipe 路径
  Set/Wait FIX_MTE2

# C = A@B0；若 doAcc：C += A@B1。L1 已是 Nz。
MchMatmulL1AccFix(l1A, l1B0, l1B1, l0A, l0B, l0C, gmC, baseC, doAcc):
  CopyL1ToL0A(l0A ← l1A)
  CopyL1ToL0B(l0B ← l1B0)
  Set/Wait MTE1_M
  TileMmad(l0C, l0A, l0B, initC=true)

  if doAcc:
    Set/Wait M_MTE1
    CopyL1ToL0B(l0B ← l1B1)     # 注意：A 仍在 L0A，勿破坏
    Set/Wait MTE1_M
    PipeBarrier<PIPE_M>()       # 对齐 SolveTri：防 L0C RAW/WAW
    TileMmad(l0C, l0A, l0B, initC=false)   # ★ ACC

  MchFixpipeToGm(l0C, gmC, baseC)
```

### 5.3 本算子：`MchL0Acc`（经典 Neumann）

```text
MchL0Acc(slot):
  lBase  = CmatOff(slot, PLANE_AKK, 0, 0)
  xBase  = SolveOff(slot, SOLVE_X, 0, 0)
  yBase  = SolveOff(slot, SOLVE_Y0, 0, 0)
  eyeBase= SolveOff(0,    SOLVE_Y1, 0, 0)   # 常驻 I

  alloc L1: I, X, Y, L, B   # 5× BC² float，互不 overlapping
  alloc L0A, L0B, L0C

  MchLoadGmToL1(L, cmatWs_, lBase)
  MchLoadGmToL1(B, cmatWs_, lBase)          # B 独立副本（禁 L1 别名）
  MchLoadGmToL1(X, solveWs_, xBase)
  MchLoadGmToL1(I, solveWs_, eyeBase)

  # Y = L @ L
  MchMatmulL1AccFix(L, B, B, ..., solveWs_, yBase, doAcc=false)
  MchLoadGmToL1(Y, solveWs_, yBase)

  for iter in 0 .. MCH_ITERS-1:             # 3
    # X ← X + X@Y  （L0C 内 ACC，一次 Fixpipe）
    MchMatmulL1AccFix(X, I, Y, ..., solveWs_, xBase, doAcc=true)
    if iter + 1 < MCH_ITERS:
      MchLoadGmToL1(X, solveWs_, xBase)
      MchLoadGmToL1(B, solveWs_, yBase)     # B ← Y 副本
      MchMatmulL1AccFix(Y, B, B, ..., solveWs_, yBase, doAcc=false)
      MchLoadGmToL1(Y, solveWs_, yBase)

  PipeBarrier<PIPE_ALL>()
  # 最终 X 在 SOLVE_X；Store 读此平面
```

### 5.4 Phase B（可选）：SolveTri 双缓冲流水版

在单缓冲精度绿且仍要抠 latency 时，再引入与 `MCHInvertDiagonal` 同构的双流：

```text
MchL0Acc_DualBuf(slot):
  # L1 常驻：I, X, Y, L（与上同）；L0A/B/C 各双份 offset 0 / TILE
  # EVT_X / EVT_Y 两套 HardEvent（MTE1_M, M_FIX, FIX_M, M_MTE1）

  # Init Y=L@L on Y_BUF; X already in L1 from GM
  LoadToL0(Y_BUF ← L,L); Mmad(init); StoreY → L1_Y / SOLVE_Y0

  SetFlag FIX_M / M_MTE1 for both streams   # 循环 priming

  for iter in 0..2:
    Wait M_MTE1(X); Load L0A/B[X] ← X, I; Set MTE1_M(X)
    Wait M_MTE1(Y); Load L0A/B[Y] ← Y, Y; Set MTE1_M(Y)

    Wait FIX_M(X); Wait MTE1_M(X); Mmad_Offset(X, true)     # X@I

    if iter < 2:
      Wait FIX_M(Y); Wait MTE1_M(Y); Mmad_Offset(Y, true)
      StoreY → L1_Y; Set FIX_M(Y)

    PipeBarrier<PIPE_M>()
    if iter == 2: Wait MTE1_M(Y)   # 最后一轮 Y 只作 B 源
    Mmad_ACC_Offset(X, Y)           # L0C_X += X@Y
    Set M_FIX(X); Wait; StoreX → L1_X / SOLVE_X
    Set M_MTE1(X,Y); Set FIX_M(X)

  drain unpaired flags
```

Phase A **先不要**上双缓冲，降低事件配对风险。

### 5.5 CV / 热路径接线伪代码

```text
# ---------- AIV ----------
ProcessChunkAiv_prologue():
  if subBlockIdx == 0:
    FillIdentity(UB)
    CopyVectorOut(solveWs_, SolveOff(0, SOLVE_Y1, 0, 0), UB, BC*BC)
  CrossCoreBarrier(...)   # 两 AIV 都看到 I

ProcessChunkAiv():
  PrepareSub(slot0); SetFlag(ready)
  for iSub in 0..nc-1:
    slot = iSub % DEPTH
    WaitFlag(done)
    PostSubWriteSolve(...)                  # 写 L + X0
    CrossCoreBarrier(MTE3)
    SetFlag(solveReady)                     # ★ 唯一 MCH kick

    if iSub+1 < nc:
      PrepareSub(next); SetFlag(ready)

    WaitFlag(solveDone)                     # 取代 PostSubMchClosedForm 全身
    PostSubStore(..., src=SOLVE_X)          # 不再读 SOLVE_TMP

# ---------- AIC ----------
ProcessChunkAic():
  for iSub in 0..nc-1:
    slot = iSub % DEPTH
    WaitFlag(ready)
    ComputeMmad(slot)
    SetFlag(done)

    WaitFlag(solveReady)
    MchL0Acc(slot)                          # 取代 CubeGemmSolve×6
    SetFlag(solveDone)
    # 不开 stealNext MMAD

ComputeMchAic_legacy_S5b(...):              # 保留，#ifdef / 开关回退
  ... existing closed-form ...
```

### 5.6 函数级 diff 清单

| 符号                         | 动作                                                              |
| ---------------------------- | ----------------------------------------------------------------- |
| `ComputeMchAic`            | 调`MchL0Acc`；单次 `solveDone`；`return false`（关 steal）  |
| `PostSubMchClosedForm`     | 仅`Wait(solveDone)`；删 `AddIdentity` + 第二次 `solveReady` |
| `AddIdentityToMchPlanes`   | 热路径删除（可留死代码至 L0.4）                                   |
| `PostSubStore`             | 读**`SOLVE_X`**（非 `SOLVE_TMP`）                       |
| `ProcessChunkAiv` prologue | 写常驻`I`                                                       |
| `CubeGemmSolve`            | 保留回退 / 对照单元                                               |
| `MchL0Acc` 等              | 从 WIP → 热路径；失败则切回 S5b                                  |

---

## 6. Bring-up 教训（必须遵守）

| 尝试                             | 结果                             | 结论                                           |
| -------------------------------- | -------------------------------- | ---------------------------------------------- |
| 直接切热路径 L0 ACC Neumann      | `akkd_rel≈2300`；`aqk` 正常 | ACC/搬运未对齐，**禁止一步切流**         |
| GM→L1 裸`DataCopy`            | 炸到 1e16                        | float 必须**Nd2Nz**                      |
| `CopyGmToL1A`                  | 改善但仍 ~2300                   | 搬运对了仍可能有 ACC / 别名 / 事件问题         |
| AscendC`LoadData2D`（照 half） | 错误                             | fp32 → Catlass`LoadData3D` / `CopyL1ToL0` |
| Catlass Tile + 独立 L1 slot      | 仍 ~2300                         | 需**单 GEMM / 单 ACC 对照** 再叠迭代     |

**根因假设（待 L0.1 证伪）**：

1. `initC=false` 与 Catlass `TileMmad` 语义不一致（未设 `cmatrixSource` 等）
2. Fixpipe 后再读 X/Y 的 hazard 不足
3. `X@I` 用单位阵乘代替「保持 L0C=X」时，I 未正确 Nd2Nz
4. L1A/L1B 曾别名（已用 `l1B` 规避，需确认热路径版本仍分离）

---

## 7. 分阶段落地（可执行）

### L0.0 — 文档与开关（本文）

- [X] 方案 MD
- [X] 编译期开关：`USE_MCH_L0_ACC`（当前默认 **1**；置 0 回退 S5b）

### L0.1 — 单笔 GEMM 数值对齐（**精度门禁，不开热路径**）

在 AIC debug 分支或临时测试核上，同一组 GM `A,B`：

```text
ref  = CubeGemmSolve(A, B → C_ref)
dut  = MchMatmulL1AccFix(A, B, *, doAcc=false → C_dut)
assert max_abs(C_dut - C_ref) < atol   # 建议 atol=1e-4 ~ 1e-3 fp32
```

门禁：不过则 **禁止** 开 ACC / 切热路径。 **[已过：`USE_MCH_L0_GEMM=1` 全 suite 绿]**

### L0.2 — 单笔 ACC 对齐

```text
# 参考：两次 BlockMmad + AIV Add
T0 = A @ B0
T1 = A @ B1
C_ref = T0 + T1

C_dut = MchMatmulL1AccFix(A, B0, B1, doAcc=true)
assert max_abs(C_dut - C_ref) < atol
```

可选：对照 SolveTri 同尺寸 half 路径不要求 bit 一致，只要求与本算子 `CubeGemmSolve` 链一致。  
**[实际：同槽 L0B 中途改写 → `akkd_rel≈2300`；改为双 L0B 预载后随 L0.3/L0.4 一次过]**

### L0.3 — 完整 `MchL0Acc` vs S5b / golden

- 固定输入（小 batch）：`MchL0Acc` 输出 `SOLVE_X` vs S5b `SOLVE_TMP` vs CPU `(I-L)^{-1}`
- 门禁：`akkd` 相对误差回到 suite 阈值内 **[已过]**

### L0.4 — 热路径切换

- `USE_MCH_L0_ACC=1`：`ComputeMchAic` → `MchL0Acc`；关 steal；改 Store / 砍 I+Y CV
- 跑 `test_npu_chunk_kda_fwd_intra_sub_chunk.py` **[all cases passed]**

### L0.5 — 性能

- 空闲卡：`prof_chunk_kda_fwd_intra_sub_chunk_model.py`
- 对比 Task Duration、`aiv_scalar`、`aic_scalar` vs 6.59 ms
  - **实测（device1）**：Dur **5.325** / aiv **1.621** / aic **1.016** / mac **0.308**（S5b：6.588 / 2.076 / 2.766 / 0.266）
- 可选：Phase B 双缓冲；再评估 S2b steal

### L0.6 — 收尾

- 删死代码或明确 `#if` 回退（S5b 路径仍由 `USE_MCH_L0_ACC=0` 保留）
- 更新 `CUBE_PIPELINE_DESIGN.md`

**降级策略**：任一精度门禁失败 → 保持 S5b；或仅 ACC `X+=X@Y`、Y 幂仍 `CubeGemmSolve`（半切，验证 ACC 单独收益）。

---

## 8. 验收标准

| 项     | 标准                                                      |
| ------ | --------------------------------------------------------- |
| 功能   | NPU suite 不回退；`akkd`/`aqk` 阈值内                 |
| 同步   | WriteSolve 后仅**1** 次 solveReady/Done             |
| 热路径 | 无三平面`I+Y`；Store 读 `SOLVE_X`                     |
| 性能   | 空闲卡 Task Dur 相对 6.59 ms 有可见下降（目标方向 ~5 ms） |
| 回退   | `USE_MCH_L0_ACC=0` 行为与当前 S5b 一致                  |

---

## 9. 风险与非目标

| 风险                                | 缓解                                         |
| ----------------------------------- | -------------------------------------------- |
| fp32 Tile ACC 与 BlockMmad 数值漂移 | L0.1/L0.2 强制对照                           |
| HardEvent 配对 / 与 Score MMAD 冲突 | 关 steal；独立`Resource`；Barrier          |
| L1 容量（5×16×16 float）          | Atlas A2 / 950 足够；不足则砍 Btmp、串行重载 |
| 过早双缓冲                          | Phase A 单缓冲优先                           |

非目标：MBH、调用 `SolveTriCube` 整核、改 `bc_≠16`、改 Score `BlockMmad`。

---

## 10. 相关文件

| 路径                                             | 角色                    |
| ------------------------------------------------ | ----------------------- |
| `op_kernel/chunk_kda_fwd_intra_sub_chunk.cpp`  | 实现 / WIP helpers      |
| `gdn/.../solve_tri/op_kernel/solve_tri_cube.h` | ACC / 双缓冲参考        |
| `CUBE_PIPELINE_DESIGN.md`                      | CV 流水总览             |
| `SCALAR_BOTTLENECK_ANALYSIS.md`                | 历史标量瓶颈与 MCH 动机 |
| `test/test_chunk_kda_fwd_intra_sub_chunk.py`   | 精度门禁                |

---

## 附录 A：闭式 vs Neumann flop 对照（BC=16）

```text
S5b CubeGemmSolve 次数（每 sub）：
  Y 幂 3 次 + 乘积链 3 次 = 6 × (16³ MAC) + AIV 3×(I+Y)

Neumann L0 ACC：
  Y=L@L 一次
  每轮：X@I + X@Y（ACC）+（非末轮）Y@Y
  ≈ 1 + 3×2 + 2 = 9 次 16³ MAC，但其中 ACC 省一次 Fixpipe/GM
  且去掉 AIV I+Y 与一轮 CV
```

收益主要来自 **同步税 + Fixpipe/GM**，而非裸 MAC 计数减少。

## 附录 B：开关接线草图（C++）

```cpp
#ifndef USE_MCH_L0_ACC
#define USE_MCH_L0_ACC 0
#endif

__aicore__ inline bool ComputeMchAic(uint64_t slot, uint64_t iSub)
{
    Catlass::Arch::CrossCoreWaitFlag(solveReadyFlag_);
#if USE_MCH_L0_ACC
    (void)iSub;
    MchL0Acc(slot);
    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(solveDoneFlag_);
    return false; // no steal
#else
    // existing S5b closed-form + optional steal
    ...
#endif
}
```
