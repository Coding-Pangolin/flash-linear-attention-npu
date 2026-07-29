# ChunkKdaBwdWyDqkgFused — 代码量与可行优化方向

> 日期：2026-07-29  
> 板端现状：model Task Dur med ≈ **5.89 ms**（E1 on；stretch 0.8 ms）  
> 仿真：`results/prof_sim_t1024_p1a` — **AIV-bound**（vec/cube cycles ≈4.6×）；BAR≈30%、MOVEMASK≈12%、Cube WAIT/BAR≫MMAD  
> 约束：单变量宏 · suite 绿 · Δ≤−0.05 ms 才 default on · 不回退 §5.2 反模式

---

## 1. 代码量统计

路径：`fla/ops/ascendc/kda/chunk_kda_bwd_wy_dqkg_fused/`（不含 `results/`、`arch35/` 副本）

### 1.1 按文件（原始行数）

| 文件 | LOC |
|------|-----|
| `op_kernel/chunk_kda_bwd_wy_dqkg_fused_vector.h` | **2010** |
| `op_kernel/chunk_kda_bwd_wy_dqkg_fused_common.h` | 599 |
| `op_kernel/chunk_kda_bwd_wy_dqkg_fused_cube.h` | 588 |
| `op_kernel/chunk_kda_bwd_wy_dqkg_fused.cpp` | 61 |
| **kernel 小计** | **3258** |
| `op_host` + `op_api` | **706** |
| `test/test_chunk_kda_bwd_wy_dqkg_fused.py` | 184 |
| torch_custom test/prof（`*wy_dqkg*`） | ~268 |
| **实现相关合计（kernel+host+local test）** | **≈4148** |

### 1.2 结构含义

| 项 | 值 |
|----|-----|
| Kernel 非空非纯注释行（粗估） | ≈2800 |
| `__aicore__ inline void` 数量 | ≈60 |
| 代码重心 | **Vector ≈ 62% kernel LOC**（与 AIV-bound 一致） |
| Cube | 调度 + DirectTileGemm 调用，相对瘦 |

**解读**：后续刀应优先动 **Vec 热路径 / Process 调度**，少扩 `DirectTileGemm` 通用层；大改调度时 Cube/Vec **必须镜像**，改动面 ≈ `cube.h` Process + `vector.h` Process（各 ~100–200 行级），不是全文件重写。

---

## 2. 现状结论（优化前提）

```text
~30 ms ──► ~22 ──► ~21.5(I1/I2) ──► ~7.7(Vec/Epilog) ──► ~5.9(I5+Epilog+P1a+E1)
```

| 已证明 | 含义 |
|--------|------|
| L0C accum / L1 A / Prefill=2 / Gate MTE2 PP / Mask slim | 局部正向刀 **边际已尽**（E2–E4 flat 或回归） |
| `USE_FIX_MTE2_OVERLAP` / L0 dbuf | **Cube 双缓冲未开**；model ECC → parked |
| `USE_WIN_SOFT_LEAD_V2` | 试过 **+0.37 ms** → reject |
| 仿真 FIX∩MTE2 = 0 | 与宏一致，不是看错流水 |

**目标分层**

| 档 | Task Dur | 路径 |
|----|----------|------|
| 近期可验收 | ≤ **4 ms** / ≤ **3 ms** | 结构或算法刀 |
| Stretch | ≤ **0.8 ms** | 大概率要切分 / 减工作量，单靠叠流水不够 |

---

## 3. 可行方向总览（按推荐序）

| ID | 方向 | 期望 | 风险 | 建议 |
|----|------|------|------|------|
| **D1** | Mask/Select 路径降本（减 MOVEMASK+BAR） | 中（仿真 12%+BAR） | 精度 | **优先试** |
| **D2** | 协议/握手降频（少 CrossCore + 少 JoinBarrier） | 中（board wait_id10≈3 ms） | hang | 单变量、PEM |
| **D3** | 算法切分 / 多 kernel 流水外重叠 | 大（逼近 0.8） | 接口/融合语义 | stretch 主路径 |
| **D4** | BK/BV retile 改 CV 负载比 | 中 | 精度/WS | D1/D2 后 |
| **D5** | Cube Fix∥MTE2 在 Vec 税下降后再开 | 小～中 | ECC | **勿作主刀** |
| — | 再开 Epilog MTE2 PP / 盲 soft-pipe V2 | — | 已 reject | **禁止重试同构** |

---

## 4. D1 — Mask 路径降本（优先）

### 4.1 问题

Stage3 `tl.where` 因果 mask：仿真 **MOVEMASK ≈12%**，且常夹 `PipeBarrier`。E1（`USE_MASK_SELECT_SLIM`）已减 Duplicate，但 Select/MOVEMASK 主体仍在。

### 4.2 方案

1. **常量表 / Init 一次**：`m_A[i,j]=(i>j)` 位图在 `Init` 生成，热路径只 `And`/`Select` 用预置 mask，禁止每 head 重建。  
2. **按 validRows 收窄**：partial chunk 只 mask `vr×vr`，勿扫满 `BT×BT`。  
3. **能合并的 V 链合并 barrier**：`scale → select → neg` 同依赖链少插 `PipeBarrier<PIPE_V>`。  
4. **可选**：Cube 侧用带 mask 的 Fixpipe/清零上三角（若 Catlass 支持）→ Vec 只做 store（改动大，单独立项）。

### 4.3 伪代码

```text
# Init (once per Process or first use)
maskBits[BT][BT/8] = CausalMaskTable(BT)   # i>j → 1
zeroSelBuf primed                             # E1 already

# Stage3MaskVec(slot, validRows):
  vr = validRows
  Load dAWs[vr,vr] → ub
  # OLD: rebuild mask each call + Duplicate + many BAR
  # NEW:
  ApplyPrebuiltMask(ub, maskBits, vr)         # And / Select only
  # merge: Mul(beta_col) and Select in one V chain
  PipeBarrier(V) once at chain end
  Store → dAMaskedWs
  Set V_MASK
```

```text
宏: USE_MASK_TABLE_APPLY=0→门禁后1
改: vector.h Stage3MaskVec + Init maskBuf_
测: suite(varlen/partial) + model msprof
成: Δ≤−0.05 vs ~5.89
```

---

## 5. D2 — 跨核握手降频

### 5.1 问题

板端 `aiv_scalar_wait_id10≈2.9 ms`；仿真 Cube **WAIT_FLAG≈28%**、Vec **WAIT_FLAG≈8.5%**。  
每 window × 多 BK 的 `C_S1/V_GATE/C_S2` 与 `JoinAivBarrier` 过密。

### 5.2 方案（由易到难）

**D2a — 合并「成组 Set」信用（慎）**  
例如 Stage1：整窗两 head 算完再 **一次** `Set C_S1×headCnt` 的语义不变，但减少 Vec 侧无效自旋（需严格 credit 配对）。

**D2b — BK 批处理握手**  
`nBk=2` 时：Cube 连续 Stage1(BK0)+Stage1(BK1) 再 Set；Vec Kg 双 BK 再统一 Gate——**会拉长气泡**，仅当 PEM 显示 Wait≫计算时试。

**D2c — 减少 AIV↔AIV Barrier**  
`JoinAivBarrier` 仅留在 **共享 WS 归约点**（db/dgk merge）；纯行私有路径禁止 Join。

### 5.3 伪代码（D2c 示例）

```text
# GateOnlyVec / EpilogVec owned rows — private UB
if IsComputeAiv():
  work on OwnedRowBegin..End
  # NO JoinAivBarrier here

# Only at merge:
JoinAivBarrier()                    # once
if IsSub0():
  Reduce dbMergeWs[0]+dbMergeWs[1] → db
  Reduce dgkMergeWs similarly
Set V_GATE / continue               # Cube credit unchanged
```

```text
宏: USE_MERGE_BARRIER_ONLY=0
改: 删散落 Join；PEM 查 hang
禁: 改 C_S*/V_* 顺序而不改对面 Wait
```

### 5.4 伪代码（D2a 窗级 Stage1 信用 — 高风险草案）

```text
# Cube (must mirror Vec)
for h in heads:
  RunStage1(h, iK)
  # delay Set
for h in heads:
  Set C_S1                      # still headCnt Sets if Vec Wait headCnt times
# 若改为「一次 Set 两 credit」：Vec 只能 Wait 一次 — 双方同改

# Vec
for h: Kg(h)                    # still ∥ Cube, no Wait
for h: Wait C_S1; Gate(h); Set V_GATE
```

**落地前**：用仿真/日志数清 Set/Wait 次数；**单变量**；失败立即回滚。

---

## 6. D3 — 算法切分（逼近 0.8 的主路径）

### 6.1 问题

单 kernel 内串完 WyV+KvAcc+GateWy+DaFinal，AIV 协议税无法靠 dbuf 消掉。  
5.89→0.8 ≈ **7.3×**，需 **减单次 launch 工作** 或 **多核间流水外重叠**。

### 6.2 方案 A：按 Stage 拆 L0（推荐评估）

```text
OpA: Stage0 + Stage1 partial (dq/dk/dw + kg)     # 或 WyV 独立
OpB: Stage2 GateWy + Epilog
OpC: Stage3 DaFinal
```

Host `aclnn` 内顺序调用；**不同 chunk 可多流**（chunk 无依赖）：

```text
for chunk in parallel streams:
  OpA(chunk); OpB(chunk); OpC(chunk)

# 或
stream0: OpA(c0); OpB(c0); OpC(c0)
stream1: OpA(c1); ...     # 与 stream0 重叠
```

伪代码（Host）：

```text
aclnnChunkKdaBwdWyDqkgFused(...):
  if (!use_split) return fused_kernel(...)

  for task in tasks:   # or graph
    aclnnWyDqkgStageA(..., stream[task % N])
    aclnnWyDqkgStageB(..., stream[task % N])  # wait A on same stream
    aclnnWyDqkgStageC(..., stream[task % N])
  sync_all
```

**收益**：单 kernel 更短 → BAR/MOVEMASK 占比可降；多 stream 提高设备占用。  
**代价**：新 op / 新 WS / 精度契约；**非一刀宏**。

### 6.3 方案 B：保持融合，减数学工作

- GVA：HV 维输出已在外 reduce——确认无重复算。  
- `V=128` 时评估更大 `MAX_BV`（若 UB 允许）减少 `nBv` 循环（与 accum 叠加）。  
- Mask 后 `dA@A`×2：评估是否可用更粗粒度 / 融合 Fix（独立精度 plan）。

### 6.4 伪代码（方案 B：BV 放大）

```text
#if USE_BV128_IF_UB
  MAX_BV = 128   # V=128 → nBv=1；Gate UB 峰值重算
#else
  MAX_BV = 64
#endif

# Stage1 L0C accum already: nBv=1 → 单次 MMAD+Fix（自动变轻）
# Gate: 无 iv 循环
```

先做 **UB 峰值表 + suite**，再开宏。

---

## 7. D4 — Retile 改绑定侧

### 7.1 动机

Cube MMAD≈1% cycles，空等 Vec。若 **加长 Cube 有效重叠窗口**（更大 BK 连续 MMAD）同时 **缩短 Vec Gate 次数**，可能改善比值。

### 7.2 伪代码

```text
# Host tiling / common.h
MAX_BK = 128      # K=128 → nBk=1  (今日 64→2)
# 则每 head GateWy 三明治次数 ÷2

Cube:
  RunStage1 once (full K)
  Set C_S1
Vec:
  Kg once; Wait; Gate once; ...
```

**风险**：UB `6*BT*BK` 随 BK 涨；Gate arena 可能爆 192KB → 需减并行驻留或 spill。

---

## 8. D5 — Cube 双缓冲（仅辅刀）

### 8.1 现状

```text
USE_L0_AB_DBUF=0
USE_FIX_MTE2_OVERLAP=0
# DirectTileGemm: MTE2→…→MMAD→Fix→Wait 全串行
```

### 8.2 目标形态（伪代码）

```text
# DirectTileGemm with USE_FIX_MTE2_OVERLAP=1
copyGmToL1B; copyGmToL1A
if pipe.fixOutstanding: Wait(FIX_MTE2)
L1→L0; MMAD
if doFix:
  Fixpipe
  Set(FIX_MTE2); pipe.fixOutstanding = true
  # NO Wait — next gemm's MTE2 overlaps this Fix

# Prefetch next tile (optional L1A dbuf — separate macro):
# while MMAD(tile i): MTE2 issue tile i+1 into l1 ping
```

```text
启用条件（全部满足）:
  1) D1/D2 后 board aiv wait 明显下降或 cube_ratio 上升
  2) suite + model 无 507015
  3) 仿真 FIX∩MTE2 从 0 → 稳定非空
否则保持 0
```

---

## 9. 推荐落地顺序与伪迭代

```text
Week 1:  D1 Mask table/apply          目标: −0.05~−0.3 ms
Week 1–2: D2c Merge-barrier-only      目标: 降 wait_id10；防 hang
Week 2:  D4 试 nBk=1 (BK=128) 或 BV   UB 过则停
若仍 >4 ms:
Week 3+: D3 切分方案设计 + 原型 OpA/B/C
D5 全程辅刀，不挡主路径
```

每刀模板：

```text
1. 单宏 / 单调度变体
2. test_npu_chunk_kda_bwd_wy_dqkg_fused.py
3. msprof model → med；记 ITER_LOG
4. Δ≤−0.05 → default on；否则 0 + 保留代码
```

---

## 10. 明确不做（避免重复踩坑）

| 项 | 原因 |
|----|------|
| 同构再开 `WIN_SOFT_LEAD_V2` / Epilog MTE2 PP | 已 +0.37 / +0.30 |
| Gate 无同步三路 DataCopy | hang |
| I5b Post≻WaitFree | AIV stall |
| Vec-bound 下强开 FIX∥MTE2 当主收益 | ECC + 期望小 |
| 单 head 串 Stage0–3 | 毁成组重叠 |
| Kg 挂回 Wait(C_S1) 后 | 毁 Stage1∥Kg |

---

## 11. 与现有文档

| 文档 | 角色 |
|------|------|
| [ITER_LOG.md](ITER_LOG.md) | 板端年表与宏状态 |
| [P4_SOFTPIPE_PLAN.md](P4_SOFTPIPE_PLAN.md) | soft-pipe 草案；V2 已试失败，新变体需换重叠面 |
| [DESIGN.md](DESIGN.md) | 契约 |
| [results/SIM_T1024_P1A_SUMMARY.md](results/SIM_T1024_P1A_SUMMARY.md) | 仿真依据 |

---

## 12. 一句话

代码量 **kernel ≈3.3k 行（Vec 占六成）**；性能已到 **~5.9 ms / AIV-bound**。  
**可行下一程**：先 **D1 Mask 降本 + D2 握手降频**，再评估 **D4 retile / D3 切分**；Cube dbuf 仅作辅刀。  
**0.8 ms 不能再按「再开双缓冲」规划**，应按 **减单核工作或多 kernel 外重叠** 立项。
