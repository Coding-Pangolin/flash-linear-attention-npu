# Score Tile + CrossCore → 续冲 1.5ms（执行母本）

> 对应 cursor plan：`~/.cursor/plans/score_tile_crosscore_8506a51b.plan.md`  
> 基线 Dual ≈ **4.654 ms** · 目标 **1.5 ms**  
> 门禁：每阶段精度过 → commit 记账 → 再下一阶段

## 年表

| 阶段 | 状态 | commit / 备注 |
|------|------|----------------|
| T0 -g + 本文档 | **done** `232ce89` | op_host `-g`；母本落盘 |
| T1 Score Tile | **done** `7e329a5` | 精度绿；Dur **4.627** vs Dual 4.654 |
| T2 CrossCore | **done** | Identity 后移 + AIC Resource 复用；Dur **4.402** |
| T3 sim -g | in progress | |
| T3 sim -g | pending | |
| T4 MCH L1 | pending | |
| T5 S2c | pending | |
| T6 prefetch/S4 | pending | |
| T7 ceiling | pending | |

---

**执行母本（T0 落盘，之后每阶段更新年表）：**

`fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/SCORE_TILE_CROSSCORE_PLAN.md`

**必须对照的既有分析（后半段优化仍做）：**

- `.../TARGET_1P5_ANALYSIS.md`（板端 pipe + 仿真旗/instr + 原 P0/P1）
- `.../L0_ACC_MCH_DESIGN.md` / `PHASE_B_DUAL_PLAN.md` / `CUBE_PIPELINE_DESIGN.md`
- 旧 cursor plan：`intra_sub_chunk_1p5ms_d4e5f6.plan.md`（T1–T4 → 本 plan T4–T7）

**写法要求（对人 / 对 agent）：** 每刀必须能回答「改哪几行、抄哪份参考的哪段、为何适用、门禁是什么」。禁止只写口号。

---

## 门禁（每一阶段强制）

```text
改码 → 精度 suite 过 →（性能刀：空闲卡 msprof 中位记账）
    → git commit（message 写清阶段号与开关）
    → 更新 SCORE_TILE_CROSSCORE_PLAN.md 年表 + TARGET_1P5 若相关
    → 再进入下一阶段
```

精度红或明显性能红：开关回退 → commit 记「试过/回退」→ 不带坏状态进下一刀。单变量。

---

## 1. 问题证据（为何动 Score / Flag / -g）

### 1.1 现状代码（要改的点）

| 位置 | 事实 | 后果 |
|------|------|------|
| [`chunk_kda_fwd_intra_sub_chunk.cpp` L926–969](fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/op_kernel/chunk_kda_fwd_intra_sub_chunk.cpp) `ComputeMmad` | 两次 `blockMmad(qg,kg)` / `blockMmad(w,kg)`，中间 `PipeBarrier<PIPE_ALL>`；注释已写 *BlockMmad may still MTE2 B* | 串行搬算；**kneg 无法保证 L1 驻留** |
| 同文件 L1896–1898 `ProcessChunkAic` | `WaitFlag(ready)` **之后**才进整段 `ComputeMmad`（含每拍新建 `Resource` + 全部 MTE2） | 仿真上：首波 Vector 完成后 Cube 搬运空窗 |
| [`op_host/CMakeLists.txt` L9–13](fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/op_host/CMakeLists.txt) | `OPTIONS` 仅 `--cce-auto-sync=off` 等，**无 `-g`** | T=1024 sim 出现 `Code call stack is empty`，Insight 无行号 |

### 1.2 板端 / 仿真量化（依据 `TARGET_1P5_ANALYSIS.md` §1 / §5.4–5.5）

| 来源 | 证据 | 对方案的含义 |
|------|------|----------------|
| 板端 Dual | Dur 4.654；`aic_fixpipe≈1.13` + `aic_mte2≈1.04` ≫ `mac≈0.25` | 后半仍要做 **MCH 驻留**；Score tile 先消「可证的二次搬 B」 |
| 板端 wait | AIV id6≈1.9 / id2≈1.1；AIC id8≈1.21 / id4≈0.58 | Flag 前后空窗与互等仍在；T2/T4/T5 打不同 wait |
| Sim T=1024 `instr_exe` | Cube FIXP≈30%、`MOV_OUT_TO_L1_MULTI_ND2NZ`≈14%、**MMAD 指令 ≈0.8%** | 不是刷 mac；搬运/Fixpipe/BAR 才是 |
| Sim 旗时序 | MMAD≈MCH≈5k tick；Prep 相对 MCH **提前 ≈−2.5k** | **否决再抠 Prep(S2a)**；Score/MCH 体与握手才是刀口 |
| 用户 Insight | 两 GEMM 搬完算完再下一拍；右矩阵未滞留；Vector 后 Cube 晚启动 | 直接驱动 T1/T2 |

产物路径（对照用）：

- 板端：`prof_msprof_op_dual/OPPROF_20260722205728_*/`
- 仿真：`prof_msprof_op_sim_t1024/OPPROF_20260722210919_*/simulator/core*.*/instr_exe.csv`

---

## 2. T1 Score Tile — 抄什么 / 不抄什么（有行号）

### 2.1 数学与数据面（本算子）

```text
MMAD1: qg  @ kneg^T → Aqk   (A=PLANE_QG, B=PLANE_KG, C=PLANE_AQK)
MMAD2: w   @ kneg^T → Akk   (A=PLANE_W,  B=PLANE_KG, C=PLANE_AKK)
```

**共享操作数是右矩阵 B=kneg（同一 GM `PLANE_KG`）**，不是共享左矩阵。实现目标：

```text
CopyGmToL1B(kneg) 一次并驻留
CopyGmToL1A(qg) → L1→L0 → TileMmad → Fixpipe Aqk
在 MMAD1/FIX 窗口重叠 CopyGmToL1A(w)
复用 L1B（必要时再 L1→L0B）→ TileMmad → Fixpipe Akk
禁止两次完整 BlockMmad；禁止两拍之间无差别 PIPE_ALL
```

### 2.2 必抄依据

| 参考文件 | 行号 / 段落 | 抄的机制 | 映射到 Score |
|----------|-------------|----------|--------------|
| [`prepare_wy_repr_bwd_full_cube.h`](fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_full/op_kernel/prepare_wy_repr_bwd_full_cube.h) | L426–430 注释 Stage C1；L476–519 数据流 | **Tile** `CopyGmToL1A/B` + `TileMmad`；先搬 A/B 再 MMAD；L507–512 在 Dkb 的 L1→L0 之后立刻搬下一 GEMM 的 A/B | Score：先搬 kneg+qg，MMAD1 期间搬 w；**不**引入「与 Kbeta 并行的多 ready 旗」 |
| 同文件 | L1 分区 + `EVENT_L1A/L1B` HardEvent | 专用事件，不用糊成 `PIPE_ALL` | Score 事件 id **避开** 已有 `MCH_EVT` / Dual `EVT_X/Y`（见本文件 MCH 段注释 L1155 附近） |
| [`chunk_gated_delta_rule_bwd_dhu_cube.h`](fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_kernel/chunk_gated_delta_rule_bwd_dhu_cube.h) | L460–488 | **Wait CrossCore 之前**先 `copyGmToL1B` / 预取下一 GEMM 的 A；Wait 后再搬依赖面 gatedQ | T2 主模板；T1 至少做到「手动 Tile、B 驻留」 |
| [PR #190](https://github.com/flashserve/flash-linear-attention-npu/pull/190) fused（未合入本地；可读 PR diff） | `prepare_wy_repr_bwd_cube.h` L1 resident 分区、`kResident` 同缓冲不同 layout | L1 长期驻留 + AIC **无 TPipe** | 只借「驻留分区 / Resource」思想；Score 只驻留 **一个 L1B=kneg** + 可复用 L1A |
| 本算子已有 MCH Tile | `MchL0Acc` / `MchL0AccDual`（同 cpp） | `PackedTileCopyTla` + HardEvent 隔离 | Score 复用同一套 Tile API 风格，**不**与 MCH 共用同一套 event id |

### 2.3 明确不抄

| 参考点 | 原因 |
|--------|------|
| `prepare_wy_repr_bwd_da` 的 `BlockMmad` | 正是当前反模式 |
| full/PR190 的多 stage、多中间 CrossCore ready（Dkb→Dkbg→…） | Score 只需 prep→mmad→done；多旗加重 scalar |
| PR190 GQA / 多 head K 缓存 | 本路径无 HV/HK 分组 |
| SolveTri cube `TPipe` + 多 L1 slot（MCH 路径） | Score 保持 `Arch::Resource` only |
| 共享**左**矩阵那套（Aᵀ 复用） | Score 共享的是 **B** |

### 2.4 改码落点

- 函数：`ComputeMmad`（L926+）；宏 `USE_SCORE_TILE_MMAD`（默认 1，可回退两次 BlockMmad）
- 不改 AIV Prep 语义；不改 scoreWs_/cmatWs_ 平面布局（仍 GM QG/W/KG → AQK/AKK）

### 2.5 T1 验收

- 精度：`test_npu_chunk_kda_fwd_intra_sub_chunk.py` 过
- 性能：空闲卡 msprof 中位 vs Dual 4.654；允许持平，禁止明显回退
- 结构：代码审查可见「单次 L1B(kneg) + 两次 TileMmad」；commit message 含 `T1 score-tile`

---

## 3. T2 CrossCore 前移 — 抄什么 / 不抄什么

### 3.1 现状空窗机制

```text
AIV: ... Prep → SetFlag(ready)
AIC: WaitFlag(ready) → Resource{} → BlockMmad×2(全部 MTE2) → SetFlag(done)
```

`Resource` 构造与全部 Score MTE2 都在 Wait **之后**串行（L1896–1898 + L941–942）。

### 3.2 必抄依据

| 参考 | 依据 | 本算子做法 |
|------|------|------------|
| bwd_dhu L460–479 | Wait 前搬 **不依赖** gatedQ 的 B（及下一拍 A） | AIC：在 Wait 前完成 `Resource`/L1 视图/`HardEvent` priming；对 **已写完 KG 的 slot** 可 `ScorePrefetchB`（sub0：Wait 后立刻搬 B，仍早于「整段 Block 黑盒」） |
| full L426–430 | 与 Vector 无依赖的 GEMM 可与 AIV 重叠 | 仅用于「无依赖操作数可提前」原则；**不**为 Score 增加第二套 ready 旗 |
| 本文件 AIV prologue L1842–1862 | Identity 写 + `CrossCoreBarrier` 在 Prep(0)+ready 前 | 审计：Identity 是否可与 Prep 重叠或仅 subBlock0 且尽早 ready；**禁止**为此重开 S2a 双 Prep |

### 3.3 不做

- S2a / S2b（仿真 + Dual 复测已否决）
- 在 Prep 未完成时偷读 scoreWs_ 平面（违反 DEPTH 写后读）

### 3.4 T2 验收

- 精度过 → commit
- 带 `-g` 仿真（可与 T3 合并采）：`READY` → 首次 Score `MOV_OUT_TO_L1*` 空隙缩小；`aic wait id4` 不恶化

---

## 4. T0 `-g` 与 T3 仿真 — 依据

| 项 | 依据 | 动作 |
|----|------|------|
| `-g` | `add_ops_compile_options` → [`cmake/func.cmake` L280+](cmake/func.cmake) 写入 CCE OPTIONS；仿真曾 `Code call stack is empty` | [`op_host/CMakeLists.txt`](fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/op_host/CMakeLists.txt) OPTIONS 追加 `-g` |
| 仿真入口 | 已有 [`prof_chunk_kda_fwd_intra_sub_chunk_sim_t1024.py`](torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_sim_t1024.py)；`TARGET_1P5` §5.2 | 输出 `prof_msprof_op_sim_t1024_g/`；对照 instr/旗/行号写回文档 |

T0 同步落盘 `SCORE_TILE_CROSSCORE_PLAN.md`（把本文件 §1–3 与门禁抄进去，作为 agent 执行母本）。

---

## 5. 原分析优化点还做（T4–T7）— 依据与顺序

**做。** 仿真新刀（T1–T3）不取消 `TARGET_1P5` 的 P0/P1；只是插到前面，避免与 Score 刀混测。

| 阶段 | 原依据 | 打谁 | 参考实现 |
|------|--------|------|----------|
| **T4** MCH 真 L1 驻留 | `TARGET_1P5` §0/§4 P0-A；板端 fixpipe+mte2≈2.17；sim FIXP≈30% | AIV wait id6 + AIC 搬运 | SolveTri `StoreL0C`→scratch→L1；`L0_ACC_MCH_DESIGN.md`；PR190 resident 思想 |
| **T5** S2c 批 MMAD→MCH | `TARGET_1P5` §4 P0-B；每 sub 双握手 + BAR 重 | CrossCore / 交替气泡 | 结构改动大；须在 T1–T4 出数后单变量开 |
| **T6** soft-prefetch / S4 | `TARGET_1P5` §4 P1；板端 id8 大于仿真 WriteSolve | barrier/Store / 残留 mte2 | **驻留后**再开 prefetch；S4 试单 AIV Post |
| **T7** 天花板 | `TARGET_1P5` §6–7 | 文档 | 不承诺单算子必达 1.5 |

保持关闭：S2a、S2b、扁平 ×NC、MBH、aclnnSolveTri、刷 mac（依据见 `TARGET_1P5` §3）。

---

## 6. 总顺序

```text
T0  -g + SCORE_TILE_CROSSCORE_PLAN.md 母本     → commit
T1  Score Tile + L1B(kneg) 驻留                 → 精度 → msprof → commit
T2  CrossCore / 可提前 MTE2                     → 精度 → commit
T3  带 -g 的 T=1024 simulator 留档               → 文档 → commit
T4  MCH 真 L1 驻留（原 P0-A）                   → 精度 → msprof → commit
T5  S2c（原 P0-B）                              → 精度 → msprof → commit
T6  prefetch / S4（原 P1）                      → 精度 → commit
T7  天花板 / 融合另立项（若需要）               → commit
```

```mermaid
flowchart LR
  T0["T0 -g+母本"] --> T1["T1 ScoreTile"]
  T1 --> T2["T2 CrossCore"]
  T2 --> T3["T3 sim-g"]
  T3 --> T4["T4 MCH_L1"]
  T4 --> T5["T5 S2c"]
  T5 --> T6["T6 prefetch/S4"]
  T6 --> T7["T7 ceiling"]
```

## 开关

- `USE_SCORE_TILE_MMAD`（T1，bring-up=1）
- 既有：`USE_MCH_L0_ACC=1`、`USE_MCH_L0_DUAL=1`、`USE_MCH_S2B_STEAL=0`
- T4/T5 新宏在实现时写入母本文档，不在本阶段空想命名以外的行为