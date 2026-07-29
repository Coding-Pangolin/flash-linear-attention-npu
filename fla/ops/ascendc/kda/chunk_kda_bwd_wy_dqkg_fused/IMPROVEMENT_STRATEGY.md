# ChunkKdaBwdWyDqkgFused — 改进策略（PR190 对照 + Sim/Board 依据）

> 面向实现 agent 的可执行策略。  
> 日期：2026-07-29  
> 契约不变：见 [`DESIGN.md`](DESIGN.md)；禁止回退 [`DESIGN.md` §5.2](DESIGN.md) 反模式。

---

## 0. 目标与现状

### 0.1 指标口径

| 口径 | Shape | 指标 |
|------|-------|------|
| 板端主指标 | `B1 H=HV=32 T8192 K128 V128 BT64 bf16` | msprof MIX_AIC **Task Duration median** |
| 仿真诊断 | `B1 H=HV=2 T1024 …` | `msprof op simulator` → Total tick + `*_instr_exe.csv` |
| 精度 | suite + model golden | Cube-faithful，exp2，**不得**放宽阈值 |

门禁：**suite 全绿** + 单变量宏 + **Δ ≤ −0.05 ms** 才 default on（见 [`ITER_LOG.md`](ITER_LOG.md)）。

### 0.2 当前板端基线（以 ITER_LOG 为准）

| 里程碑 | med ms | 关键宏 |
|--------|--------|--------|
| P1a | 5.84 | Gate MTE2 PP |
| F1 Merge-barrier-only | **5.48** | `USE_MERGE_BARRIER_ONLY=1` |
| **F3b BV128** | **~4.74** | `USE_BV128=1`, `nBv=1` |
| Stretch | **≤ 0.8** | 需结构刀（F6 切分或算法减工作） |

下文「当前」默认 **3.86 ms**（F3a'；策略初稿写 4.74 已过时）。G 档刀序见 [`NEXT_ITER_PLAN_G.md`](NEXT_ITER_PLAN_G.md)。

### 0.3 竞品参照（PR190 PrepareWyReprBwd）

| 项 | PR190 WY | 我们 fused | 倍数 |
|----|----------|------------|------|
| 板端 model | **~2.28 ms** | **~4.74 ms** | ~2.1× |
| Sim tick T1024 | **712k** | **1,392k** | ~1.96× |
| 语义 | **仅 WY 路**（吃 dw/du） | **dqkwg + WY** | — |

**拆账（粗估）**：

- 若 WY 理想 ≈2.3 ms，则 dqkwg + 融合税 ≈ **2.4 ms**。
- Sim 多出的 ~680k tick **不全是 GEMM**，instr 显示 **WAIT/BAR 涨得比 MMAD 快**（见 [`ref_pr190/results/COMPARE_PREPARE_WY_vs_WY_DQKG_SIM.md`](../../../../../../ref_pr190/results/COMPARE_PREPARE_WY_vs_WY_DQKG_SIM.md)）。

**结论**：下一阶段的「主矛盾」不是再加 Cube dbuf，而是 **把 CrossCore / AIV barrier 税对齐 PR190 的 stage 同步模型**；stretch 0.8 ms 则必须 **F6 切分或减单次 launch 工作**。

---

## 1. 根因诊断（证据链）

### 1.1 绑定侧：AIV-bound，但 Cube 在「等」

| 证据 | 来源 | 含义 |
|------|------|------|
| AIV/AIC cycles **4.6–5.0×** | Sim T1024 | Vec 决定 wall；优化 Vec 路径 ROI 最高 |
| Cube **MMAD < 1%** cycles | Sim wy_dqkg | Cube 算力空转，不是算不动 |
| Cube **WAIT_FLAG_DEV 27.6%** vs PR190 **13.6%** | Sim 对比 | 我们 Cube **等 Vec 约 2×** |
| Cube **BAR 22.7%** vs PR190 **9%** | Sim 对比 | 协议/Join 过密 |
| AIV **BAR 30.4%** vs PR190 **17.1%** | Sim 对比 | Vec 调度税 +13pp |
| 板端 `aiv_scalar_wait_id10` 量级 ~ms | 历史 msprof | 与 sim WAIT 叙事一致 |
| `USE_L0_AB_DBUF=0`, Fix∥MTE2 parked | ITER_LOG F5 | dbuf 不能单独救 wall |

### 1.2 结构侧：融合粒度过粗

| 证据 | 来源 | 含义 |
|------|------|------|
| Vec **62% kernel LOC**（~2010 行） | OPT_DIRECTION §1 | 改调度 = 改 vector.h Process |
| 单 launch 串完 S0→S1×nBk→Gate→S2→Mask→S3 | DESIGN §5 | 每次 launch 付 **全量** BAR/握手 |
| F1 已做 Merge-barrier-only **−0.43 ms** | ITER_LOG | Join 收紧有效，但 **仍 2× 于 PR190** → 还要继续减 sync |
| F3b BV128 **−0.74 ms** | ITER_LOG | 减 nBv 循环有效；**不改变** sync 次数级 |
| F4 MASK_ONCE **−0.016 ms** reject | ITER_LOG | 仅 mask 表不够，需配合 sync 重构 |
| WIN_SOFT_LEAD_V2 **+0.37 ms** reject | ITER_LOG | 禁止同构 soft-pipe |

### 1.3 PR190 做对了什么（应对齐的「模式」）

来源：[`ref_pr190/a2_a3_common_optimization_notes.md`](../../../../../../ref_pr190/a2_a3_common_optimization_notes.md)

1. **按依赖切 Stage**，不是按「一个 GEMM 一个 stage」机械切。
2. **CrossCore wait 放在 stage 入口**，row/tile 内循环外 **只 wait 一次**。
3. **2-head window 成组推进**：窗内 S0→S1→S2→S3，再下一窗；禁止单 head 跑完全流水。
4. **固定 sync0–5** 语义，producer 末尾 set、consumer 入口 wait。
5. **Cube L1 resident**（A/DW/DU/K）跨 stage 复用，减 ND2NZ 重复搬运。
6. **Vec/Cube 分工清晰**：Vec 产 Kbg/Vb/Kbeta；Cube 产 DA 链；最后 Vec 写 dk/dv/dbeta/dg。

我们 DESIGN §5.1 已有类似骨架，但 **KvAcc 内 Cube∥Kg「互不等」+ 每 BK 一套 Gate 握手**，在 sim 里放大为 WAIT/BAR（见 COMPARE §3 P1）。

---

## 2. 策略总览（优先级）

```text
P0 度量纪律（每刀必做）                         ← G0
P1 同步协议重构 — 对齐 PR190 stage/sync（D2+） ← G1 **主刀**（P1b 跳过：nBk=1）
P2 WY 子路径移植 — L1 resident v2              ← G4
P3 DaFinal / Mask 合并（D1 升级版）             ← G3（F4 禁止同构）
P4 Gate/Epilog 计算链合并                      ← G5
P5 BK128 / owned-arena（F3a'）                 ← **done** 3.86 ms
P6 F6 三 Op 切分                               ← MVP+F6b done；stretch → G6c
P7 Cube Fix∥MTE2（D5）                         ← G7 后置
```

**明确不做（已证伪）**：

- 再开 `USE_WIN_SOFT_LEAD_V2`、Epilog MTE2 PP 同构重试
- 无 sim/board 证据地开 `USE_L0_AB_DBUF` / `USE_FIX_MTE2_OVERLAP`
- 单 head 全流水、kg 挂 Wait(C_S1) 后（DESIGN §5.2）

---

## 3. P1 — 同步协议重构（最高优先级）

### 3.1 目标

- Sim：Cube `WAIT_FLAG_DEV` 从 **~28% → ≤18%**；AIV `BAR` 从 **~30% → ≤22%**（向 PR190 靠拢）。
- 板端：**−0.3 ~ −0.8 ms**（视实现深度）；至少 **−0.05 ms** 才合入。

### 3.2 理由

PR190 用 **6 个固定 sync** 完成 WY；我们用 **8 flag + 每 BK Gate 三明治**，且 Stage1 与 Kg **故意不互 wait**（DESIGN 要求）——这在提高 overlap 的同时，若 Set/Wait 粒度过细，**信用次数爆炸**。

F1（`USE_MERGE_BARRIER_ONLY=1`）只删了 **非 merge 点的 AIV Join**，**没改 CrossCore 次数**；sim 对比说明 **CrossCore 仍是 2× 税**。

### 3.3 方案 P1a — 握手「事件表」化（推荐先做）

**改什么**：在 `common.h` 增加 **WindowSyncPlan** 常量表，Cube/Vec Process **只读表驱动** Set/Wait，禁止散落 magic flag。

**对照 PR190 sync 映射（WY 部分）**：

| PR190 | 我们现有 | 建议 |
|-------|----------|------|
| sync0 Vec→Cube Kbg/Vb | （无直接对应；Kg 在 S1） | **Stage0 末** Vec 写 K 相关 WS 后 **一次** V_S0 |
| sync1 Cube→Vec DA1/DA2 | C_S0 + 部分 dAWs | **合并**：Stage0 Cube 两 GEMM 完 **一次** C_S0 |
| sync2 Vec→Cube DA4 | V_GATE | GateOnly 完 **一次** V_GATE（已是） |
| sync3 Cube→Vec DA6_T | C_S2 | Epilog 依赖 **一次** C_S2 |
| sync4 Vec→Cube D | V_MASK | Mask 完 **一次** V_MASK |
| sync5 Cube→Vec Dkb/DK | C_S3 | Stage3 完 **一次** C_S3 |

**KvAcc 特化**：

- **KgVec 与 RunStage1 并行**保持不变（DESIGN §5.2 #2）。
- **禁止**在 Kg 内 `Wait(C_S1)`；**禁止** Cube Stage1 内 `Wait(V_GATE)`。
- **合并 Set**：每个 head、每个 stage **最多 1 次 Set**（当前部分路径 head×BK 重复 Set 需审计）。

**落点**：

- `chunk_kda_bwd_wy_dqkg_fused_common.h` — flag 枚举 + SyncPlan
- `chunk_kda_bwd_wy_dqkg_fused_cube.h` — `Process()` 内 Wait/Set
- `chunk_kda_bwd_wy_dqkg_fused_vector.h` — 镜像

**宏**：`USE_SYNC_PLAN_V1=0` → 门禁后 1

**验收**：

1. Sim T1024：统计 `SetFlag`/`WaitFlag` **次数**（instr 或手工 log）应 **下降 ≥25%**。
2. 板端 model med **−0.05 ms**。
3. suite 全绿；PEM 无 hang。

### 3.4 方案 P1b — BK 批处理握手（条件触发）

**何时做**：P1a 后 Cube WAIT 仍 >20% 且 **MMAD 仍 <2%**。

**做什么**：`nBk=2`（K=128）时，Cube **连续** RunStage1(BK0)+RunStage1(BK1) 再 **一次** Set C_S1；Vec **双 BK Kg** 再统一 Gate。

**风险**：拉长 Cube 气泡；**必须 sim 证明 Wait 下降 > 计算增加**。

**宏**：`USE_BK_BATCH_HANDSHAKE=0`

### 3.5 方案 P1c — 删 redundant PipeBarrier

审计 `vector.h` 中 **每条 V 链末尾是否都有 BAR**；能合并的链：

- Gate：`exp2 → mul beta → mul k` **一条链末一次** `PipeBarrier<PIPE_V>`
- Epilog：`dq/dk/dg` 写 GM 前 **一次** BAR

**依据**：PR190 stage2 一次产出 `D`；我们 Epilog 多段 store 夹 BAR → sim ST/MOV 分散但 **BAR 30%**。

---

## 4. P2 — WY 子路径对齐 PR190（减 ND2NZ + 复用 L1）

### 4.1 目标

- 在 **不改变融合语义** 前提下，让 Stage0 + DaFinal 的 **Cube 搬运/Fix 占比** 从 sim **~26% → ~15%**。
- 板端 **−0.2 ~ −0.5 ms**（与 P1 叠加）。

### 4.2 理由

Sim：**MOV_OUT_TO_L1 ND2NZ 14.1%** vs PR190 **4.6%**。PR190 明确用 **L1 resident ping/pong** 存 A/DW/DU/K（[`a2_a3_common_optimization_notes.md` §2](../../../../../../ref_pr190/a2_a3_common_optimization_notes.md)）。

我们 `USE_L1_A_RESIDENT=1` 已有，但 **Stage0/Stage3 仍重复搬 A**；且 fused 路径还有 **dqkwg 的 h/dh** 占用 L1 budget。

### 4.3 方案 P2a — Stage0/3 Cube L1 生命周期延长

**做什么**（借鉴 PR190 `prepare_wy_repr_bwd_cube.h`）：

1. Stage0 入口：`A, DW(=dvb路径), DU(=du/dv)` **resident 搬入 L1**，本 head **整窗**不 evict。
2. Stage3：`A resident` 复用，避免第二次 GM→L1 ND2NZ。
3. **K resident**：若 UB/L1 允许，Stage3 `D@K` 复用 K tile（GVA 注意 hk/hv 映射）。

**落点**：`cube.h` Stage0/Stage3；`common.h` L1 layout 重算（**必须 UB 峰值表**）。

**宏**：`USE_WY_L1_RESIDENT_V2=0`

**验收**：sim Cube ND2NZ+FIX 占比降；板端 −0.05 ms；无 L1 越界。

### 4.4 方案 P2b — Stage0 Vec 与 PR190 stage0 对齐

PR190 Vec stage0 一次产出 **Kbg, Kbeta, Vb** 写 WS；我们 Stage0Vec + KgVec 分离。

**评估**：是否将 **beta*g 类预处理** 合并为 **单次 MTE2 扫描**（类似 PR190 `CopyInBetaGRows` + cast 融合），减少 Vec 往返 GM。

**落点**：`vector.h` `Stage0Vec` / `KgVec`

**依据**：Sim 我们 **MOV_UB_TO_OUT 9.8%**、PR190 **ST_XD_XN_IMM 24.1%** 风格不同——不是照抄，而是 **减少 GM 往返次数**。

---

## 5. P3 — DaFinal / Mask（D1 升级版）

### 5.1 目标

- Sim：**MOVEMASK 12% → ≤8%**；**BAR 再 −2pp**（与 P1 正交）。
- 板端：**−0.05 ~ −0.2 ms**。

### 5.2 理由

- F4 `USE_MASK_ONCE=0` 仅 **−0.016 ms**：说明 **mask 表 alone 不够**，因为 mask 路径仍夹 **Join + CrossCore**。
- PR190 stage2：**一次** 算 `D = -DA6_T * exp(min(g_j-g_i,0)) * upper_tri`，无 per-BK mask。

### 5.3 方案

1. **Init 因果位图**（BT=64 常量表），Stage3 **只 And/Select**，禁止每 head rebuild（OPT_DIRECTION D1 伪代码）。
2. **partial chunk**：只对 `validRows×validRows` 子块 mask（varlen tail）。
3. **Mask 与 beta 乘合并**为一条 V 链，**链末一次 BAR**。
4. **可选**：Cube Fixpipe 写 dA 时带上三角 mask（工作量大，独立宏 `USE_CUBE_CAUSAL_FIX=0`）。

**宏**：`USE_MASK_TABLE_APPLY=1`（在 P1 后重试 F4 思路）

**验收**：F4 失败原因复盘——必须在 **P1 减 sync 后**再测；仍 flat 则降优先级。

---

## 6. P4 — Gate/Epilog 计算链（dqkwg 特有）

### 6.1 目标

- 压 **VCADD 5.5%**、**MOV_OUT_TO_UB 4.4%** 中的冗余；板端 **−0.1 ~ −0.3 ms**。

### 6.2 理由

KvAcc/GateWy 做 **2D gate exp2**（KDA 特有，PR190 无）；`dg` 是 **per-K 向量**，Epilog 比 PR190 重。但 **dq/dk/dw** 累加链可折叠（已有 `USE_EPILOG_VEC_FOLD=1`，继续审计未 fold 分支）。

### 6.3 方案

1. **Kg 与 gate 复用 WS**：已有 `USE_GATE_REUSE_KG_WS=1`；检查是否仍有 **重复 exp2(g)**。
2. **dgk 累加**：`gn` 与 `dgk` merge 是否 **每 BK 重复读 GM** → 改 L1/UB 驻留。
3. **Epilog store**：`USE_DUAL_AIV_STORE=1` 下是否 **双 AIV 写同一行** 夹 BAR → 对齐半行所有权，**merge 点唯一 Join**（延续 F1）。

**宏**：增量宏 `USE_GATE_EPILOG_FOLD_V2=0`

---

## 7. P5 — BK128 / owned-arena（F3a'）

### 7.1 目标

- `nBk: 2 → 1` → **Gate 握手次数减半**；板端 **−0.2 ~ −0.5 ms**（若 UB 允许）。

### 7.2 理由

- F3b 已证 **BV128** 有效（−0.74 ms）。
- F3a BK128 **blocked**（UB）；F3a' **owned-compact arena** 在 [`results/F3A_ARENA_NOTE.md`](results/F3A_ARENA_NOTE.md)。

### 7.3 方案

1. 完成 **owned-compact** 索引（Gate/Epilog arena 不爆 192KB）。
2. `MAX_BK=128`，`nBk=1`；与 P1 **BK 批处理**二选一，避免重复逻辑。

**宏**：`USE_BK128=1`（替换今日 0）

**验收**：UB 峰值表 + suite + **−0.05 ms**。

---

## 8. P6 — F6 三 Op 切分（stretch ≤ 0.8 ms）

### 8.1 何时启动

- P1–P5 后仍 **> 2 ms** 且 stretch 0.8 ms 仍需要 **~2.5×** 提升。
- 或 P1 后 board **≤ 3.5 ms** 但 sim BAR 仍 **>25%**（单 launch 税到顶）。

### 8.2 理由

- 单 kernel 无论怎么微操，**每次 launch 付全量 AIV BAR**；PR190 WY 仅 712k tick，我们 1392k。
- chunk **无依赖** → Host 多 stream 可叠 OpA/B/C（[`SPLIT_KERNEL_PLAN.md`](SPLIT_KERNEL_PLAN.md)）。

### 8.3 方案（实现 F6）

| Op | 内容 | 预期 tick 占比 |
|----|------|----------------|
| **OpA** Stage0 + S1∥Kg | Cube 重 | ~40% |
| **OpB** Gate + S2 + Epilog | **AIV 最重** | ~35% |
| **OpC** Mask + S3 + Store | MOVEMASK 集中 | ~25% |

**关键**：

- Op 间 **无 CrossCore**；同 chunk 同 stream 顺序。
- **复用**今日 `SlotLayout*` GM WS；三 Op 同 task 分核。
- Host 默认仍 fused；`use_split` 开关。

**验收**：单 Op sim BAR ↓；多 stream chunk 重叠提高 NPU 占用；model 精度一致；总 wall **−30%+** 才可能逼近 0.8 ms。

---

## 9. P7 — Cube Fix∥MTE2（辅刀，后置）

### 9.1 条件（全部满足才开）

1. P1 后 Cube WAIT **仍高**但 **MMAD 占比上升**（说明 sync 已解，Fix 成瓶颈）。
2. Sim **FIX∩MTE2 可非零**；suite **无 507015**。
3. ITER_LOG F5 ECC 根因已修。

### 9.2 方案

- `USE_FIX_MTE2_OVERLAP=1` + DirectTileGemm 内 Fix 与下一 tile MTE2 重叠（OPT_DIRECTION D5 伪代码）。
- **不要**与 P1 同时上；**不要**在 WAIT≫MMAD 时开。

---

## 10. 推荐实施顺序（给改写 agent）

```text
迭代 1（1–2 天）
  ├─ P0：固定脚本 prof + sim T1024 + model T8192
  ├─ P1a：USE_SYNC_PLAN_V1 — 事件表 + 减 redundant Set/Wait
  └─ 验收：sim WAIT/BAR ↓ + board −0.05 ms

迭代 2
  ├─ P1c：V 链 BAR 合并（Gate/Epilog/Mask）
  ├─ P3：USE_MASK_TABLE_APPLY（P1 后重试）
  └─ 验收：sim MOVEMASK/BAR ↓

迭代 3
  ├─ P2a：WY L1 resident v2（Stage0/3）
  ├─ P5：F3a' BK128（若 UB 表过）
  └─ 目标：board ≤ 3.5 ms

迭代 4（stretch）
  ├─ P6：F6 OpA/B/C 原型 + 多 stream
  └─ 目标：board 向 0.8 ms 逼近

迭代 5（可选）
  └─ P7：Fix∥MTE2（ECC 解后）
```

每迭代 **单变量**；失败 **宏回 0**，写 [`ITER_LOG.md`](ITER_LOG.md)。

---

## 11. 验证清单（每刀复制）

```bash
# 环境
source /data/wnc/cann/ascend-toolkit/set_env.sh
conda activate fzy_atk
# wheel 安装略

# 精度
python fla/ops/ascendc/kda/chunk_kda_bwd_wy_dqkg_fused/test/test_chunk_kda_bwd_wy_dqkg_fused.py
python torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_wy_dqkg_fused.py

# Sim T1024
export FLA_SIM_T=1024 FLA_SIM_H=2
msprof op simulator --kernel-name=ChunkKdaBwdWyDqkgFused --soc-version=Ascend910B3 \
  --output=results/prof_sim_<tag> \
  torch_custom/fla_npu/test/prof_chunk_kda_bwd_wy_dqkg_fused_sim.py

# 板端 model
msprof --aic-metrics=PipeUtilization --output=results/prof_model_<tag> \
  torch_custom/fla_npu/test/prof_chunk_kda_bwd_wy_dqkg_fused_model.py
```

**解析**：`simulator/*_instr_exe.csv` 按 `instr`/`cycles` 聚合；对比 [`COMPARE_PREPARE_WY_vs_WY_DQKG_SIM.md`](../../../../../../ref_pr190/results/COMPARE_PREPARE_WY_vs_WY_DQKG_SIM.md) 中 PR190 基线。

---

## 12. PR190 代码阅读清单（移植时必读）

| 主题 | 路径（ref_pr190） |
|------|-------------------|
| Stage 依赖图 | `ref_pr190/a2_a3_common_optimization_notes.md` §1 |
| Cube 主循环 | `ref_pr190/fla/.../prepare_wy_repr_bwd/op_kernel/prepare_wy_repr_bwd_cube.h` |
| Vec 主循环 | `ref_pr190/fla/.../prepare_wy_repr_bwd/op_kernel/prepare_wy_repr_bwd_vector.h` |
| Slot/WS | `prepare_wy_repr_bwd_common.h` |
| L1 resident | `a2_a3_common_optimization_notes.md` §2 |

**注意**：PR190 是 **标量 g** + **无 dqkwg**；移植的是 **同步模式与 L1 生命周期**，不是逐行抄 gate 公式。KDA 的 **2D g、v_new、h/dh** 路径保留在 OpA/KvAcc。

---

## 13. 成功标准汇总

| 阶段 | 板端 med | Sim 特征 |
|------|----------|----------|
| 近期 | **≤ 3.5 ms** | Cube WAIT ≤18%，AIV BAR ≤22% |
| 务实 | **≤ 3.0 ms** | Total tick ↓30%+ vs 1392k |
| Stretch | **≤ 0.8 ms** | F6 多 stream + 单 Op 短 launch |

---

## 14. 一句话给改写 agent

**先 P1 把 CrossCore/AIV barrier 税降到 PR190 量级（sim 已证 2× 差距）；再 P2 借 PR190 L1 resident 压 ND2NZ；P3–P5 减 nBk/ mask；仍不够就 F6 切分。禁止在 WAIT≫MMAD 时开 Cube dbuf/Fix overlap。**
