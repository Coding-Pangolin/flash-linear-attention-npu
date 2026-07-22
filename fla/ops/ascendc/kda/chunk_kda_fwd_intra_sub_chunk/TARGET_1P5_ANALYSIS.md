# Intra-sub-chunk：1.5 ms 目标深度分析

> 当前最佳（空闲卡 device1，msprof PipeUtilization 中位）：**Phase B Dual ≈ 4.654 ms**  
> 新目标：**1.5 ms**（缺口 ≈ **3.15 ms / −68%**）  
> Shape 不变：`(B=1,H=32,T=8192,K=128,BT=64)` bf16 · `task=B×HV×NT` · NC=4 · MIX_AIC_1_2  
> 工具：[msOpProf 快速入门](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/optool/docs/zh/quick_start/msopprof_quick_start.md)

---

## 0. 结论先说

| 判断 | 内容 |
|------|------|
| **1.5 ms 是否「再抠几刀」可达** | **极难单靠现有 CV 骨架**。按 ~820 sub/核，1.5 ms ⇒ **~1.83 µs/sub**；现在 ~5.7 µs/sub。需同时砍 AIV 与 AIC，且砍掉大部分 **GM Fixpipe/Nd2Nz + CV 交替**。 |
| **当前真正瓶颈（相对 5 ms 时代已变）** | 墙钟两边几乎打满：`aiv_time≈aicore_time≈4.52 ms`。AIC 不再是「干等 AIV I+Y」，而是 **`fixpipe(1.13)+mte2(1.04)+scalar(1.02)`**；AIV 仍是 **`scalar(1.61)+vec(0.60)+mte2/3`**。 |
| **仿真交叉验证（T=64 smoke + T=1024）** | Cube **真算力极轻**：`MMAD` 指令 ≈**0.8%** cycle；**FIXP≈30% + MTE2(Nd2Nz)≈14% + BAR/等旗** 才是 AIC 主体。AIV **`WAIT_FLAG_DEVI≈20%`**，主要在等 `DONE/SOLVE_DONE`。Prep‖MCH **已重叠到位**（下一拍 `READY` 早于本拍 `SOLVE_DONE` ≈2.5k tick）。 |
| **最值得做（P0）** | **① MCH 真 L1 驻留**（直接打 FIXP+Nd2Nz+AIV 等 MCH）；**② S2c 批 MMAD→批 MCH**（打 BAR/CrossCore 次数与 Fixpipe 批处理）。二者都比再抠 Prep 优先。 |
| **降权 / 仍否决** | **S2a 双 Prep**：仿真已显示 Prep 不是临界路径。**S2b steal**：已复测无收益。soft-prefetch 仅在驻留后 `aic_mte2` 仍高时再开。 |
| **可能必须另立项** | 与上下游融合、改分核/扩核、减 NT×NC 有效工作量——纯本算子内优化到 1.5 不承诺。 |

---

## 1. 当前画像（Dual `op_summary` 中位）

| 指标 | ms | 解读 |
|------|-----|------|
| Task Duration | **4.654** | 验收主指标 |
| aiv_time / aicore_time | ≈4.521 / 4.521 | **CV 两侧同长** → 任一侧单独加速若另一侧不降，墙钟不动 |
| aiv_scalar | 1.613 | 控制流 / CrossCore / HardEvent 空等 |
| aiv_vec | 0.596 | Prep/WriteSolve/Store 向量体 |
| aiv_mte2 / mte3 | 0.644 / 0.481 | GM↔UB |
| aic_mac | 0.248 | Cube 算力已很轻（Dual 后还降了） |
| **aic_fixpipe** | **1.127** | **MCH 中间结果狂出 L0C→GM** |
| **aic_mte2** | **1.043** | **GM→L1 Nd2Nz 回灌 + Score 加载** |
| aic_mte1 | 0.308 | L1→L0 |
| aic_scalar | 1.018 | 等旗 / 编排（已比 S5b 的 2.77 好很多） |

**Pipe 加总（AIV≈3.33、AIC 关键路径 fixpipe+mte2+mac+mte1≈2.7）均小于墙钟**：仍有 **气泡 / 互等**，但主矛盾已从「等 I+Y」转为 **搬运税 + 仍偏厚的标量同步**。

每 sub（≈820 sub/核）粗算：

```text
现在：4.65 ms / 820 ≈ 5.7 µs/sub
目标：1.5 ms / 820 ≈ 1.8 µs/sub
```

---

## 2. 今日流水（热路径）

```text
AIV prologue: 写常驻 I → Prep(0) → ready
for iSub in 0..3:
  AIV: WaitDone → WriteSolve → barrier → solveReady
       → [Prep(i+1)+ready]          # 与 MCH 名义重叠
       → WaitSolveDone → Store(SOLVE_X)
  AIC: WaitReady → MMAD×2 → Done
       → WaitSolveReady → MchL0AccDual → solveDone
```

相对 S5b 已砍：第二次 CV（I+Y）、闭式 6×GM GEMM、Phase A 串行 ACC。  
**仍在交税：**

1. 每 sub **仍 1×** `(ready/done)+(solveReady/solveDone)` + WriteSolve 后 **AIV CrossCoreBarrier**  
2. Dual 每轮 **Fixpipe X/Y → `solveWs_` → CopyGmToL1**（SolveTri 是 scratch→L1；我们对外可见 GM 平面更重）  
3. Score `BlockMmad` 与 MCH **各构造 `Resource` + `PIPE_ALL`**，无法同 L0 流水  
4. Prep(i+1)‖MCH(i) 有了，但 AIV Store 仍在 MCH 之后串在 AIV 临界路径上  

---

## 3. 历史刀：现在还值不值得重开

| 项 | 旧结论 | 在 Dual@4.65 下的再评估 | 建议 |
|----|--------|-------------------------|------|
| S1 HardEvent / S3 Prep | 已合入 | 仍可再砍残留 Wait 对 | **P1 扫尾** |
| **S2a 双 Prep** | +0.18 ms（挡 solveReady0） | 仿真：Prep 已提前 ~2.5k tick 结束，非临界路径 | **否决（仿真坐实）** |
| **S2b steal** | Dual 后 4.681，无优 | 已复测 | **保持关** |
| soft-prefetch（等旗时装下拍 L1） | 未认真做 | 当前 MTE2 主因是 MCH Nd2Nz，不是 Score | **P0 驻留后再开** |
| **S2c 批 MMAD→批 MCH** | 侵入大，5 ms 时跳过 | 仿真 BAR/每 sub 双握手仍重 | **P0** |
| S4 非对称 AIV | 可选 | 板端 id8 大于仿真 WriteSolve → barrier/Store 嫌疑 | **P1 重开** |
| 去半行 barrier | S4 一部分 | 若改单 AIV 做满 BC=16 | 随 S4 |
| DEPTH=3 | 未做 | 仅当 profile 显示大量 WaitReady | **先量再开** |
| 扁平 ×NC | 否决 | 不变 | **不做** |
| aclnnSolveTri / MBH | 否决 | 不变 | **不做** |
| Phase B Dual | 已合入 4.65 | 保留 | 默认开 |
| **真 L1 驻留 MCH** | Dual 只做了「回灌」，仍每步 GM | `fixpipe+mte2≈2.17 ms` | **P0** |
| 上下游融合 | plan 标「1.5 另立项」 | 单算子天花板可能 >1.5 | **并行跟踪** |

---

## 4. 值得做的优化包（按优先级，已按仿真重排）

### P0-A — MCH「真驻留」：砍 Fixpipe/Nd2Nz（第一刀）

**现状：** 每 iter `StoreX/Y → solveWs_ → Nd2Nz→L1`。  
**目标：** 中间轮只 L0C→scratch→L1（或能直接 L1），**仅末轮 X Fixpipe 到 `SOLVE_X` 供 Store**；Y 中间不进 GM 平面。

**仿真钉死：** Cube `FIXP≈30%` + `MOV_OUT_TO_L1_MULTI_ND2NZ≈14%`，`MMAD` 指令 **<1%**。  
预期：`aic_fixpipe` / `aic_mte2` 与 AIV `wait id6` 同降；乐观 **−0.5~−1.5 ms**。  
风险：事件/别名再现 `akkd_rel≈2300` → 单笔对照门禁。

### P0-B — S2c：同 chunk「先 MMAD×NC，再 MCH×NC」

```text
# 今： (MMAD‖握手 → MCH‖握手)×4
# 目标： MMAD×4（DEPTH 或更大 WS）→ 一次/少次 barrier → MCH×4
```

减 CrossCore / BAR；Fixpipe 可批。仿真里 BAR 在 Cube/AIV 都极重。  
预期：**−0.5~−1.5 ms**。  
侵入：workspace、DEPTH、WriteSolve 批完再批 MCH。  
注意：今日 Prep‖MCH 重叠在 S2c 后要 **重做 overlap 设计**（否则可能吃回一部分收益）。

### P1-A — S4 / 减 Dual-AIV barrier + Store 重叠

板端 `aic wait solveReady` 大于仿真 WriteSolve 体 → 怀疑 **半行 barrier + Store 串行**。  
试：单 AIV 满行 Post，或 Store(i)∥MMAD(i+1)/Prep。  
预期：不确定 **±0.3~0.7**。

### P1-B — soft-prefetch（驻留之后）

仅当 P0-A 后 `aic_mte2` 仍高（Score 侧 Nd2Nz）再开；现在 MTE2 主因是 MCH 回灌。  
预期：小（**−0.1~−0.3**）。

### P2 — 工具驱动再发现

见 §5.5。不开「刷 mac」。

### 明确不做 / 低优先

- **S2a 双 Prep**（仿真：Prep 已提前结束）  
- **S2b steal**（已复测）  
- 扁平 ×NC、MBH、aclnnSolveTri、score 改 fp32  
- 为 1.5 承诺「只改注释级」的微优化  

---

## 5. 工具：`msprof op` / `msprof op simulator`（本机标准用法）

官方概念入门：[msOpProf 快速入门](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/optool/docs/zh/quick_start/msopprof_quick_start.md)  
本机可执行：`msprof`（子命令 `op` / `op simulator` 即 msopprof）。

Kernel 名（`--kernel-name`）：用 **GE Op 名** `ChunkKdaFwdIntraSubChunk`（前缀匹配实际 launch  
`ChunkKdaFwdIntraSubChunk_<hash>_1_mix_aic`）。  
不要用 C 符号 `chunk_kda_fwd_intra_sub_chunk`——会全部 skip、无 dump。

### 5.1 上板采集（真实墙钟 / pipe / 冲突）

```bash
source /data/wnc/cann/ascend-toolkit/set_env.sh
cd /workspace/fzy/code/kda/flash-linear-attention-npu
conda activate fzy_atk

ASCEND_DEVICE_ID=1 msprof op \
  --kernel-name=ChunkKdaFwdIntraSubChunk \
  --output=prof_msprof_op_dual \
  --aic-metrics=PipeUtilization \
  --launch-count=8 --warm-up=3 \
  python torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_model.py
```

产物目录示例：`prof_msprof_op_dual/OPPROF_*/ChunkKdaFwdIntraSubChunk_*_mix_aic/<i>/`  
内含 `PipeUtilization_*.csv`、`visualize_data.bin`（**Insight Import**）。

也可用精度脚本：

```bash
ASCEND_DEVICE_ID=7 msprof op --kernel-name=ChunkKdaFwdIntraSubChunk \
  --output=prof_msprof_op_pytest --aic-metrics=PipeUtilization \
  python torch_custom/fla_npu/test/test_npu_chunk_kda_fwd_intra_sub_chunk.py
```

### 5.2 仿真流水（`msprof op simulator`）

```bash
source /data/wnc/cann/ascend-toolkit/set_env.sh
conda activate fzy_atk
export LD_LIBRARY_PATH=/data/wyf/cann/cann-9.1.0/cann-9.1.0-beta.1/aarch64-linux/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
# 等价：/data/wnc/cann/cann-9.1.0-beta.1/aarch64-linux/simulator/Ascend910B3/lib

cd /workspace/fzy/code/kda/flash-linear-attention-npu

# 小 smoke：B=1,H=2,T=64（~24s wall / Total tick≈119k）
ASCEND_DEVICE_ID=1 msprof op simulator \
  --kernel-name=ChunkKdaFwdIntraSubChunk \
  --soc-version=Ascend910B3 \
  --aic-metrics=PipeUtilization \
  --output=prof_msprof_op_sim \
  --launch-count=1 \
  python torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_sim_smoke.py

# 中等 shape：B=1,H=2,T=1024（~73s model / Total tick≈301k；勿上 H=32×8192）
ASCEND_DEVICE_ID=1 msprof op simulator \
  --kernel-name=ChunkKdaFwdIntraSubChunk \
  --soc-version=Ascend910B3 \
  --aic-metrics=PipeUtilization \
  --output=prof_msprof_op_sim_t1024 \
  --launch-count=1 \
  python torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_sim_t1024.py
```

产物：

| 跑次 | 目录 | 备注 |
|------|------|------|
| smoke T=64 | `prof_msprof_op_sim/OPPROF_*` + `prof_msprof_op_sim_run.log` | 早期 OPPROF 可能空壳；**以 PEM 日志旗时序为准** |
| **T=1024** | `prof_msprof_op_sim_t1024/OPPROF_20260722210919_NXUDGDMLSOGDTZVZ/` | 有 `simulator/core*.{cube,vec}*/{instr_exe.csv,trace.json}` |
| T=1024 日志 | `prof_msprof_op_sim_t1024_run.log` | Total tick / 核 duration 表 |

仿真 **不一定** 产出板端那种 `visualize_data.bin`；优先读 `*_instr_exe.csv`（按 `pipe`/`instr` 聚合）+ PEM `set_flag` 时序。

### 5.3 「小算子 / 分段」耗时

本 op 是单 kernel。优先看 **§5.4 的 flag wait id**（比粗 pipe 更直接）；再加深用 `msprof op --mstx=on`。 

### 5.4 已采：`msprof op` PipeUtilization（Dual 热路径）

目录：`prof_msprof_op_dual/OPPROF_20260722205728_HVORDBOAMNWOUDVZ/`  
Insight：各 launch 下的 `visualize_data.bin`（共 8 份）。

**每 MixBlock 中位（与全卡 msprof 一致量级）：**

| 指标 | ≈ms | 备注 |
|------|-----|------|
| aic/aiv_time | 4.53 | 两侧打满 |
| aic_fixpipe | 1.12 | ratio≈0.24 |
| aic_mte2 | 1.05 | ratio≈0.22 |
| aic_cube | 0.20 | 很轻 |
| aiv_scalar | 1.63 | |
| **aiv wait id6 (solveDone)** | **≈1.90** | AIV 等 MCH 完成 |
| **aiv wait id2 (done)** | **≈1.10** | AIV 等 MMAD |
| **aic wait id8 (solveReady)** | **≈1.21** | AIC 等 WriteSolve |
| aic wait id4 (ready) | ≈0.58 | AIC 等 Prep |

→ 墙钟里仍有 **大量 CV 互等**；P0-B 批 MMAD/MCH、P0-A 缩短 MCH（少 Fixpipe）都能直接打这些 wait。

### 5.5 仿真定位（smoke + T=1024）— 更新优化策略的依据

旗语义（PEM `FFTS→AIC set_flag` = **对端已置位、本端收到通知**）：

| flag_id | 名 | 接收端 |
|---------|----|--------|
| 4 | `READY` | Cube（AIV Prep 完成） |
| 2 | `DONE` | AIV（Cube MMAD 完成） |
| 8 | `SOLVE_READY` | Cube（AIV WriteSolve 完成） |
| 6 | `SOLVE_DONE` | AIV（Cube MCH 完成） |

**A. 稳态子 chunk 时序（T=1024，MixBlock core0，中位 tick）**

| 段 | tick（中位） | 相对 READY 周期≈9.1k |
|----|-------------|----------------------|
| MMAD：`READY→DONE` | ≈5.1k | ~53% |
| WriteSolve：`DONE→SOLVE_READY` | ≈1.6k | ~17% |
| MCH：`SOLVE_READY→SOLVE_DONE` | ≈5.0k | ~52% |
| Prep 重叠：`READY_{i+1} − SOLVE_DONE_i` | **≈ −2.5k** | Prep **早于** MCH 结束 |

含义：

1. **MMAD 与 MCH 几乎一样长**，且两者之和 > 周期 → 靠 Prep‖MCH、以及 Store 与下一拍 MMAD 的交错才能跑到 ~9k/sub。  
2. **再加速 Prep（S2a）几乎不动墙钟**——Prep 已不是临界路径。  
3. **缩短 MCH（少 Fixpipe/少回灌）** 会同时：缩短 AIC 段、减少 AIV 在 `SOLVE_DONE` 上的等待（板端 id6≈1.9 ms）。  
4. WriteSolve 相对短；板端 `aic wait id8≈1.2 ms` 更大，部分来自 **Store / Dual-AIV barrier / 上一拍收尾**，不单是向量体。

**B. T=1024 `instr_exe` 管道占比（`core0.cubecore0` / `veccore0`）**

| 侧 | 证据 | 解读 |
|----|------|------|
| Cube FIXP | **~30%** cycle；`FIX_L0C_TO_DST` 靠前 | MCH 中间结果狂出 L0C |
| Cube MTE2 | **~14.5%**；头名 `MOV_OUT_TO_L1_MULTI_ND2NZ` | `solveWs_` GM→L1 回灌 |
| Cube CUBE / `MMAD` | pipe ~9.5%；**指令 MMAD ~0.8%** | **不是算力墙** |
| Cube BAR + SCALAR + WAIT | BAR 单独 ~36% cycle 量级 | 管道互斥 + CrossCore 厚 |
| AIV VECTOR | ~46% | Prep/WriteSolve/Store 有实活 |
| AIV `WAIT_FLAG_DEVI` | **~20%** | 与板端 id2/id6 互等一致 |

smoke（T=64）旗时序同构：MMAD≈MCH≈4.7k、WriteSolve≈1.17k、Prep 负重叠 ≈−2.6k → **结构结论不依赖 1024**；1024 只是把 instr 占比钉死。

**C. 据此重排的优化策略**

| 优先级 | 动作 | 打谁 | 仿真/板端证据 |
|--------|------|------|----------------|
| **P0-A** | MCH **真 L1 驻留**（中间 X/Y 不经 GM；仅末轮 X Fixpipe 供 Store） | FIXP + Nd2Nz + AIV wait id6 | Cube FIXP+MTE2≈45%；MCH≈MMAD 长 |
| **P0-B** | **S2c** 同 chunk：先 MMAD×NC，再 MCH×NC（少握手、Fixpipe 可批） | BAR / CrossCore / 交替气泡 | 每 sub 仍 2 次 CV；BAR 占比高 |
| **P1** | Store∥下一拍 / 减 Dual-AIV barrier（**S4** 变体） | 板端 id8 偏大、AIV barrier | WriteSolve 仿真短但板 wait 长 |
| **P1** | soft-prefetch 下拍 Score L1 | 残留 `aic_mte2` | **驻留后**再开；现在 Nd2Nz 主因是 MCH 回灌 |
| **P2 / 不做** | S2a 双 Prep、S2b steal、刷 mac | — | Prep 已重叠；MMAD 指令 <1% |

**执行顺序不变：** T1 驻留 → T2 S2c（或证据更强者先）→ T3 S4/Store 重叠 → T4 天花板/融合。

### 5.6 旁证（裸 msprof 分能力，较早）

| 能力 | 读数 | 含义 |
|------|------|------|
| ResourceConflictRatio | bank cflt ≈0.3%–0.7% | UB 冲突非主因 |
| MemoryL0 | L0 带宽远未打满 | 非 L0 bound |

---

## 6. 通向 1.5 ms 的情景（已用实测更新）

| 情景 | 实测 / 假设 | 大约落到 |
|------|-------------|----------|
| Dual → T4 L1 resident | **实测** −0.54 | **~4.1** |
| + S2c 批 MMAD/MCH | **实测回归** +1.25 | 5.36（已关） |
| + S4/prefetch | 未做；上限 −0.3~0.7 | ~3.4–3.8（乐观） |
| 仍 >1.5 | 需融合 / 改调度 / 减工作量 / 950 L0C→L1 | **另立项** → `CEILING_1P5.md` |

**诚实上限：** 本 sprint 证实 **~4.1 ms** 为现骨架下最佳；原「2.0~2.5」乐观带在 S2c 证伪后更不稳。**1.5 ms 单算子内不可靠承诺。**

---

## 7. 执行顺序（结案）

```text
T0–T4  完成；最佳 Dur ≈ 4.1 ms（USE_MCH_L1_RESIDENT）
T5     试过 S2c，默认关（dfe9b34）
T6     延期（见 CEILING_1P5 §3）
T7     天花板文档 CEILING_1P5.md
```

单变量、空闲卡、精度不回退；开关可回退到 Dual。

---

## 8. 相关路径

| 文件 | 角色 |
|------|------|
| `CUBE_PIPELINE_DESIGN.md` | 墙钟年表 |
| `PHASE_B_DUAL_PLAN.md` | Dual / S2b 结案 |
| `L0_ACC_MCH_DESIGN.md` | ACC 母本 |
| `prof_intra_sub_chunk_dual/` | 4.65 ms 对照 |
| `prof_msprof_op_dual/` | 板端 msprof op + visualize |
| `prof_msprof_op_sim_t1024/` | **T=1024 仿真** `instr_exe` / `trace.json` |
| `torch_custom/.../prof_chunk_kda_fwd_intra_sub_chunk_model.py` | 板端入口 |
| `torch_custom/.../prof_chunk_kda_fwd_intra_sub_chunk_sim_t1024.py` | 仿真 T=1024 入口 |
