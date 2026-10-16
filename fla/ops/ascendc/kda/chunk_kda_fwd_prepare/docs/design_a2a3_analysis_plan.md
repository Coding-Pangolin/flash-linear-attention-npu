# ChunkKdaFwdPrepare（A2/A3）阶段划分：详细分析计划

> 目的：用**可复核的证据**回答"这个算子在 A2/A3 上的 stage 划分与调度该怎么定"，
> 并为后续实现/调优给出取舍依据。
> 参照：《线性 Attention 算子设计培训：Stage 划分与 4-Head 调度》19 条约束 + 9 步设计流程。
> 现状基线：`fla/ops/ascendc/kda/chunk_kda_fwd_prepare`（arch22 分支）。
> 前置结论（[`design_a2a3_stage_redesign.md`](design_a2a3_stage_redesign.md)）：8 阶段划分合规；
> 值得查的是 **V6 提前**、**回合 3→2**、**GVA norm 去重**、**4-head window 消融**。

## 0. 要回答的问题（每条给"证据"和"决策阈值"）

| # | 问题 | 需要的证据 | 决策阈值 / 判据 |
|---|---|---|---|
| **Q1** | 8 阶段划分是否已经满足 19 条约束？ | 逐条约束 → 代码位置（stage 函数、policy 偏移、flag 表） | 19 条全部有落点；违规项必须能归因到"平台约束"或"已知妥协" |
| **Q2** | 当前瓶颈在 AIV、AIC 还是握手？ | 整核 pipe busy（msprof `--aic-metrics`）+ 每回合时长 + flag 往返代价 | 若 AIV busy/AIC busy 都 < 60% 且等长大，瓶颈是握手 ⇒ 优先减回合 |
| **Q3** | `V6` 提前能拿多少？ | V6 与 C2 的实测时长（分 stage 计时） | 收益 ≈ `min(V6, C2)`；若 ≥ 总时长 5% ⇒ 值得做 |
| **Q4** | 回合 3→2 能拿多少？ | 单次 ready/free 往返代价（flag 延迟）+ 第二回合实际排队 | 若往返 ≥ 3% ⇒ 值得做；同时确认集合同步计数仍严格配对 |
| **Q5** | GVA 下 Q/K norm 去重值不值？ | V0 中 norm 部分占比 × `(1 − HK/HV)`；去重后的额外同步代价 | 净收益 > 0 且不引入跨 AIV 共享 ⇒ 值得做 |
| **Q6** | 4-head window 是否最优？ | 1/2/4/8 head 消融（总时长 + 尾波 + L1/UB 占用） | 取总时长最小且预算不超者；若差异 < 噪声则维持 4（并记录"未证明最优"） |

**最终决策产物**：A2/A3 的 stage 表 + 调度回合表 + 数据路径表 + 预算表 + 风险清单（见 §7）。

## 1. 分析原则

1. **证据分级**：静态可算（预算/依赖图）< 上板实测（msprof）< 消融实验。能用静态算清楚的不做实验。
2. **只测现状，不改数学**：所有实验保持数值路径不变；实验版只允许"屏蔽某个 stage / 改调度顺序"，禁止改公式。
3. **同机同频同口径**：A2 真机固定一张卡；每次改动对比都用同一采法（msprof op，3 warmup + 10 次取中位），记录 `Current Freq`。
4. **一次一个变量**：分 stage 计时、调度重排、norm 去重、head window 各自独立批次。
5. **A3 只能构建级**：910_93 只做编译与静态审计，性能/精度结论以 A2(910B3) 真机为准，文档里写明。

## 2. 静态分析（不需要真机，先做完）

### S1 结构审计（Q1）

- 输入：`op_kernel/arch22/{chunk_kda_fwd_prepare_cube,vec}.h`、`chunk_kda_fwd_prepare_policy.h`、
  `docs/design.md §3/§6/§7`。
- 动作：把 19 条约束做成表格，逐条填"代码落点 + 证据 + 判定（合规 / 平台被迫 / 待改）"。
- 已有结论（需补代码行号）：合规 11 条、平台被迫 1 条（C2→V3 经 GM relay）、待改 2 条（V6 位置、GVA norm 重复）。
- 产物：`docs/design_a2a3_constraint_audit.md`（表）。

### S2 预算账（UB / L1 / workspace 峰值与余量）

- 输入：`policy.h` 的 `Arch22Ub / L1 / Workspace / ScorePayload` 常量。
- 已知常量：A2/A3 `kUsableBytes=0x2E000`(184 KiB) = 2×`kPrivateBytes`(72 KiB) + `kSharedBase` 40 KiB；
  L1 `kPeak=0x5C000`(368 KiB)/512 KiB，其中 4×`kHeadLaneBytes`(72 KiB)；
  workspace `kArch22SlotStride = 0x1A400 + 0x5000`，`kArch22SlotCount=4`。
- 动作：对每个 stage 列出"活跃工作集（UB/L1/workspace）"，检查：① 每 stage 单次搬运完（约束 12）
  ② 双份缓冲是否足够（约束 4/5）③ 4 份常驻是否成立（约束 7/9）④ 余量能否支撑 V6 提前。
- 产物：`docs/design_a2a3_budget.md`（每 stage 一行，含峰值与余量）。

### S3 依赖图与关键路径模型（Q3/Q4 的解析基础）

- 动作：按设计流程第 2 步画依赖图（节点=stage，边=数据依赖 + 同步边），标注每条边当前的落点
  （UB / L1 / GM-relay）与同步方式（跨核 flag / 核内 HardEvent）。
- 由依赖图导出两个候选调度：
  - **现状**：`V0+V1 → C2 → V3 → C4 → C5 → V6 → C7`（3 个跨核回合）
  - **候选 B**：`V0+V1+V6 → C2 → V3 → C4 → C5 → C7`（2 个跨核回合，V6 与 C2 重叠）
- 关键路径公式（用于 §4 的反事实推算）：
  `T_now = V0V1 + C2 + V3 + C4C5 + V6 + C7 + 3×δ_flag`
  `T_B   = V0V1 + max(V6, C2) + V3 + C4C5 + C7 + 2×δ_flag`
- 产物：`docs/design_a2a3_depgraph.md`（图 + 公式 + 每个符号的待测项）。

## 3. 动态分析（A2 真机）

### D0 基线口径（每批实验前必做）

```bash
# 机器/卡/频率/commit 记录；同机同频；3 warmup + 10 次采样取中位
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info                                  # 记录型号/频率/温度
bash <op>/scripts/gates/run_msopprof.sh 1024:8 4096:8 11264:32
```

判据：同一配置重复采 3 次的中位差 < 2%；否则先解决环境噪声（换卡/降载）。

### D1 整核画像（Q2）

- 采 `msprof op --aic-metrics=Default`，读 `OpBasicInfo.csv` 的 `Task Duration` 与 pipe busy：
  `cube / fixpipe / mte1 / mte2 / vec / scalar`。
- 目的：判断 AIV 与 AIC 各占多少、谁在等谁。**关键指标**：AIV 侧 `vec+scalar+mte2+mte3` 之和 vs AIC 侧
  `cube+fixpipe+mte1+mte2` 之和，以及各自的 idle 比例。
- 产物：`perf/round_000_baseline/`（含 csv 与解读）。

### D2 分 stage 计时（Q3 的核心证据）

A2/A3 **没有 VF/RegBase**（那是 A5 的写法）、也未必有 simulator，采用**屏蔽差分法**：

1. 基线 `T_all`；
2. 对每个 stage k 做一版"**body 置空但同步不变**"的实验核（stage 函数立即 return，仍走原来的 flag 回合），
   得到 `T_k`；
3. 该 stage 的关键路径贡献 ≈ `T_k − T_all`（差分为负即为净贡献；对不在关键路径上的 stage 差分会接近 0，
   这本身就是"该 stage 已被隐藏"的证据）。
- 必做：`V0V1 / C2 / V3 / C4 / C5 / V6 / C7` 共 7 版；每版同口径采 3×10。
- 交叉验证：用 MindStudio Insight 看 timeline（`visualize_data.bin`），在 `CrossCoreSetFlag/WaitFlag`
  的 marker 之间读各阶段耗时，与差分法对得上（差异 > 20% 要查明）。
- 产物：每 stage 一张"贡献表"（时长、占关键路径比、是否已在等待窗口内被隐藏）。

### D3 握手代价（Q4）

- 方法一（实测）：把两次相邻 ready/free 回合**人为合并**（例如把 C4 的 wait 提前到 C2 的 free 之后同一回合），
  比较总时长差 ⇒ 得到单次往返 `δ_flag` 的量级。
- 方法二（解析）：统计 `CrossCoreSetFlag/CrossCoreWaitFlag` 次数 × 典型 flag 延迟（可从 Insight 读两次事件间隔）。
- 判据：`δ_flag × 省下回合数 ≥ 总时长 3%` 才值得动调度。

## 4. 反事实与消融（Q3–Q6）

### E1 V6 提前（候选 B）

- 变更：AIV 侧把 `StageV6` 移到 `StageV1` 之后、`StageV3` 之前；AIC 侧把 C4/C5/C7 放在 V3 之后的同一回合。
- 需要同时核对的 4 件事（任一不成立则放弃）：
  1. V6 输出（`K_beta_g/V_beta`）对 AIC 的可见性：确认现有 `DataSyncBarrier`/FIX 事件在 C7 之前生效；
  2. UB 私有区 V6 与 V3 **顺序执行**（同 AIV，不并行）；
  3. 共享区（`kSharedG/kSharedScratch`）在 V6→V3 之间换义正确（V3 的 `kV3BetaEff=kSharedBase` 在 V6 之后）；
  4. 集合同步计数：ready/free 从 3 对降到 2 对，两侧 set/wait 严格配对（不配会 hang）。
- 判据：L1 位级与基线**逐位一致** + 总时长下降 ≥ 5% + 无 hang（跑 3 次全量门禁）。

### E2 回合 3→2（与 E1 同批验证）

- 若 E1 通过，则第二回合内合并 C4→C5→C7；核对 L1 峰值是否仍 ≤ 512 KiB、T/q10/Akk 的 L1 槽位是否冲突。
- 判据：总时长下降 ≥ 3%（相对 E1 之后），且 `δ_flag` 部分可解释。

### E3 GVA norm 去重

- 变更：norm 只由 GVA 组首 owner 做一次，结果按 HK 写 workspace；其余 head 读归一化结果。
- 消融矩阵：`HK/HV ∈ {1/2, 1/4}` × `norm on/off`（4 组），记录 V0 时长与总时长。
- 判据：`Δ总时长 ≈ −(1 − HK/HV) × norm占比 × V0占比` 的 60% 以上（否则说明被掩藏）；
  且不引入跨 AIV UB 共享（若必须引入，按 design §5 的警告单列风险）。

### E4 head window 消融（Q6）

- 变更：`kHeadsPerGroup ∈ {1,2,4,8}`（L1 4 head lane 与 UB 预算允许的话）。
- 观测：总时长中位、每核 work item 数与最大工作量的**尾波比**（培训 §7.2 的算法：`ceil` 分配后最大/最小工作量比）、
  L1/UB 峰值。
- 判据：差异在噪声内（< 2%）⇒ 维持 4 并在文档写明"未证明最优"；若某窗口稳定 ≥ 3% 更优 ⇒ 记录为改进项。

## 5. 执行顺序与预算

| 阶段 | 内容 | 机器 | 预计耗时 | 前置 |
|---|---|---|---|---|
| P1 | S1+S2+S3（纯静态，无真机） | 开发机 | 0.5 天 | — |
| P2 | D0+D1（基线 + 整核画像） | A2 真机 | 1–2 h | P1 |
| P3 | D2 分 stage 计时（7 版 × 3×10） | A2 真机 | 4–6 h（含编包） | P2 |
| P4 | D3 握手代价 | A2 真机 | 1 h | P3 |
| P5 | E1+E2（V6 提前 + 回合合并） | A2 真机 | 半天（含门禁与 hang 排查） | P3+P4 |
| P6 | E3（GVA 去重）4 组消融 | A2 真机 | 半天 | P5（或独立） |
| P7 | E4（head window）4 组消融 | A2 真机 | 半天 | P5 |
| P8 | 结论与决策文档 | — | 0.5 天 | 全部 |

单批口径：`msprof op` 一组 = 3 shape × 3 次 ≈ 15–20 min；全量门禁 ≈ 5 min。

## 6. 风险与"可能推翻结论"的情形

1. **V6 与 C2 的时长比**：若实测 `V6 ≫ C2`，提前后 V3 仍要等 V6 ⇒ 收益被自身限制（此时应改为"V6 拆薄/只提前可并行的部分"）。
2. **集合同步配平**：0x2 下 AIC 一次 set 唤醒两个 AIV、两个 AIV 都 set 才唤醒 AIC；改动回合数会改变计数，
   配对错误表现为 hang 而不是数值错——必须跑满 3 次全量门禁 + hang 看护。
3. **C7 的 GM 可见性**：V6 提前后 C7 读的 `K_beta_g/V_beta` 与写者之间少了显式回合，需确认真有
   `DataSyncBarrier`/FIX 语义覆盖（否则会变成"偶发读到旧数据"，属于典型隐性竞态）。
4. **共享区换义**：V6→V3 共用 `kSharedBase/kSharedScratch`，顺序变化会改变换义约定（约束 11/17 允许换义，
   但必须证明生命周期不重叠）。
5. **4 未必最优**：培训 §7.4 明确"4 是当前采用的选择，不能从双 AIV 证明最优"；若消融显示 2/8 更好，
   需要同时检查 L1 占用与尾块（`H` 非 4 倍数）行为。
6. **A3 无法实测**：910_93 只做构建级 + 静态审计；任何"在 A3 上更好"的结论都不能在本轮给出。

## 7. 交付物

1. `design_a2a3_constraint_audit.md`：19 条约束逐条证据表（含代码行号）。
2. `design_a2a3_budget.md`：UB/L1/workspace 每 stage 峰值与余量。
3. `design_a2a3_depgraph.md`：依赖图 + 两条候选调度的关键路径公式。
4. `perf/round_000_baseline/` 起各阶段的 msprof 原始数据与解读。
5. **决策文档**：stage 表 / 回合表 / 数据路径表 / 预算表 / 风险清单，并明确
   "Q1–Q6 各自的结论 + 支撑证据编号 + 未采纳方案及原因"。
6. 若结论支持：给出 arch22 调度改动的实现批次计划（每批 = 位级 + 门禁 + 性能 A/B）。

## 8. 判据速查（避免"看起来更快"）

```text
功能：L1 位级必须逐位一致（屏障/调度改动不允许改数值）；L2/L4 全绿；V6 提前这类改动额外跑 hang 看护（≥3 轮）
性能：同机同频，3 warmup + 10 次取中位；改善需 ≥ 3~5% 且大于重复采样极差，否则视为噪声
预算：UB ≤ 184 KiB(A2/A3) / 248 KiB(A5)；L1 ≤ 512 KiB；workspace 峰值与 host 侧一致
同步：每次 set 都有配对 wait；未配平的表现是 hang，不是数值错
```
