# pre_process_fwd_kernel_merged 开发期验证记录

本文件是开发期记录，收敛到算子验收 README 后删除。

## 01 接口修订（2026-09-21）

用户在 03 阶段提出三条要求，触发**接口与支持范围变化**，按主 Skill 的恢复矩阵回到 01：

| 要求 | 处理 | 影响 |
| --- | --- | --- |
| `chunk_size=128` 不用 | `api.md` 第 5 节把 `BT` 收敛为"固定 64；128 不纳入本轮"，第 6 节新增 host 拒绝项 | 支持范围 |
| `V=256` 不用 | `api.md` 第 5 节把 `V` 收敛为 `V<=128`，第 6 节新增 `V>128` 拒绝 | 支持范围 |
| `K=256` 不用 | `api.md` 第 5 节把 `K` 收敛为 `K<=128`，第 6 节把拒绝条件改成 `K>128` | 支持范围 |
| GVA 支持就行（`HK` 与 `HV` 可以不一致、成倍数） | 保留 `k` 在 `HK` 维、`w/v/u/g/gk` 在 `HV` 维，kernel 按 `hk = hv // (HV/HK)` 复用 key head；`api.md` 第 3 节显式写明 head 约定，第 6 节保留 `HK>HV` 与 `HV%HK!=0` 的拒绝项 | 接口（`operator_contract` 回到 provisional） |

同时确认**布局统一为 BNSD**（与仓内其他 AscendC 算子一致），并顺带把分档改为由容量模型
算出（`docs/design.md` 3.4）。

### 本次修订的验证

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 标杆自检 | `python reference/reference.py` | `(4, 128, 256)`、`finite` 通过 |
| 值域校准（含 GVA） | `python scripts/calibrate_reference.py`（T=11264） | 6 个变体（`HK=HV` 与 `HV:HK=1:2` × `g`/`gk`/`dplr`）全部跑通 |
| 与修订前一致性 | 同上，"同为 FP32、tile 分组"一行 | `h_half abs=7.213e-03`（tile=16）/ `8.029e-03`（tile=8），与修订前记录的 `7.2e-3 ~ 8.0e-3` **逐位一致** |

过程中有一次理解偏差：曾把"所有输入输出都是 valueHead"读成"`k` 也要在 `HV` 维、GVA 展开
移到调用方"，并据此改过一版 `api.md`/`reference.py`。经用户澄清——**GVA 要支持就意味着
`HK` 与 `HV` 可以不一致**——已全部改回：标杆仍是 `k[T,HK,K]` + 内部 `head_of_k` 索引，
`api.md` 仍是 `k` 在 `HK` 维。上表 `HK=32 HV=32` 两行与修订前逐位一致
（`7.213e-03` / `8.029e-03`），说明改回后与 02 阶段冻结的标杆完全等价。

范围收敛（`K`/`V` 都到 128）之后，`docs/design.md` 3.4 的容量模型给出**单一配置**：
`BLOCK_SIZE=128`、`SPLIT=1`、`hK=K`，18 个 `(K,V,mode)` 组合无例外，
最紧一档 L1 峰值 256 KiB / UB 164 KiB。因此原先把 `K=256`/`V=256` 逼出来的
T0/T1/T2 分档与 `SPLIT=2` 方案全部作废。

`operator_contract` 与 `golden_contract` 回到 `provisional`，等待用户确认修订后的
`docs/api.md`；确认后按恢复矩阵前进。

### 用户确认与状态回归

用户确认了修订后的 `docs/api.md`。按恢复矩阵的"01 用户确认接口"一行先切到
`stage=reference`、`golden_contract=provisional`；标杆随后在本机（torch 2.12.0+cpu）重跑
自检与值域校准并复现 02 阶段冻结的全部数字，达到"02 标杆实际运行并验收通过"，于是
`operator_contract` 与 `golden_contract` 一并置回 `frozen`、`stage=design`，继续 03。

## 03 方案设计

### 第一步：Stage 划分（已完成）

产物在 `docs/design.md` 第 1、2 章与第 3 章的容量闭包、第 6 章；第一步评审的 6 个通过条件
逐条记录在 `docs/design.md` 2.1.5。关键结论：

* Stage 链为 `S0(Cube) -> S1(Vector) -> S2(Cube) -> S3(Vector) -> S4(Cube)`，5 个 Stage 是
  依赖与容量下的最小值；`S2`/`S4` 同为 Cube 但因 `S4` 的操作数派生自 `S2` 的 Cube 输出而
  不能合并。
* 两半（`h` 与 `m`）不是并行关系而是共享关系：`L_c^T` 只从 L1 取一次进 L0A，同时服务
  `dH` 与 `Kw`，`k`/`w` 每 chunk 只读一遍（上游为 4 遍）。
* 分档由 `scripts/partition_model.py` 的容量模型算出；范围收敛到 `K<=128`、`V<=128`、
  `chunk_size=64` 后，18 个 `(K,V,mode)` 组合全部落在 `BLOCK_SIZE=128 / SPLIT=1 / hK=K`
  同一组配置，最紧一档 L1 峰值 256 KiB、UB 164 KiB。

### 第二步：具体详设（已完成）

`docs/design.md` 2.7（CrossCore 通知分配、核内 HardEvent 分配、覆盖 tail/空分片/varlen/
`NT=1`/退出排空的完整伪代码）、2.8（逐 Stage 的 CATLASS TileShape 与合法分支）、
3.1（L0 占用表）、3.2（24 个 `TilingKey` 与 `workspace=0`）、3.3（编译期模板与运行时 tiling
的字段划分）已补齐。

随后按参考设计（`chunk_gdn_fwd_prepare` / `gdn-fwd-h`）的粒度把 3.1 重写成完整的片上地址图：

| 小节 | 内容 |
| --- | --- |
| 3.1.1 | L1 固定地址图：9 个区间（`L1[0,64)` `m` … `L1[192,256)` `M_c`），小计 256 KiB，最大连续空闲 `L1[256,512)` = 256 KiB；**全部 24 个 TilingKey 共用同一张图** |
| 3.1.2 | UB 固定地址图（每 AIV）：6 个区间（`UB[0,32)` `h` 常驻 … `UB[128,132)` 小量），小计 132 KiB，最大连续空闲 `UB[132,248)` = 116 KiB |
| 3.1.3 | L0 槽位与复用：逐槽列出区间、内容、使用者与分时复用关系；L0C 阶段内峰值 128 KiB，L0A/L0B 阶段内各 64 KiB（S4 独占） |
| 3.1.4 | 生命周期表：每个 buffer 的生产 → 各次消费 → 释放/复用点 → 份数依据；并补上窗口初始化（写 `m=I`、清零 `h` 分片、发布 `init_ready`） |

同时把 2.2~2.6 五个 Stage 的"地址、大小、份数"条目都改成引用具体区间，并在 2.7.1 增加
`init_ready` 这条跨核边（CrossCore flag 由 7 个增至 8 个），2.7.3 补上 `InitWindow`。
容量模型也同步修正：`dH`/`Kw` 的 staging 在 `PIPE_SERIAL` 下只需单份，UB 峰值从保守的
164 KiB 收敛到 132 KiB（模型里的 `x2` 系数只适用于将来开 `PIPE_PACK_OVERLAP` 的候选）。

`docs/design.md` 通过 `validate_design.py`（`--design` 与 `--workflow` 两种调用）。
剩余未决项见 `docs/design.md` 5.2，其中第 1~4 项需要用户提供模型 case 与性能口径。

### 规格与 TilingKey 收敛（2026-09-21，用户侧核查结论）

用户提供了一份侧线核查结论，据此把 `K`/`V` 从"`<=128` 的档位"改为**写死 128**：

* 仓内**所有** AscendC 算子的 `K` 都精确要求 128 —— `npu_chunk_fwd_h`（1539 行
  `"K and V must both be 128."`）、`npu_chunk_gated_delta_rule_bwd`（735）、
  `npu_chunk_gdn_bwd_intra`（1177）、`npu_chunk_kda_bwd`（3118）、
  `npu_chunk_kda_bwd_recompute`（4230）、`npu_chunk_kda_fwd_finalize`（1679）等，
  只有 `npu_chunk_gated_delta_rule_fwd_prepare` 与 `npu_recurrent_kda` 把 `V` 放宽到 `{128, 256}`。
* 参考设计文档（`gdn-fwd-h`、`gdn-backward-finalize`、`chunk_fwd_o`、`chunk_gdn_fwd_prepare`）
  都把 `BT/K/V` 当固定规格、host 显式拦截、"不能静默 fallback"。
* `GDN泛化用例表.xlsx` 的 34 行 `Kdim` 全是 128、`Vdim` 只有 128 与 256。

改动清单：

| 位置 | 改动 |
| --- | --- |
| `docs/api.md` | `K`/`V`/`chunk_size` 改为固定规格（128 / 128 / 64）；第 6 节拦截 `K!=128`、`V!=128` |
| `docs/design.md` 3.2.1 / 3.3 | `TilingKey` 由 `(K_TILE, V_TILE, GATE_MODE, GATE_T)` 的 24 个收敛为 `(GATE_MODE, GATE_T)` 的 **6 个**；字段划分表删掉 `K_TILE`/`V_TILE` 两行 |
| `docs/design.md` 3.4 | 原"分档参数化"整章删除，改写为"`K`/`V` 写死为 128 的依据"+ 一组具体容量数字 |
| `docs/design.md` 全文 | 移除 `BLOCK_SIZE` / `SPLIT` / `hK` 三个旋钮；`Nwork` 简化为 `= HV`；1.1、1.2、2.1.4、2.2、2.4、2.6、2.7、2.8、3.1、4.3、5.1、5.2 与 R20 同步 |
| `scripts/partition_model.py` | 删除，改为 `scripts/capacity_check.py`：只算固定规格下的 L1/UB/L0C 并断言与 3.1 的地址图一致 |
| `docs/pre_process_fwd_kernel_merged_泛化用例.xlsx` | 重新生成，正向用例 38 → **36 条** |

用例表的换法：删掉 8 条"分档边界"（K/V 档位）与 3 条 `V=64` 的三路径基线，补 9 条
"精度-窗口规模"（`T_win=1023/8191/16383/32767` 的长窗口与非 64 倍数、varlen+GVA 组合、
`HV=64/128` 的尾部负载）。其余分组不变：尾块 6、变长 3、GVA 5、dtype 4、性能 6。
异常用例 12 条同步改为 `K!=128`、`V!=128`。

验证：`capacity_check.py` 复算 L1 192 / 208 / 256 KiB、UB 132 KiB、L0C 阶段内 128 KiB，
与 3.1 的地址图逐项一致；工作簿公式错误扫描 0 命中，覆盖计数（GDN 29 / KDA 4 / DPLR 3、
GVA 8、尾块 11、变长 5、Nwork 最大 128）与 36 条自洽。

GVA 的倍数**不设上限**，采用 `npu_chunk_fwd_h` 的口径（而不是
`npu_chunk_gated_delta_rule_fwd_prepare` 的 `HV/HK ∈ {1,2,3,4}`），用例表的 1:8 / 1:32 保留。

### 用例覆盖扩均衡与 DPLR 转预留（2026-09-21）

用户反馈两处覆盖不均，据此重排：

| 问题 | 处理 |
| --- | --- |
| 算法路径覆盖不均（KDA 4 条、DPLR 3 条 vs GDN 29 条） | 把「形状维度」与「算法路径」做成**叉乘**：`精度-窗口规模` = 路径 × 5 种 `T_win`、`精度-变长` = 路径 × 4 种 `cu_seqlens`、`精度-dtype` = 路径 × 2 种 gate dtype。KDA 由 4 条增至 13 条 |
| 变长用例偏少（5 条） | `精度-变长` 由 3 条扩到 8 条（两路径 × 4 种 `cu_seqlens`），加 GVA 组里的 2 条 varlen 组合，共 **10 条** |
| DPLR 暂时用不到 | **转为预留**：设计侧全部保留（`USE_BG` 分支、S2 的 `bg^T @ V_c`、`M_c` 取 `+`、L1 的 `bg`/`V_c` 槽），但用例、实现、验收都不进本轮 |

重排后正面用例 **36 条**：`精度-窗口规模` 10、`精度-变长` 8、`精度-GVA` 7（仅 GDN）、
`精度-dtype` 4、`精度-核数负载` 3、`性能` 4。路径分布 **GDN 23 / KDA 13**；
GDN 多出的 7 条来自 GVA（GVA 只存在于 g-only 路径）。异常用例 13 条不变。

工作簿已覆盖到 `outputs/` 与算子工程的 `docs/`，公式错误扫描 0 命中，覆盖计数与 36 条自洽。

### 硬件绑定参数审计（2026-09-21）

用户指出 `blockDim = min(28, Nwork)` 里的 28 是硬件绑定值，要求盘清还有哪些。审计后：

| 分类 | 项 | 处理 |
| --- | --- | --- |
| **应运行时读取** | `AIC_NUM`（可用 Cube 核数） | §2.1.4 与 R20 由 `min(28, Nwork)` 改为 `min(AIC_NUM, Nwork)`，并写明 host 从设备读后经 tiling 传入、kernel 不写死 |
| 编译期容量常量 | L1 512 / UB 248 / L0A 64 / L0B 64 / L0C 256 KiB | 集中在新的 §3.5 平台常量表，注明取自 `docs/precheck.md`，换 SoC 要重算 3.1.1~3.1.3 |
| **本设计自定的预留** | L1 按 448 KiB、UB 按 224 KiB 计（512−64 / 248−24） | §3.5 明确标注**没有硬件依据、不能按比例缩放**，04 必须用目标版本的组件实际占用核对 |
| 架构/分型绑定 | `ArchTag=Arch::Ascend950`、`CATLASS_ARCH=3510`、NZ 分型 16 列、512 B 对齐 | 列入 §3.5，换架构需要重核 S1 的手工 NZ 构造 |
| 资源上限待核对 | CrossCore flag 8 个、片上事件 9 个 | 列入 §3.5，需对照目标版本上限 |
| 仅用于估算 | `cube_freq=1650 MHz`、BF16 约 378 TFLOPS、HBM 约 1.5 TB/s | 只进 4.2 的 281 us 与 3.2 的 138 us 估算，不参与 kernel 逻辑 |
| 非硬件绑定 | `K=V=128`、`BT=64`、`BNSD`、`B=1`、`HV`、`NT`、`Nwork` | 算子接口规格或 shape 派生量，不随硬件变 |

`scripts/capacity_check.py` 顶部新增 `PLATFORM` 字典，容量与预留集中一处，换平台只改那一段；
输出会打印平台标识与上限，便于核对。

### 验收顺序与全用例性能承诺（2026-09-21）

用户明确：**先精度、后性能，且所有精度用例都是模型场景，每一条都要保证性能**。据此调整
`docs/design.md` 第 4、5 章：

| 位置 | 改动 |
| --- | --- |
| 4.1 | 新增"验收顺序：先精度、后性能"与"每条用例都要双过"：36 条正向用例**每一条**都要同时满足精度阈值与 1.0x 性能目标，不做"边界 case 放行" |
| 4.2 | 由"一条 model case 的目标"改为**逐用例建档框架**：4.2.1 目标框架、4.2.2 逐用例字段表（H20 基线 / 昇腾实测 / 倍率 / 结论）、4.2.3 已有参考值与设计下界。并明确**其余 35 条还没有 H20 基线**，需要逐条采集 |
| 4.3 | 新增"这 36 条同时是性能用例"与按预期不达标风险的分级表 |
| 5.1 | 新增两条风险：短窗口的固定开销（`NT=1`/`NT=2`）、全用例性能承诺 |
| 用例表 | 新增 `性能风险` 列（高 5 / 中 6 / 低 25），表头加一行说明"36 条既是精度用例也是性能用例" |

风险分级里两项标"高"，都需要在 **04 阶段就预留优化手段**，不能等到 05：

* `T_win=1` 与 `T_win=100`（`NT=1`/`NT=2`）：时间几乎全由 kernel launch + 窗口初始化
  （写 `m=I` 的 64 KiB）+ 3 次 MMAD 构成，没有东西可摊薄。预留手段是"首 chunk 直接把 `M_c`
  写进 `m` 区、跳过 `m = M_c @ I`"，优先砍窗口初始化而不是加 Stage。
* `HV=128`（`Nwork=128`）：4.57 波，尾部只剩 16 个核。预留手段是 R20 的 `CG` 候选或更细的
  工作项粒度。

### 并行度审计：`B × HV` 与「小 HV 用不满核」（2026-09-21）

用户追问：`HV` 小是不是用不满核？Batch 参与分核吗？NT 能不能分核？核对了上游 kernel、
仓内 chunk 依赖参考与 `gdn-fwd-h` 设计后确认，并把结论写进 `docs/design.md` 2.1.4：

| 问题 | 结论 | 依据 |
| --- | --- | --- |
| Batch 参与分核吗 | **参与，而且是并行度的一半**。`Nbase` 就是 sequence/batch 数，`Nwork = B × ceil(HV/CG)` | R20 原文"chunk 间有依赖时 `Nbase` 为 sequence 总数" |
| NT 能分核吗 | **不能**。chunk 维被依赖砍掉 | R20 同上；仓内参考 `chunk-dependent-development.md` 第 2 条"同一序列的 chunk 按依赖顺序执行；**不同序列或互不依赖的 head 可以并行**"，第 4~5 条"每核按 head round 推进一个 chunk……当前 chunk 的全部 head round 完成状态更新后，再进入下一 chunk"；参考算子 `ChunkGatedDeltaRuleFwdH` 也只按 sequence × head 分核，没有任何一层按 chunk 或按列分核 |
| `HV` 小会怎样 | **会严重用不满核**：`B=1, HV=8` 只有 8 个工作项，28 个 AIC 里 20 个空转 | `GDN泛化用例表` 的 V1/V2 就是 `B=1` 配 `HV=16/8` |

三条出路写进 2.1.4，推荐第一条：

1. **放开 `B > 1`**（首选）：同一 part 的多个 batch 元素合成一次调用，`Nwork = B × HV`。
   上游 kernel 里留着 `MULTI_SEQS` 这条能力（`grid dim 2` 承载 sequence 号、
   `hm += i_n * HV * K * (K+V)` 按它偏移），但这条 CP 路径上未启用。零额外搬运。
2. 按 `hm` 列拆工作项（上游 `i_col`）：`k`/`w` 按 `SPLIT` 倍数重复读，HBM 流量上升。
3. 把 m 半边的预算拆成独立并行 phase：`M_c` 与 `m` 无关可预并行，但需要额外缓冲与跨 phase 同步。

**`B>1` 因此被提升为 5.2 的第 0 项（04 之前的阻断项）**，同时写进 5.1 第 8 条风险。
用例表也补回了 `B` 列：新增「精度-并行度」组 6 条，`B × HV` 从 `1×8`（`Nwork=8`）扫到
`64×8`（`Nwork=512`）；其余各组的 `(B, HV)` 配对改取 `GDN泛化用例表` 里的真实组合
（如窗口规模组用 `B=16, HV=32`、变长组用 `B=1, HV=32`）。正向用例 36 → **39 条**，
路径分布 GDN 24 / KDA 15。

#### 更正：竞品 CP 路径确实是 `B=1`

用户追问"竞品哪来说明 B 不会 >1"，复查后确认**不是靠文字声明，而是代码结构**，
并把上一轮"上游本来就支持 B>1"的说法改准确：

| 证据 | 位置 |
| --- | --- |
| `hm = k.new_zeros(HV, K, V + K)` —— **没有 batch 维** | 非 zigzag 路径 |
| `hm = k.new_zeros(2, HV, K, V + K)` —— 首维是 front/back `part`，也没有 batch 维 | zigzag 路径 |
| `grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)` —— **二维 grid** | 两条路径 |
| `MULTI_SEQS=False` —— 全文件 `MULTI_SEQS` 只出现 4 次（kernel 形参、kernel 内分支、两处调用），**从没传过 `True`** | 第 64、68、833、910 行 |
| `cu_seqlens[-2:]` / `cu_seqlens[fns-1:fns+1]` —— 只取一段窗口 | 两条路径 |

反向证据：kernel 内确有 `MULTI_SEQS` 分支（`i_n = tl.program_id(2)`、`hm += i_n*HV*K*(K+V)`），
但这条路径上 `hm` 的布局装不下 sequence，属**未启用能力**；多 sequence 只出现在
`merge_fwd_bwd_kernel` 的 `INTRACARD_MODE`（`ag_hm` 注释里的 `[S_split, HV, K, K+V]`）。

**结论**：对标口径下 `B=1`；放开 `B>1` 是**超出竞品 CP 路径的扩展**（允许，但必须配套定义
1.0x 口径——竞品 B 次调用 vs 我们 1 次，只有按"一个 rank 的整个 pre_process 总时长"比才公平）。
这两项在 5.2 里已联动标注。

#### 再更正：竞品在系统层支持大 B，只是不在这个 kernel 里

用户继续追问"当输入的 B 很大时竞品是怎么处理的，还是默认只处理一个 sequence 吗"。
读 `fla/ops/cp/context.py` 与 `gdn/chunk.py` 的调用链后，把结论修正为**分三层**：

| 层 | 并行度 / 承载维度 | 证据 |
| --- | --- | --- |
| CP 编排层（`context.py`） | **按 token 切**：`part_len = total_tokens / num_parts`，`num_parts = W`（contiguous）/ `2W`（zigzag）；一个 rank 的窗口**可以跨多个 sequence**（`_interval_cp_meta` 的 `local` 是区间内序列边界 clamp+shift 的结果） | `get_cp_cu_seqlens` |
| CP 编排层（wrapper） | `initial_state = k.new_zeros(N, HV, K, V)`，**`N` 就是本 rank 的 sequence 数**；只有边界行被 `merge` 写值（`initial_state[0]` / `initial_state[fns]`） | 804、873、846、1142 行 |
| **本 kernel** | **grid 恒为 `(列块数, HV)`、`hm` 恒无 sequence 维**；B 再大也不多开 program | 804、873、810、889 行 |
| 下游 `chunk_gated_delta_rule_fwd_h` | 拿 `initial_state[N,...]` 处理本 rank 的全部 N 个 sequence，sequence 并行度落在这里 | GDN `chunk.py` |

所以之前"竞品一次只处理一个 sequence"的说法**只对本 kernel 成立**，对系统不成立。
`MULTI_SEQS`（grid dim 2 + `hm` 的 sequence 前导维）正是为"一段窗口含多个 sequence"准备的，
只是这条 CP 路径没有启用。

**对标含义**：大 B 时我们和竞品这一层同构（都不吃 B）；"用不满核"在竞品里同样存在，
它把 B 的并行度交给了下游算子。我们要不要吃这块是独立选择，若吃则 1.0x 口径必须按
"一个 rank 的整个 pre_process 总时长"定。已写进 `docs/design.md` 2.1.4 与 5.2 第 0/3 项。

> 未完全验证的一点：当一段窗口跨多个 sequence 时，"只喂最后一段"的 `hm` 与 `merge` 的
> 逐 rank 复合 `b_h = m@b_h + h` 如何对多 sequence 窗口保持正确，我没有实跑验证
> （推导上它能成立，因为每个 sequence 都从零状态开始，跨 rank 携带的状态只在窗口起点
> 落在序列内部时非零）。这一条不影响我们的设计选择，若后续要完全对齐竞品的多 sequence
> 窗口语义，需要单独跑一次数值验证。

### 更正：变长/定长的标准语义（2026-09-21）

用户指出：**`cu_seqlens` 多段 = 变长序列；定长就是 `B != 1`**，这才是标准约定。复查后确认
用户说得对，是我们在 01/03 里把它窄化成了"单窗口"，而且 `api.md` 自己内部就不一致：

| 位置 | 原文 | 问题 |
| --- | --- | --- |
| §5「序列」行 | "定长（`cu_seqlens` 缺省）与变长（`cu_seqlens` 存在）" | ✅ 与标准一致 |
| §3.2 / §5「`B`」行 | "`B` 为窗口条数……通常 `B = 1`" / "本轮 CP 用例以 `B = 1` 为主" | ❌ 把 batch 窄化成"窗口条数" |
| §3.2 `cu_seqlens` 行 | "变长模式；本算子只消费当前窗口的起止（`N = 1`）" | ❌ 把变长窄化成单段 |
| §6 | "变长模式下 `N != 1`：本轮拒绝" | ❌ 同上 |

**标准约定（也是仓内 `chunk_fwd_h` 与上游 `chunk_delta_h` 的写法）**：

```text
定长（dense）:  cu_seqlens = None  →  Nseq = B
变长（varlen）: B = 1，T 轴打包多段 →  Nseq = len(cu_seqlens) - 1
上游原式: N = B if cu_seqlens is None else len(cu_seqlens) - 1
R20:     Nbase 取 sequence 总数
```

据此改齐了：

| 位置 | 改动 |
| --- | --- |
| `docs/api.md` §3.2 | 新增"序列的两种表达"表（定长 `Nseq = B` / 变长 `Nseq = len(cu_seqlens)-1`），并注明与上游 `N = B if cu_seqlens is None else ...` 一致 |
| `docs/api.md` `cu_seqlens` 行 | 改为"变长模式的序列边界（T 轴打包多段），`N = 序列数 >= 1`" |
| `docs/api.md` §5/§6 | `B` 改为 batch；序列行写明两种模式都支持多序列；删掉"`N != 1` 拒绝"，改成 `cu_seqlens` 合法性校验 |
| `docs/design.md` 2.1.4 | R20 块改为 `Nseq`（定长 `= B`、变长 `= len(cu_seqlens)-1`）→ `Nwork = Nseq × HV`；工作项带自己的 `bos`/`T_win`/`NT`/`M(c)` |
| `docs/design.md` 1.1/1.2/3.5/4.3/5.1/5.2 | 同步（`B` 为 batch；用不满核的条件从 `B=1` 改成 `Nseq=1`） |
| `gen_cases.py` + 用例表 | 新增 `Nseq` 列；变长组改成**真多段**（2 段等长 / 3 段等长 / 3 段不等长含 NT=1 短段 / 单段对照）；`Nwork` 公式改为 `= Nseq × HV` |

**待办**：`reference/reference.py`（02 工件）目前仍只接受 `cu_seqlens` 两元素（单段）。
按新契约它需要泛化成"遍历 batch / 遍历打包段"，属于 02 的机械改动，等接口修订确认后一并做。

### `hm` 的 `B` 维 vs 上游的 part 维（2026-09-21）

用户问为什么我们的 `hm` 带 `B` 维而上游不带。核对后确认**这不是偏离**，而是把上游的
"part 维"统一成了 `B` 维：

| 场景 | 上游 `hm` | 本算子 `hm` |
| --- | --- | --- |
| 单 part（contiguous CP 或非 CP） | `[HV, K, V+K]` | `B = 1` → `[1, HV, K, V+K]`（去掉 size-1 维后**逐字节相同**） |
| zigzag CP（每 rank 两个 part） | `[2, HV, K, V+K]` | `B = 2` → `[2, HV, K, V+K]`（**形状完全一致**） |

带 `B` 维的理由：输入输出对称；与仓内 `npu_chunk_fwd_h` 的输出 `h [B,HV,NT,K,V]` 约定一致；
`B > 1` 正是并行度 `Nseq × HV` 的来源。上游"无 `B` 维"是它一次只算一段窗口的调用粒度
导致的，它一用 zigzag 就不得不加一个 `2` 维。已写进 `api.md` 3.4 与 `design.md` 1.4。

真正剩下的差异只有**返回值 vs 原地写**（上游 `hm` 是指针入参、包装层预分配；我们返回新张量），
已在 `design.md` 5.2 第 7 项挂着。

### CP layout 与算子感知边界（2026-09-21）

用户追问 layout 由谁决定、算子要不要感知。核实结论：

* `layout` 由建 CP context 的一层决定：`build_cp_context(..., layout=...)`，默认 `'contiguous'`；
  `FLACPContext.layout` 只是记录字段，kernel 侧唯一的消费者是 `chunk_delta_h.py` 的
  `if context.layout == 'zigzag'`。没有配置文件、环境变量或启发式。
* **算子不感知 `layout` 这个名字，也不应加这个参数**（它是编排概念）。算子要感知的是它的后果：
  **本 rank 的窗口切成几个 part、每个 part 覆盖哪些 token**。

| 场景 | 我们的表达 | 是否够 |
| --- | --- | --- |
| contiguous（默认，1 个 part = 整段） | `B` 个独立窗口 + 一维 `cu_seqlens` | ✅ |
| zigzag（2 个 part = 本地 buffer 的前后两半，切点 `part_len`） | `B = 2` 表达"两个 part"，**但切点不在 `cu_seqlens` 里** | ⚠️ 缺一条 part 边界 |

上游用 `context.front_num_seqs` 补这条边界（两次调用分别写 `hm[0]`/`hm[1]`）。我们若要一次
调用吃下两个 part，可选 (a) `cu_seqlens` 升成 `[B,N+1]`、(b) 加 `part_offsets[B+1]`、
**(c) 不感知（本轮默认，接口零改动，与上游现状完全一致）**。
三种做法的对比写在 `design.md` 1.4，并作为 5.2 第 9 项待确认。

## 02 标杆生成

### 参考实现与固定版本

| 项目 | 内容 |
| --- | --- |
| 参考实现 | `fla-org/flash-linear-attention` @ `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2` |
| 参考文件 | `fla/ops/cp/chunk_delta_h.py::pre_process_fwd_kernel_merged`（第 42 行起） |
| 参考文件 SHA256 | `a6ed6aaaf0bc6c7dc8c9235a5a9346d14a0bbeb3630b8baf8f1c2bc7ac92435c` |
| 许可证 | MIT |
| 标杆源码 | `reference/reference.py`（唯一可编辑） |
| 生成产物 | `reference/definition.json`（由 `definition.template.json` 自动嵌入标杆源码） |

### 接口语义 → 参考实现位置 → CPU 实现方式

| 语义 | 参考实现位置 | CPU 标杆实现 |
| --- | --- | --- |
| 窗口起止 `bos/eos`（恒走 varlen 分支） | 第 75–81 行 | `cu_seqlens=(bos, eos)`，缺省 `[0, T]` |
| k 的 head 展开 `i_h // (HV // HK)`（2026-09-21 起在调用方完成） | 第 89 行 | 调用方先把 `k` 展开到 `HV`；标杆直接按 `hv` 索引 |
| h 半边的列块与 `hm` 左块 | `is_h_part`，第 86、244–254 行 | `h` 累加器 → `hm[:, :, :V]` |
| **先用未衰减的 h 算 `v_decay`，再衰减 h** | 第 137–178 行 | 同名顺序（易错，见 §踩过的坑） |
| GDN/KDA `v_new = v − v_decay`；DPLR `v2 = v_decay + u` | 第 153–161 行 | 同名分支 |
| `USE_G`：`v_new *= exp2(g_last − g)`、`h *= exp2(g_last)` | 第 166–178 行 | 同名 |
| `USE_GK`：`h[k,:] *= exp2(gk_last[k])` | 第 181–197 行 | 同名 |
| `h += kᵀ @ bf16(v_new)`；DPLR 追加 `bgᵀ @ bf16(v)` | 第 198–240 行 | `einsum("thk,thv->hkv", ...)` |
| m 半边 `M_c = diag ∓ kw`、`m = M_c @ m` | 第 272–317 行 | `M = diag ∓ kw`，`m = M @ m` |
| `kw = kᵀ @ w`（DPLR 用 bg） | 第 276–284、306 行 | `einsum("thk,thj->hkj", ...)` |
| 输出打包 `hm[.., :V]=h`、`hm[.., V:]=m` | 第 244–254、315–317 行 | `torch.cat([h, m], dim=-1)` |
| 衰减一律 base-2 | `exp2`（全文件） | `torch.exp2` |

### 契约的数值精度（四个点）

它们改变结果的程度远超容差，属于接口契约的一部分，标杆必须复刻：

1. **累加精度为 FP32** —— 上游的 `h`/`m` 累加器都是 `tl.zeros(..., dtype=tl.float32)`，
   所有 `tl.dot` 都往 FP32 累加器累加；因此标杆默认 `accum_dtype=torch.float32`
2. `h` 在进 `w @ h` 前降到输入 dtype（BF16）——`round_h_to_input_dtype`
3. `v_new` 在进 `kᵀ @ v_new` 前降到输入 dtype（BF16）——`round_v_new_to_input_dtype`
4. `M_c @ m` 每个 chunk 更新后回落 FP32（上游 `input_precision="ieee"`）
   ——`round_affine_chain_to_float32`

把 `accum_dtype` 换成 `float64` 并关掉三个开关即"纯数学"版本，仅用于灵敏度对照。

> **第 1 条是 2026-09-20 与 H20 实测对齐时才发现并补上的。** 最初标杆用 FP64 累加，
> 与 H20 `ieee` 实测在 h 半边差 9.371e-03；单独把累加器换成 FP32（其余完全相同）
> 就产生 8.735e-03 —— 两者一致，说明差异全部来自这一条，而不是公式或别的舍入点。

### 值域校准

用例：`T=11264, BT=64, K=V=128`，输入按 delta 规则构造（`k` 沿 head 维单位化、
`w = beta·k`、`beta~U(0,0.02)`、`bg = gamma·k`、每 chunk log2 总衰减 0.013），`NT=176`。
脚本：`scripts/calibrate_reference.py`。

**（a）契约 vs 纯数学 FP64** —— 即"实现不遵守契约会差多少"：

| 变体 | \|h\|max | \|m\|max | 契约 vs 纯 FP64 |
| --- | --- | --- | --- |
| g (HK=32) | 17.893 | 0.0593 | abs 3.282e-2 / rel 1.83e-3 |
| gk (HK=32) | 17.974 | 0.0591 | abs 3.249e-2 / rel 1.81e-3 |
| dplr (HK=32) | 29.070 | 0.1401 | abs 4.612e-2 / rel 1.59e-3 |
| g (HK=16) | 19.230 | 0.0599 | abs 3.181e-2 / rel 1.65e-3 |
| gk (HK=16) | 19.308 | 0.0597 | abs 3.035e-2 / rel 1.57e-3 |
| dplr (HK=16) | 23.975 | 0.1398 | abs 4.094e-2 / rel 1.71e-3 |

**（b）同为 FP32、只是累加顺序不同** —— 即"任何正确实现对标杆的预期偏差"。
做法：把每个 chunk 的 token 累加按 16/8 分组后再相加，模拟 NPU Cube 的 tile 累加。

| 变体 | tile | h_half | m_half |
| --- | --- | --- | --- |
| gk (HK=32) | 16 | abs 7.213e-3 / rel 4.01e-4 | **0** |
| gk (HK=32) | 8 | abs 8.029e-3 / rel 4.47e-4 | **0** |
| gk (HK=16) | 16 | abs 7.153e-3 / rel 3.70e-4 | **0** |
| gk (HK=16) | 8 | abs 7.389e-3 / rel 3.83e-4 | **0** |

### 精度策略校准结论

模板的 float32 初始值是 `atol=1.53e-5 / rtol=9.77e-4 / max_abs_limit=0.01`，按上表：

* `max_abs_limit=0.01` **过紧**：h 半边的内在散布已达 8.0e-3，加上余量必然超限。
* `rtol=9.77e-4` 偏紧：h 半边同口径 rel 已到 4.5e-4，余量不足。
* `atol=1.53e-5` **严重过紧**：h 半边的绝对噪声底约 8e-3，是它的 500 倍。

据此把本工程的 float32 策略校准为 `atol=1.5e-2 / rtol=2e-3 / max_abs_limit=0.05`：

* `atol` 覆盖（b）表实测最坏值 8.03e-3 的约 1.9 倍；
* `rtol` 覆盖实测最坏值 4.47e-4 的约 4.5 倍；
* `max_abs_limit` 取 0.05，落在"接受带 ~1e-2"与"违反契约 3.0e-2~4.6e-2"之间，
  作为离群元素的门限；绝大多数判定由 `matched_ratio` 完成。

`global_matched_ratio` 保持 0.999。本算子没有结构性零区或单位区，不设 `critical_*` 分区，
不启用严格 ULP。

### GPU 标杆对齐（2026-09-20）

入口：`scripts/compare_with_gpu.py`（H20 侧用 `benchmarks/cp/bench_pre_process_h20.py
--precision ieee --save-io` 导出 `case.pt`）。

结果（`model-gk`，T=11264，HK=HV=32）：`m_half` **matched 1.000000 / max_abs 9.164e-07**，
`h_half` matched 0.900463 / max_abs 9.371e-03，整体判 FAIL（当时的策略是 atol=1e-4）。

**诊断：h 半边的 9.371e-3 不是契约缺失，而是本条目的契约点漏了。** 三项证据：

1. `m_half` 匹配到 9.2e-07（rel 1.5e-5），说明 m 的公式、正负号、FP32 链式累加全部正确；
2. 单独把标杆的累加器从 FP64 换成 FP32 产生 8.735e-03 —— 与实测的 9.371e-03 同量级；
3. 即使标杆也用 FP32，累加顺序不同仍会产生 7.2e-3~8.0e-3（上表 (b)）——
   这是 `bf16(h)` 量化不连续在反馈环里放大任何微小差异的**内在敏感性**。

所以处理是：把"累加精度 FP32"补进契约（标杆默认值已改），同时把 `atol` 放开到 1.5e-2
以容纳内在散布。按新策略，这一份实测数据即可通过。

### 已知限制：h 半边是弱检查，m 半边是强检查

由内在敏感性决定，h 半边的判定阈值必然偏松（atol 1.5e-2，相对 |h|≈18 只有 8e-4）。
具体地说，"未做 `bf16(h)` 转换"这种错误只产生 1.05e-2 的偏差，落在接受带附近，
单靠 h 半边无法可靠区分。

对策（在 05 阶段落实）：

1. **以 m 半边作为主证据**：它对累加顺序完全免疫（差异恒为 0），阈值可以收到
   `atol=1e-6`；`compare_with_gpu.py` 已把它作为"严格复核"单独打印。
2. **补短窗口用例**：`NT` 小（1~4 个 chunk）时反馈放大尚未累积，`bf16` 转换点
   的影响可以干净地暴露出来，用同一份策略也能形成有效约束。
3. 接口/结构项（有限性、`hm` 形状与 dtype、支持范围拒绝行为）单独用精确检查覆盖。

### 性能基线（半边拆解，H20 参考）

`--half h` / `--half m` 分别只跑两半，`model-gk T=11264 HK=HV=32 K=V=128`（event p50）：

| precision | h (µs) | m (µs) | h+m | both | both/max | both/sum |
| --- | --- | --- | --- | --- | --- | --- |
| default (tf32) | 388.1 | 1058.1 | 1446.2 | 1620.7 | 1.53 | 1.12 |
| tf32x3 | 392.8 | 2564.4 | 2957.2 | 3404.7 | 1.33 | 1.15 |
| ieee | 401.7 | 16465.2 | 16866.9 | 18321.6 | 1.11 | 1.09 |

三条结论（作为 03 的设计输入，记录在 `scripts/estimate_ascend_design.py` 第 [7] 节）：

1. **`both ≈ h + m`，不是 `max(h,m)` —— 两半并没有并行。** 128 个 program 铺在 78 个 SM 上
   只有 1.64 波；昇腾是 28 AIC，128 个 program 要 4.57 波，更不可能并行。**"两条独立链可以
   并行"这个前提不成立**，设计要按"两半争同一批核、串行推进"来算。
2. **m 半边是瓶颈**：default 下占 65%，ieee 下占 90%。m 的 MAC 只是 h 的 2.00×，
   耗时却是 2.73×（default）。
3. **h 半边对 precision 几乎不敏感（+3.5%）**：它的两个 dot 输入都是 BF16（`h`、`v_new`
   都降到 bf16），`AFFINE_CHAIN_PRECISION` 只作用在 `M_c @ m`。m 半边跨档位差 **41×**，
   全部来自那一个 FP32 乘。→ **h 半边在昇腾上没有 FP32 计算压力；`M_c @ m` 走哪条路
   才是性能的决胜点。**

注意 `--half m` 通过把 V 置 0 实现，编译器会把 h 分支判成死代码，所以 m 的实测值偏乐观；
`both > h+m` 的 9~15% 差值主要来自这里。

### 运行证据

1. 标杆自检（`python reference/reference.py`）：形状 `(4, 128, 256)`、`finite=True`。
2. 值域校准：6 个变体全部跑通，覆盖 `HK=HV` 与 `HK=HV/2`、`g`/`gk`/`dplr`。
3. GPU 对齐：`m_half` 与 H20 `ieee` 实测一致到 9.2e-07。

### 踩过的坑（供后续复用）

* 采集脚本最初用独立随机 `w` 造输入，`kᵀw` 谱范数远大于 1，递推在 20 多个 chunk 内
  发散成 NaN。输入必须按 delta 规则构造（`k` 单位化、`w = beta·k`）。
* 校准脚本里曾把"衰减 h"写在算 `v_decay` 之前，与 kernel 相反，量出来的是算法差异
  而不是累加顺序差异（5.3e-2 vs 正确的 7.2e-3）。**校准脚本自身也需要交叉验证。**

### 待补证据

`golden_contract` 目前仍为 `provisional`。标杆默认值刚由 FP64 改为 FP32，
需要 H20 侧按新标杆重跑一次对齐，确认残差落在新策略内后再冻结：

```bash
# H20 侧
git fetch origin && git checkout bench/cp-pre-process-h20
python -m benchmarks.cp.bench_pre_process_h20 --case model-gk --precision ieee \
    --warmup 5 --repeat 10 --save-io ./case_gk
python benchmarks/cp/pre_process_h20/compare_with_gpu.py \
    --case ./case_gk/case.pt \
    --policy benchmarks/cp/pre_process_h20/precision-policy.json --precision ieee
```

## CP 语义对齐实测（2026-09-22，H20）

要把两件事从推断变成实测：（1）2.1.4 首选方案"吃满多序列"在竞品侧有没有现成依据；
（2）竞品 part 级 wrap 在"一个 part 含多条序列、切点落在序列内部"时对不对。
脚本 `benchmarks/cp/check_cp_alignment.py`（fork `bench/cp-pre-process-h20`，§15 有用法）。

### [A] hm 级：多段一次算 == 逐段算（已通过）

`python -m benchmarks.cp.check_cp_alignment --mode hm`，`K=V=128`、`BT=64`、
`AFFINE_CHAIN_PRECISION="ieee"`、`HK=HV=4`（最后一个 case `HK=2,HV=4`）：

| case | A1 逐段 vs CPU | A2 多段 vs CPU | A3 多段 vs 逐段 |
| --- | --- | --- | --- |
| aligned（`0,64,128,256,512`，含 64 行满 chunk） | max_abs=2.409e-04 matched=0.999990 | 2.409e-04 / 0.999990 | **0.000e+00 / 1.000000** |
| 多段/非 64 倍数（`0,96,256,320,512`） | 1.907e-06 / 1.000000 | 1.907e-06 / 1.000000 | **0.000e+00 / 1.000000** |
| GVA `HK:HV=1:2`（同上段边界） | 1.907e-06 / 1.000000 | 1.907e-06 / 1.000000 | **0.000e+00 / 1.000000** |

结论：

* **A3 逐位相等（`0.000e+00`）** ⇒ kernel 的 `MULTI_SEQS`（grid dim2 承载段号 +
  `hm` 的段前导维）与逐段调用**完全等价**。我们接口"一次调用处理多段"的语义
  在竞品侧有现成依据，不需要新造。
* A1/A2 的残差即"同一窗口 vs 基准"的 hm 级残差：只有 aligned 那一路到 `2.4e-4`
  （它的段边界是 64 的倍数，尾块整行参与），其余两路 `1.9e-06`；都远在
  `precision-policy.json` 的 `max_abs_limit=0.05` 之内。
* GVA（`HK<HV`，k 的 head 按 `i_h // (HV//HK)` 展开）在 hm 级成立。

### [B] CP 级：竞品 part 级 wrap

脚本侧修掉 3 个 bug + 1 个"对照组陷阱"，前三次失败都由此而来：

| # | 现象 | 原因 | 修法（commit） |
| --- | --- | --- | --- |
| 1 | contiguous 下 `NoneType * int` | `FLACPContext.part_len` **只在 zigzag 分支赋值** | 脚本按 `T // num_parts` 自算（`66710a7`） |
| 2 | `expected 6, got 5` | wrapper 要 `[B,T,H,D]`，脚本给的是 3 维 | gather 后补 `B=1`（`66710a7`） |
| 3 | `Pointer argument (at 8) ... (cpu tensor?)` | `build_cp_context` 从传入的 `cu_seqlens` **推导设备**（`local.to(device=cu_seqlens.device)`），给 CPU 张量就把 `context.cu_seqlens` 留在主机 | 传 device 张量 + 断言；**索引 8 正是 `cu_seqlens`**（`0e0870b`） |
| 4 | `--preset aligned` "全 0 通过" | 该 preset 两个 rank 都 `is_first_rank & is_last_rank`：**kernel 与 merge 都不发**，等于空跑；它只能当"没有多余状态"的对照 | header 回显 `is_first_rank/is_last_rank`、`ctx_cu_seqlens.device`；新增 `cut` preset（`0e0870b`） |

**方法论**：脚本里的"对照组"必须核对它真的执行了被测代码路径，否则空跑会被读成通过。
第 4 条就是这个教训——`aligned` 过没过，与被测语义无关。

#### 实测结果（2026-09-22，H20，world=2，`--cp-precision tf32x3`）

每个"需要携带状态"的段（= 起点不是任何序列起点的 local segment）三路对照：

| preset / layout | 需要携带状态的段 | CP 路径 `\|init - 单调用ieee\|` |
| --- | --- | --- |
| aligned / contiguous | 无（两个 rank 都 `is_first & is_last`，不发 kernel） | 期望为 0 的段上 `\|init\|max=0.000e+00` |
| multi / contiguous | rank1 `seg#0` 全局[512,600) ← 从 40 扫到 512（472 tokens） | **0.000e+00**（vs CPU 标杆 2.343e-03） |
| multi / zigzag | rank1 `seg#0` 全局[256,512) ← 从 40 扫到 256（216 tokens） | **0.000e+00** |
| 同上 | rank1 `seg#1` 全局[512,600) ← 从 40 扫到 512（472 tokens） | 1.189e-01（rel 1.33e-02） |
| 同上 | rank0 `seg#2` 全局[768,1024) ← 从 700 扫到 768（68 tokens） | **0.000e+00** |
| 其余段 | 起点即序列起点 | 期望 0，实测 `\|init\|max=0.000e+00` |

**逐位为 0 的三段就是决定性的语义证据**：`multi/contiguous` 里 rank1 要的前缀 [40,512)
恰好是 rank0 窗口（[0,512)）的**末段**——如果竞品导出的是整个窗口的链（把 seq0 的
[0,40) 也算进去），差值是 O(|h|)，不可能是 0。zigzag 下 `seg#0`（前缀 [40,256)，只需
rank0 front 的末段）与 `rank0 seg#2`（前缀 [700,768)，只需 rank1 back 的末段）同样逐位为 0。
⇒ "每 part 导出末段 + 前缀复合"在多段 part、zigzag 两种情况下都**取对了段**。

唯一非零的 `seg#1`，前缀 [40,512) 跨两个 part（拆点 256 距前缀起点 216 tokens，不是 64 的倍数），
分块口径与"一次算完"不同。用 `work/cp_decomp_probe.py`（纯 CPU，同输入同 seed，跑
`reference.py` 的契约版）单独量这个效应：

| 对照（前缀 [40,512)，H20 实测同源输入） | max_abs | rel(`\|h\|max`) |
| --- | --- | --- |
| A 单窗口 vs P 纯数学（fp64+关三个舍入开关） | 1.344e-02 | 1.504e-03 |
| B1 两段 [40,256)+[256,512)（拆点非 64 倍数）vs P | 1.191e-01 | 1.333e-02 |
| B2 两段 [40,360)+[360,512)（拆点 64 的倍数）vs P | 1.200e-02 | 1.342e-03 |
| **B1 vs A（拆点 256，非 64 倍数）** | **1.189e-01** | 1.329e-02 |
| B2 vs A（拆点 360，64 的倍数） | 1.261e-02 | 1.411e-03 |

**B1 vs A = 1.189e-01 与 H20 上 `seg#1` 的 1.189e-01 一致到三位有效数字**
（rel 1.329e-02 vs 1.33e-02）。⇒ `seg#1` 的差异是"同一实现、不同分块口径"落在不同
bf16 量化点上的结果，**不是取错状态**。`m` 链在拆分后只差 1.6e-2（m 量级 ~1），
h 的差主要来自契约第 2 点（h 进 `w@h` 前先降到 bf16）：同一段前缀，只要换一种窗口
分解，量化点就变，结果就漂这个量级。

**结论（可直接对外说）**

1. **竞品 part 级 wrap 的语义是对的**：多段 part、切点落在序列内部、contiguous 与 zigzag
   两种布局下，三个"结构性"证据逐位为 0（末段选择正确），唯一非零项被独立探针定量复现
   为分块舍入（1.19e-01 ↔ 1.19e-01）。
2. **对接口的含义**：我们"一次调用处理多段（多序列）"的语义有竞品侧依据（[A] 的 A3 逐位相等），
   而"每 part 只喂末段"这种做法本身不引入语义错误，只引入 ~1e-3 相对量级的**分块差异**。
   推论：**精度验收必须固定分块口径**（窗口起点决定 chunk 边界），否则同一算子的参考值
   会在 1e-3~1e-2 相对量级上漂移——这正是 02 的 `precision-policy.json` 要固化的东西。
3. **对测量的含义**：拿"换精度档位"（tf32x3 vs ieee）当噪声基准不可靠——窗口短时它正好是 0
   （tf32x3 接近 fp32）。真正该用的基准是"同精度、换窗口分解"，已作为噪声基准 B 写进脚本
   （`131947e`）。`--preset cut`（一条序列被切开）尚未跑，但它与 `multi` 的同构性已由
   `multi/contiguous` 覆盖。

#### 补记：CP 下 `B ≡ 1`，而竞品有**两种**调用形态（2026-09-22 复查源码）

用户问"GDN 模型场景是非 CP 的输入，用 CP 切分后是不是就不会有 `B>1`"。复查 fla-org 源码，
答案是对，而且这牵出竞品的第二条调用路径：

**1. CP 恒为 `B = 1`（三重证据）**

> **补充核对（2026-09-23）：竞品 wrapper 里对 `B` 没有任何显式处理，也没有 `assert B == 1`。**
> 通读 `chunk_gated_delta_rule_fwd_h_pre_process`（`cp/chunk_delta_h.py:762-950`），`B` 只出现两次：
> `:789` 解包 shape、`:795` `N = B`（仅 `cu_seqlens is None` 分支用来分配 `initial_state` 第一维）；
> 函数内的断言只有 `:780 initial_state is None`、`:798 K <= 256`、`:880` graph 模式那条。
> 也就是说 **`B == 1` 是"契约式期望"，不是代码强校验**，三条独立证据说明 B>1 走不通：
> ① `cu_seqlens=None`（真·定长）→ 两条分支的 `cu_seqlens[-2:]` / `cu_seqlens[fns-1:fns+1]` 直接抛
> `TypeError`；② kernel 寻址 `k += ((bos*H + i_h//R)*K)` 按**单条连续 token 轴**算，B>1 时偏移
> 错位，且 `hm` 没有 batch 维（`:68-74` 只有段号与 head 两个前导偏移）；③ merge 只写
> `initial_state[0]`（zigzag 再加 `[fns]`，`:848`/`:921`）——"一个窗口只有第一段可能需要 carry"
> 这个前提在 dense B>1 下不成立（每条序列都要 carry，它写不出来）。
> 结论：**竞品 CP 路径实际只支持 `B = 1`；我们对 `B != 1` 做显式拒绝，是把这条隐含前提显式化。**
>
> **关于 `N = B` 这行（`k.new_zeros(N, HV, K, V)`）**：它确实只有在 `cu_seqlens is None`
> 分支里才会取到 `N = B`，但要注意两点：① 这里的 `N` 是 **`initial_state`（序列数）的第一维**，
> 不是 `hm` 的维（`hm` 只有"段号/head"两个前导偏移）；② 这条分支**跑不完**——分配之后紧接着
> `cu_last = cu_seqlens[-2:]`（contiguous，`:878`）就抛 `TypeError`，zigzag 同理在 part 循环里崩；
> 即使绕过去，merge 也只写 `initial_state[0]`（zigzag 加 `[fns]`），dense 下 index `1..B-1`
> 恒为 0（语义错）。所以 `N = B` 是**唯一一处"照顾定长"的痕迹，且属死代码**（产品路径不可达），
> 不能作为"支持定长"的证据。

| 证据 | 内容 |
| --- | --- |
| `fla/ops/cp/README.md` | "CP expects `B == 1` for varlen and uses rank-local `cu_seqlens` from context"；局部输入是 `[1, T_local, D]` |
| `tests/context_parallel/test_cp_gdn.py:131` | 测试里 `B = 1`，序列多段靠 `lengths` → 打包 `cu_seqlens` 表达 |
| `chunk_delta_h.py` 结构 | 跨卡分支 `cu_last = cu_seqlens[-2:]`（`cu_seqlens=None` 直接崩）、merge 只写 `initial_state[0]` 且 `NUM_SEQ_ENTRIES=0`；`compress_h0` 的注释也说"只有局部 batch 的第一条序列可能是上一张卡续下来的，因此只有一个初始状态非零" |

⇒ GDN 表 C1~C16 的 `B=8~128` 是**非 CP** 口径；而这个 kernel 在非 CP 下
`chunk_gated_delta_rule_fwd_h_pre_process` 直接 `return initial_state`，根本不执行。
**CP 下"多条序列"换了表达：打包轴上的段数 `Nseq = len(cu_seqlens)-1`。**

**2. 竞品自己就有"一次调用算 N 段"的形态——卡内 CP（推理）**

`fla/ops/common/backends/intracard.py`："Accelerates **prefill** by splitting long sequences
into sub-sequences and processing them in parallel across SMs. Only active under
`torch.inference_mode()` with varlen."；`MAX_SUBSEQS = FLA_INTRACARD_MAX_SPLITS`（默认 32，
注释："Limits merge chain depth to control precision loss"）。

`fla/ops/common/intracard_cp.py::intracard_pre_scan` 的启动形态与我们提议的接口**完全同构**：

```python
hm = kg.new_empty(S_split, HV, K, V + K, dtype=torch.float32)          # 前导维 = 段数
grid = (triton.cdiv(V, BS) + triton.cdiv(K, BS), HV, S_split)          # 段号进 grid dim2
pre_process_fwd_kernel_merged[grid](..., MULTI_SEQS=True)
```

而且它有明确的动机（`compute_subseq_len` 的 docstring）："For linear recurrence (fwd_h),
the sequential scan is the bottleneck. **Splitting always reduces the critical path and helps**"，
段数按 `target_splits = max(4, num_sms // (NUM_V_BLOCKS * num_heads))` 反推（下限 4）。

**3. 两种形态对照（我们接口两种都能表达）**

| 形态 | 竞品入口 | 一次算几段 | 竞品 `hm` | `MULTI_SEQS` | 每段链是否都要用 |
| --- | --- | --- | --- | --- | --- |
| 跨卡 CP（长上下文） | `chunk_gated_delta_rule_fwd_h_pre_process` | 1（每 part 的末段） | `[HV,K,V+K]` | False | 只用到被切开那条 |
| 卡内 CP（推理 prefill） | `intracard_pre_scan` | `S_split`（≥4，≤32） | `[S_split,HV,K,V+K]` | True | **全部要用**（`intracard_merge` 给每个非首段算初始状态） |

**4. 修正上一轮的一句判断**：我们说"Nseq 只能补占用率、不能缩短关键路径"——对**不同序列并行**
成立，但对**把同一条长序列切成 S 段并行再复合**不成立：切段后每个 program 的串行长度变成
`NT/S`，关键路径真的被砍到约 `1/S`（竞品注释原话），代价是 pre_scan + merge 的额外开销与
复合链更深带来的精度损失（所以他们把段数限在 32，与本文上面测到的"分块/复合带来 1e-3~1e-2
相对漂移"是同一件事）。**序列长、HV 小时，切段是唯一能缩短关键路径的手段**；列拆分只能降低
每步活量，两者可叠加。

#### 判据：本算子什么时候"有活干"（空转 vs 真负载）

状态是逐序列递推的，序列之间互不影响，所以：

| 判断（框架侧，等价于竞品的 `is_first_rank` / `is_last_rank`） | 后果 |
| --- | --- |
| 窗口起点**落在序列内部**（该序列来自左边） | 需要一个非零初始状态 → **要 merge**（= 有人用别人的链） |
| 窗口终点**落在序列内部**（该序列往右延伸） | 要把本窗口末段的 `(h,m)` 交出去 → **要发 kernel** |
| 两者都不是（窗口两端都压在序列边界上） | 前后都没有同序列的 token → 输出恒 0，**调都不用调** |

**"是否有序列跨过 rank 边界"是唯一判据**，与"窗口里有几条序列"无关（窗口里的完整序列本来
就从零状态开始）。三种典型配置：

| 配置 | 边界落在序列内部？ | 结果 |
| --- | --- | --- |
| 等长定长 + `W` 能整除 `B`（`part_len` = 整数条序列） | 否 | **全空转**（实测 `--preset aligned`：两 rank 都 `is_first & is_last`，全 0） |
| 一条长序列被切（`B=1, T_total > part_len`） | 是（每个内部 rank） | 真负载：rank0 只导出、中间既导入又导出、最后一个只导入 |
| 打包但切点落进序列内部（`W` 不整除 `B` / 长度不一） | 是 | 真负载（实测 `--preset multi`：唯一非零项就出现在这里） |

**所以"空转"不是"切分不完整才好用"的反面，而是 CP 退化到"每条序列都自包含"的特例**：
只要存在跨卡序列（长上下文训练常态、varlen 打包常态），本算子就一直在干活。与之并列的
第二类负载是**卡内切段**（intracard，推理 prefill）：人为把长序列切成 S 段并行，所有段的链
都要用，关键路径 ÷S。两类负载的统一判据都是"**存在接续关系**"。

## 03 复核闭环（2026-09-22）

对另一 agent 的 `design.md` review 逐条核对（清单与逐条证据见
`docs/design_review_checklist.md`），并落地用户拍板的三条：

| 决定 | 结论 | 落地位置 |
| --- | --- | --- |
| `M_c @ m` 精度口径 | **FP32 原生**（两侧 FP32、累加 FP32），不做 BF16 三分拆；回退候选顺序 单遍 HF32 → BF16 三分拆 | `design.md` 2.8 / 4.2.3、`api.md` 修订记录 |
| DPLR 范围 | **注册 6 个 `TilingKey`、本轮只验收 4 个**（`USE_G`/`USE_GK` × BF16/FP32） | `design.md` 1.1 / 3.2.1 / 4.3 |
| `hm` 前导维 | **链条数 `Nseq`**（定长 `= B`；变长 `= len(cu_seqlens)-1`） | `design.md` 1.1 / 1.4 / 5.2、`api.md` 3.4 |
| UB/L1 口径 | **硬上限**取平台实测（L1 512 / UB 248 KiB）；**设计预算**（448 / 224）是自定预留，**实际占用按目标芯片 + 目标 CANN 版本的组件实测重算** | `design.md` 3.1 / 3.5、R08 |
| 序列表达（2026-09-22 追加） | **收掉 `B > 1`**：只保留 varlen 打包窗口（`B ≡ 1`、`cu_seqlens` 必给、`Nseq = len(cu_seqlens)-1`）；等长 batch 由调用方打包成等长多段。`hm` 前导维**保留**（= 链条数）：竞品"跨卡每一次调用"是 `[HV,K,V+K]`（无前导维、`MULTI_SEQS=False`），"卡内切段"是 `[S_split,HV,K,V+K]`（`MULTI_SEQS=True`），我们保留的是**两者并集**，`Nseq=1` 时与跨卡形态逐字节相同 | `api.md` 3.2/3.4/5/6、`design.md` 1.1/1.2/2.1.4/5.2、用例表（`B` 列恒 1、`cu_seqlens` 列全部给出、`NT/尾块` 按最长段算、新增异常用例 E14「B≠1」） |
| **子区间窗口（2026-09-23 定稿）** | **支持"窗口是张量 T 轴的子区间"**：`cu_seqlens=[bos,eos]`，`bos>0`、`eos<T` 均合法 ⇒ **调用形态与竞品 1:1**（竞品每次就是"整根张量 + `cu_seqlens[-2:]`/`[fns-1:fns+1]`"），零拷贝零浪费。实现：host 段基址按 `bos` 相对张量起点 + tiling 带 `T_full`，内核地址表达式不变形（`design.md` 1.4.2）；校验放宽为 `0 ≤ cu[0] < cu[-1] ≤ T`（仍拒绝 `B≠1`/零长段/非递增/越界） | `api.md` 修订记录 + 3.2/3.4/3.5/3.6/6、`design.md` 1.4.1(C1)/1.4.2/2.1.4/4.3/5.2（第 9、12 项关闭）、用例表新增「精度-子区间」2 条（张量 T=512、窗口 `[40,512)` 与 `[444,512)`） |

同批修掉的文档缺陷（编号沿用 review）：A2（`m_free`/`M_c_free`/`m_ready` 三套名字统一为
"核内 HardEvent + `M_c_ready` 单槽 ping-pong"）、A3 的两条改进（`c==0` 不得读 `P` 区、
`P_ready` 空发布升为协议条款）、B1（AIV 分工理由改为"整列段 UB 峰值≈264 KiB > 248 KiB 硬上限
+ S1/S3 瓶颈段 + 列天然可分"）、B2（`W` 一号两义）、B3（`L_c` 挪到 S1、补 `FP32(x)` 记法、
`K/V` 符号改 `=128`）、C1–C10（24 个 TilingKey、"K 行 64 分块"与 `kb`/`nb` 循环、"4 或 8 次"、
6 处"第二步"、悬空引用 10.1→5.1、`vec_ready` 复用条件、§2.2 同步缺 `c==0`）、D2/D3、E1/E2。

**新查出的两条**（review 未覆盖）：

1. **N1（严重）**：`design.md` §2.2 的 `Stage0AIC` 伪代码缺 `c==0` 分支且无条件 `wait H_ready`
   ——首 chunk 之前没有 S3 发布 `H_ready`，会**死锁**；已与 §2.7.3 统一。
2. **N2**：用例表异常用例 `E07` 与 `design.md` 4.3 的非法输入列表仍把"变长 `N≠1`"当非法
   （01 收窄的残留，与 `api.md` 3.2 和用例表自身的"精度-变长"组矛盾）；已改为
   "变长 `cu_seqlens` 首元素非 0 / 非严格递增 / 末元素 > `T_win` / 含零长段"，并同步
   `api.md` §6、重新生成用例表（39 正向 + 13 异常，公式错误 0）。

**新增交付**：`api.md` §3.5「Python 调用示例（落地形态）」——含注册/导入方式
（`import fla_npu` + `from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged`）、
完整签名，以及定长+GVA / 变长打包 / CP 两种形态（单段对标、整窗多段吃并行度）三个可执行示例。

## 接入层实现（ctypes 路由，2026-09-28）

用户指定**接入走 ctypes**（`_aclnn_ctypes.py` + `_runtime.py` 的两段式 aclnn 调用），已在本地仓
副本 `C:\Code\Acsend\0907\flash-linear-attention-npu` 落地并通过**离线单测**（无需 NPU 设备）：

| # | 文件 | 改动 | 证据 |
| --- | --- | --- | --- |
| ① | `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` | 加 `_GET_WORKSPACE_ARGTYPES["aclnnPreProcessFwdKernelMerged"]`（8 个 descriptor + `int64` + `hmOut` + `workspaceSize*` + `executor*`）与 `npu_pre_process_fwd_kernel_merged(...)` 包装：契约校验 → `_zeros` 预分配 `hm[Nseq,HV,K,V+K]` fp32 → `_call_aclnn` 两段式调用 | `python -m unittest test_pre_process_fwd_kernel_merged` → **Ran 10 tests … OK** |
| ② | `.../ops/ascendc/__init__.py` | `_ASCENDC_OPS` 加 `"npu_pre_process_fwd_kernel_merged"`（自动导出带/不带 `npu_` 前缀两个名字） | 同上一次运行中的 `test_registered_in_ascendc_ops` 通过 |
| ③ | `torch_custom/fla_npu/test/test_pre_process_fwd_kernel_merged.py` | 新增 10 条离线单测：子区间 `[40,512]`→`hm[1,…]`、多段 `[0,88,188,512]`→`hm[3,…]`、`B≠1`/缺 `cu_seqlens`/门控冲突/`bg` 不配套/越界拒绝、`ARGTYPES` 与签名核对 | 同上；并回归 `test_aclnn_ctypes_abi` → **Ran 8 tests … OK** |

交付副本：`work/impl/ctypes_wrapper_snippet.py`（①的代码片段）、
`work/impl/test_pre_process_fwd_kernel_merged.py`（③的完整单测）、
`outputs/pre_process_fwd_kernel_merged_api.md` §3.1.1（落地记录）。

**§5.2 第 7 项（`hm` 返回值 vs 可选 `hm_out`）随之关闭**：ctypes 路由下 `hm` 由 Python 侧
`_zeros` 预分配、作为输出张量传入 aclnn 再返回——既与竞品"调用方预分配 `hm` buffer"一致
（`k.new_zeros(HV,K,V+K)`），也与仓内其它 ctypes 算子（`h_out/v_new_out`）一致，无需新增参数。

## 实现里程碑：v1 算子**编译通过**（910b + 950，2026-09-28）

算子工程已落到仓里：`fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged/`

```
CMakeLists.txt                     # 目录 glob
op_host/CMakeLists.txt             # add_op_to_compiled_list + add_modules_sources + ascend950 编译选项
op_host/pre_process_fwd_kernel_merged_def.cpp        # OpDef：k/w/u BF16、g/gk FP32、cu_seqlens INT64、hm FP32
op_host/pre_process_fwd_kernel_merged_tiling.{h,cpp} # 契约校验 + tiling 填充 + TilingKey(1..3) + workspace
op_host/op_api/{pre_process_fwd_kernel_merged,aclnn_pre_process_fwd_kernel_merged}.{h,cpp}
                                                    # l0op 内部入口 + aclnn 两段式（与 ctypes ARGTYPES 逐参对齐）
op_kernel/pre_process_fwd_kernel_merged_struct.h    # host/device 共用 tiling 结构 + 每核 GM 临时区大小
op_kernel/pre_process_fwd_kernel_merged.cpp         # 入口 + v1（AIV-only）向量实现，按 design 1.3 的 S0–S4
```

**编译证据**（容器 `admin123-gdn-work`，CANN 9.1.0，`build.sh --ops=` 白名单只编本算子）：

```bash
bash build.sh --opkernel --soc=ascend910b --ops=pre_process_fwd_kernel_merged   # EXIT=0（Built target pre_process_fwd_kernel_merged）
bash build.sh --opkernel --soc=ascend950  --ops=pre_process_fwd_kernel_merged   # EXIT=0（100% ops_transformer_kernel）
bash build.sh --ophost   --ops=pre_process_fwd_kernel_merged                    # EXIT=0（Build libs ophost_transformer success）
```

**过程中清掉的编译问题**（备查）：

1. `tiling.h` 的相对 include 写成了 `../../op_kernel/...`（那是 `op_host/op_tiling/` 布局才成立），
   本算子 `op_host/` 下应为 `../op_kernel/...`；
2. `Muls(dst, src, scalar, count)` 要求 dst/src **同 dtype**：bf16 行必须先整行 `Cast` 到 fp32 再算；
3. UB→GM 的 `DataCopy` 需要 `GlobalTensor`（不能传裸 `__gm__ T*`）；
4. AscendC 没有标量 `exp`：改用 8 lane 向量 `Exp` + `SetValue/GetValue` 做标量 exp2（`exp2(x)=exp(x·ln2)`）；
5. **bisheng 后端不支持标量 bf16↔fp32 转换**（`fatal error: not support bf16 type cast`）：
   全部改成"逐行向量 Cast"——`k/w/v` 行整行 Cast 成 fp32 后读标量；`USE_G` 的 `L` 行走
   "`Muls(dg)` → `Cast` 到 bf16（契约量化点）→ `Cast` 回 fp32"三步向量化；
6. OpDef 的 gate dtype 统一成 FP32，BF16 gate 由 aclnn 层 `l0op::Cast` 先转 FP32（与
   `chunk_gated_delta_rule_fwd` 的做法一致），避免设备侧 bf16 标量转换。

**v1 已知限制**（不影响编译，后续迭代解决）：

* `USE_BG`（DPLR）分支未实现（接口/TilingKey/用例均已预留）；
* 4 个矩阵乘是 UB 向量 FMA 手写，**性能远低于 1.0x H20**；按 `design.md` 2.2–2.6 换 Cube/CATLASS tile 是性能版本任务；
* **尚未上板对拍精度**：本容器当前 `npu-smi info` 报 `dcmi model initialized failed ... device is used`
  （设备被其它容器占用），且容器内 `python3.12` 未装 torch/torch_npu。

---

## 12. 真机联调（2026-09-28，246 / Ascend950PR）

### 12.1 环境（新增，不动原工作容器）

| 项 | 值 |
| --- | --- |
`admin123-gdn-test`（新起，`--privileged`，`ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`，同一镜像 `admin123-gdn-snap:20260919`，`-v /data/admin123:/workspace`）
`torch 2.7.1+cpu` + `torch_npu 2.7.1.post8`（cp312 wheel 来自 `/home/t00985010/pkg/`），`torch.npu.device_count() == 4`

> 原工作容器 `admin123-gdn-work` **没有 privileged**，`torch_npu` 报 `Can't get ascend_hal device count`、
> `device_count()==0`，`npu-smi info` 在容器内报 `dcmi model initialized failed ... device is used`；
> 同镜像起一个带 `--privileged` 的容器后设备可见。**编译用原容器、上板用这个新容器**，两者共享 `/workspace`。

### 12.2 上板路径（全部实跑通过）

```bash
# 1) 编 wheel（只编本算子）
FLA_NPU_SOC=ascend950 FLA_NPU_OPS=pre_process_fwd_kernel_merged python3 scripts/build_wheel.py
# 2) 安装（wheel 内嵌 OPP，Python 侧 ctypes 直调 aclnn）
pip install --force-reinstall --no-cache-dir --no-deps dist/flash_linear_attention_npu-*-950.*.whl
# 3) 调用（脚本见 op 目录 scripts/ 与 impl/npu_smoke_ppfm.py）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 npu_smoke_ppfm.py
```

**结论：ctypes 通路（`fla_npu.ops.ascendc.pre_process_fwd_kernel_merged` → `aclnnPreProcessFwdKernelMerged`
→ tiling → kernel 启动 → `hm` 回读）在真机上完整打通**；`hm` 的 shape 恒为 `[Nseq, HV, K, V+K]`，
单段 / 子区间 / 多段 / GVA 四种调用形态都能正常启动并返回。

### 12.3 真机暴露并已修复的问题（编译通过 ≠ 上板正确）

| # | 现象 | 根因 | 修复 |
| --- | --- | --- | --- |
| 1 | `rtsFuncGetByEntry failed / kernel name: PreProcessFwdKernelMerged, tiling key: 1` | kernel 入口只用 `KERNEL_TASK_TYPE_DEFAULT`，二进制里只生成 `_0` 一个入口；host 侧 `SetTilingKey(1..3)` 查不到 | 入口改成 `if (TILING_KEY_IS(1)) { KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY); ... }` 三个分支（对齐 `recompute_w_u_fwd` 的写法） |
| 2 | 首次运行设备错误 `507035` | 单条 TBuf 用了"float 元素偏移"，bf16 缓冲实际按元素数索引造成**缓冲互相覆盖**，且总申请 283 KB > UB 硬上限 248 KB（`ub_size=253952`） | 拆成 9 个按 dtype 精确分配的独立 TBuf，总占用 ≈212 KB |
| 3 | 偶数次/多次调用结果不同（非确定性） | 回归补丁里 `mBuf_` 被 `InitBuffer` 了两次，TPipe 出现重叠分配 | 删除重复分配后**两次调用逐位一致**（`max_abs=0.0`） |
| 4 | 数值大面积错、量级到 1e7 | 标量 `LocalTensor::GetValue` 读 UB 前没有 V→S 同步（在 vector 写之后只放了 `PipeBarrier<PIPE_V>`） | 读前统一改 `PipeBarrier<PIPE_ALL>`；`PrepareLRow` 收尾改 PIPE_ALL |
| 5 | `h` 半边与标杆有 ~1e-2 偏差 | 契约要求 `h` 进 `w @ h` 前降到 BF16，v1 直接用 FP32 | 逐行 `Cast→CAST_RINT→Cast` 回 FP32 后再做 `w @ h` |
| 6 | GDN（`USE_G`）的 `h` 半边随 chunk 数发散（T=256 时偏差 1e7），KDA 正常 | **算法 bug**：标杆里 `h` 项用的是**未加门控的 `k`**（门控只作用在 `v_new` 上：`v_new *= 2^(g_last-g_t)`），`m` 半边才用 `left = k * 2^(g_last-g_t)`；v1 两处都用门控后的 L，等于把门控算了两遍 | `PrepareLRow(t, gateScale)`：S2（dH）传 `false`，S3（Kw）传 `true` |
| 7 | `h` 半边偶发尾段 lane 为 0（且两次调用不一致） | 向量累加（`Muls`+`Add`）之间缺 `PipeBarrier<PIPE_V>`；S0/S1 每轮都有、S2/S3/S4 没有 | 每条向量指令后补 `PipeBarrier<PIPE_V>`，跨 pipe 处用 `PIPE_ALL` |
| 8 | 结果非确定性 | 回归补丁里 `mBuf_` 被 `InitBuffer` 两次 → TPipe 重叠分配 | 删除重复分配 |

### 12.4 精度状态：6/6 全部 PASS（2026-09-28 收口）

`npu_smoke_ppfm.py`（BF16 输入 / FP32 `hm`，`T=256, K=V=128, BT=64`，模型同构数据
`k = normalize(·)`、`w = β·k (β~U(0,0.02))`、gate 为 chunk 内 cumsum 的负对数衰减）：

| 用例 | 形态 | 整体 matched | 整体 max_abs | h 半边 | m 半边 |
| --- | --- | --- | --- | --- | --- |
| `gdn-single` | GDN，单段 `[0,256]` | **1.000000** | 1.402e-04 | 1.000000 / 9.5e-07 | 1.000000 / 1.4e-04 |
| `kda-single` | KDA（gk），单段 | **1.000000** | 3.020e-04 | 1.000000 / 4.8e-07 | 1.000000 / 3.0e-04 |
| `gva-1x2` | GVA，HK=1/HV=2 | **1.000000** | 9.537e-07 | 1.000000 / 9.5e-07 | 1.000000 / 9.5e-07 |
| `gva-2x4-sub` | GVA + 子区间 `[64,192)` | **1.000000** | 5.364e-07 | 1.000000 / 0.0e+00 | 1.000000 / 5.4e-07 |
| `gdn-multiseg` | GDN，多段 `[0,88,188,256]` | **1.000000** | 5.960e-07 | 1.000000 / 4.8e-07 | 1.000000 / 6.0e-07 |
| `kda-multiseg` | KDA，多段 `[0,64,128,256]` | **1.000000** | 4.172e-07 | 1.000000 / 2.4e-07 | 1.000000 / 4.2e-07 |

对拍对象是仓内 CPU 标杆 `reference/reference.py`（本身已与 H20 `ieee` 实测对齐）。
另有 `npu_dbg11_realgate.py` 用**标杆 self_test 同款**数据做单 case 交叉验证：
`T=64/128/256` 下 h 逐位一致（`max_abs=0.000e+00`）、m `matched=1.000000`（`max_abs ≤ 3.6e-04`）。

> **用例数据集的选择有讲究**：若把 `w` 取成满幅随机（`|w|~N(0,1)`），`m = Π M_c` 会把
> fp32 求和顺序的 1 ulp 差异放大到 O(1)（实测 `|m|` 涨到 2.9e7，`matched` 掉到 0.984）。
> `cpu_factorization_check.py` 在 **CPU 上**用两种等价求和顺序重算同一用例，差异同样是
> `matched=0.999939`（仅 1 个元素超限）——证明那是**用例病态**、不是实现缺陷。
> 真实模型（以及 H20 基准 case）都在 `w = β·k` 的良态区间，故以模型同构数据为准。

### 12.4.1 m 半边缺陷的定位与最终修复（排查全过程）

用单 chunk、单 head、`g = 0` 的极小用例 + 内核内多路转储逐个排除：

已用多路转储把范围收敛到 **S4（m 链的矩阵乘）里的累加写入**：

| 观察 | 结果 |
| --- | --- |
| `vNewBf_`（S0/S1 的 `v_new`）转储 vs 数学值 | **逐位一致**（`max_abs=0.000e+00`）→ k/w/v 的 GM→UB 拷贝与 S0/S1 全对 |
| `wUb_` 转储 | **逐位一致** |
| `m` 初值（`InitIdentityM`）转储 | **严格单位阵**（128 个对角元全 1，非对角 0） |
| 进入 m 链**之前**再转储 `mF_` | 仍是**严格单位阵** → 排除"UB 缓冲被后续阶段覆盖" |
| `Kw` 累加块转储 | 列方向无零段 → Kw 累加本身没错 |
| 旧写法 `m = M_c @ m`（逐 j 标量读 + 128 lane 累加）| 列 `0..95` 逐位一致、列 `96..127` 恒为 0 |
| 新写法 `m_new = decay⊙m - Lᵀ(W_c@m)`（T1/T2 两次矩阵乘，全向量）| 仍不对；`T1` 转储显示列 `0..79` **逐位精确**、列 `≥80` 开始错（误差同量级于数据） |
| 把 S4 的 128 lane 累加拆成 64 lane 分段 | **无改善**（排除"128 lane 指令宽度"） |
| 把 `blkBuf_`/`scratchBuf_` 换到 UB 低地址 | **无改善**，换序后直接 `507035`（UB 布局已进敏感区） |
| 在 T1 里先把 64 KiB 大缓冲的行拷进小缓冲再参与 `Muls` | **无改善** |

**结论（最终修复）**：缺陷不在输入、不在 S0/S1/S2、不在 m 的初值、不在 Kw 累加。真因是两个
**同步缺失**，与"lane 宽度""UB 地址区间"都无关：

| # | 真因 | 现象 | 修复 |
| --- | --- | --- | --- |
| ① | 标量 `GetValue` 读 UB 前只有 `PipeBarrier<PIPE_V>`（缺 V→S） | 累加结果里"前若干 lane 对、之后整段错"、边界随写法漂移（93/96/80） | 标量读前统一 `PipeBarrier<PIPE_ALL>` |
| ② | 向量算完写回 GM（MTE3）前只有 `PipeBarrier<PIPE_V>`（缺 V→MTE3） | **只有每个循环的第 0 次迭代错**：m 第 0 行写成上一轮残留（`e_127`） | 写回 GM 前改 `PipeBarrier<PIPE_ALL>` |

此外把 m 从"UB 大缓冲常驻"改为**GM 常驻 + 小行缓冲参与向量运算**（`m`/`T1`/`T2`/`mNext` 四段都在
本核 user workspace，共 224 KiB < 400 KiB 预算），并去掉对 m 的"就地读改写"。三处一起改完后
m 半边一次性收敛（见 12.4 表）。

### 12.5 性能现状与下一步（当前瓶颈）

`npu_bench_ppfm.py` 实测（v1 向量实现，`hnpu:0`，warmup 5 / repeat 20）：

| 用例 | 耗时 |
| --- | --- |
| `T=1024, HK=HV=4, K=V=128, BT=64` | **38.2 ms/call** |
| `T=1024, HK=HV=8` | 38.3 ms/call |
| `T=4096, HK=HV=8` | 152.8 ms/call |

v1 的 4 个矩阵乘是 AIV 标量/向量循环，**实测与 1.0x H20 目标相差约 30–100 倍**（H20 基准：
`model-gk` `T=11264, HK=HV=32` 1620 µs `default` / 18321 µs `ieee`）。

下一步（性能路径，按 `design.md` 2.2–2.6）：

1. 把 `W_c@h`、`L_c^T@v_new`、`W_c@m`、`L_c^T@T1`、`M_c@m` 五个矩阵乘换成 **Cube/CATLASS tile**
   （仓内 `chunk_scaled_dot_kkt`、`chunk_gated_delta_rule_fwd_h`、`chunk_fwd_o` 有现行用法），
   累加交给 MMAD/Fixpipe，AIV 只做 gate/量化等 elementwise；
2. 分核从"1 任务 = 1 段 × 1 head"改成"段 × head 切分 + Cube 行列 tile"，把 28 个 AIC 填满；
3. 保留 v1 作为**精度回归基线**：`scripts/npu_smoke_ppfm.py`（6 形态全 PASS）、
   `scripts/npu_dbg11_realgate.py`（标杆 self_test 同款数据，h 逐位一致）、
   `scripts/compare_with_gpu.py`（H20 标杆对拍）三条链都已可用。

### 12.6 Cube(v2) 内核落地进展（2026-09-28，进行中）

`op_kernel/pre_process_fwd_kernel_merged_v2_cube.cpp.txt` 是 v2（Cube/MIX）实现：
**AIC 跑 4 个 bf16 matmul（`MatmulImpl`，GM 输入输出）+ AIV 做全部 elementwise，
两者用 CrossCore flag 按 chunk 串成流水**；一个工作项 = 一条链（段 × value-head），
状态 h/m 常驻本核 workspace（512 KiB/核，host 侧 `PPFM_CORE_WS_BYTES`）。

已打通并落地（都是真机上踩出来的，写进代码注释）：

| # | 现象 | 真因与修法 |
| --- | --- | --- |
| ① | host 侧 tiling 段错误（进程 core dump，无 plog） | `MatmulApiTiling::GetTiling(optiling::TCubeTiling&)` 重载在 CANN 9.1.0 上崩；**改用 `AscendC::tiling::TCubeTiling`**（仓内其它算子正是这个重载）。tiling 结构体新增两个 `PpFwdCubeTiling` 成员（host/device 同名同布局） |
| ② | `MatmulImpl` 编译报 `no member named 'WaitIterateAll'` | 950 上用 `MatmulImpl::IterateAll<false>(gm,0,false,false)` + `End()`（对齐 `chunk_kkt_cube.h`），没有 `WaitIterateAll` |
| ③ | 首次运行 `aicore timeout`（死锁） | AIV→AIC 的 `CrossCoreSetFlag` 用了 `0x4`；官方/仓内样例两向都用 **`0x2`**（只是 PIPE 不同：AIC 用 `PIPE_FIX`、AIV 用 `PIPE_MTE3`） |
| ④ | UB 缓冲错位、`identF_` 越界 | UB 布局常量是**字节**偏移，取值时却按**元素**索引（bf16 差 2×、fp32 差 4×）；已按 `sizeof(T)` 换算 |

**当前卡点**：v2 能跑完（无超时/无异常），但 `hm` 输出等于初值——诊断显示 **AIV 侧读输入张量（k/w/v/g）全为 0**
（`cu_seqlens` 的 int64 读正常、tiling 字段读正常、UB→GM 写正常）。下一步只需在这条线上继续定位
（怀疑是 AIV 在 MIX 内核里拿到的输入 GM 地址/访问路径问题）。

为保证仓处于**可用状态**，当前在用内核仍是 **v1**（`npu_smoke_ppfm.py` 6/6 PASS，
见 12.4）；v2 源码与上述结论以 `.cpp.txt` 形式留档，host 侧的 cube tiling 已就位（v1 不使用、无副作用）。

### 12.7 v2 第二轮排查：数学全对，卡在 AIC→AIV 的 C 数据可见性（2026-09-28）

给 v2 加了逐级探针（bf16 一律"DataCopy 回读 + 向量 Cast 转 fp32"，因为**标量 bf16↔fp32 转换不可靠**；
`GlobalTensor::SetValue`（GM 标量写）也不生效，探针值必须先落 UB 再用 DataCopy 带出）。单 chunk 实测：

| 环节 | 实测值 | 判定 |
| --- | --- | --- |
| 输入读取（k/w/v/g） | k=-0.0957、w=-0.00061、v=0.26562、g=-0.000297 | ✅ 与 Python 真值一致 |
| staging（k/w/left/v 落 GM） | 与输入一致；left = k·0.98893 | ✅ |
| gate（dg/decay） | dg=0.98893、decay=0.98873 | ✅ |
| matmul③ `T1=W_c@bf16(m)` | t1[0:2] = [-0.00063, -0.00014] ≈ W[0,0:2] | ✅ |
| matmul② `dH=k^T@bf16(v_new)` | dH[0:2] = [-0.58828, 0.80363] | ✅ |
| **AIV 在等待 DH flag 后立刻读 dH** | **0** | ❌ 数据尚未可见 |
| 同一块 dH 在 epilogue（chunk 之后）读 | -0.58828（正确） | ✅ 数据最终会到 |

⇒ **四个 matmul、staging、gate 的数学全部正确**，唯一问题是 AIC 的 C 数据在"设置 flag 的那一刻"对 AIV
还不可见（AIV 随后读到 0，导致 `h` 恒为 0；`m` 的偏差恰好落在 atol 内所以看不出来）。已尝试且**都无效**：

1. `IterateAll<false>` → `<true>`（`sync` 模板参数，本以为这是根因）；
2. 置 flag 前加 `PipeBarrier<PIPE_ALL>()`；
3. 置 flag 前对 C 做 `DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>`。

下一步的可选路线（按把握排序）：

1. **把 C 显式取回**：用 `GetTensorC(...)` 把 C 取到 UB/L1 后再写 GM（`IterateAll` 的 C 可能是
   "延后写回"路径），或在 AIC 侧用 `SetFlag/WaitFlag<HardEvent::FIX_*> ` 显式等 fixpipe；
2. **把状态更新也放进 Cube**（避开跨核传 C）：`h_new = decay⊙h + dH` 可写成一次矩阵乘
   `[diag(decay) | k^T] @ [h ; v_new]`，`m` 同理 `[diag(decay) | left^T] @ [m ; bf16(T1)]`；
3. 或退一步：**混合方案**——`v_new`/`dH`（h 半边）用 Cube，`m` 链继续用已验证正确的 AIV 路径。

### 12.8 v2 第三轮：再加 4 组排除实验，定位到"AIC 的 C 写回时机"（2026-09-28）

本轮又试了 4 种办法，**都不能让 AIV 在消费点读到 AIC 的 C**：

| # | 尝试 | 结果 |
| --- | --- | --- |
| 1 | `IterateAll<false>` → `IterateAll<true>`（`sync` 模板参数） | 仍读到 0 |
| 2 | **把状态更新推迟一个 chunk**（新增 `decayPrevF_` 保存上一 chunk 的 decay；AIV 在 chunk c 的开头更新 chunk c-1 的 h/m，此时 AIC 已完成本 chunk 的 mm1/mm3） | 仍读到 0 |
| 3 | matmul tiling 的 UB 预算设 0（`SetBufferSpace(-1,-1,0,-1)`，怀疑 AIC 的 matmul 与同核 AIV 抢 UB） | 仍读到 0 |
| 4 | 置 flag 前 `PipeBarrier<PIPE_ALL>()` + `DataCacheCleanAndInvalid<..., ENTIRE_DATA_CACHE>`（12.7 已试） | 仍读到 0 |

**唯一有效线索**：同一块 `dHF_` 在**所有 chunk 结束之后**（AIC 已空转）由 AIV 读，值是**正确的**
（-0.58828，与数学值一致）。即 **AIC 的 C 写回要等到 AIC 不再连续发 matmul 之后才可见**，
高度怀疑与 L0C 双缓冲/fixpipe 的延迟写回（或 AIC 与配对 AIV 共享 UB 的冲突）有关。

据此，下一步首选：**不用 `IterateAll` 的隐式 C→GM，而是显式 `GetTensorC`/把 C 落到 L1 再落 GM**，
或在每次 matmul 之后插入一次"dummy matmul/同步"把写回逼出来；再不行就用
`AscendC::SyncAll<false>()` 在 chunk 边界做一次全核同步。

### 12.9 v2 第四轮：reader 侧 DCCI 与显式 waitIterateAll 也无效（2026-09-28）

| # | 尝试 | 结果 |
| --- | --- | --- |
| 5 | AIV 在每个 `CrossCoreWaitFlag` 之后做 **reader 侧失效**：`DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(dHF_/t2F_/vTmpF_/t1F_)`（CANN `matmul_client.h` 里"读跨核 GM flag 前先 DCCI"的同款写法） | 仍读到 0 |
| 6 | `IterateAll<false>(gm, 0, false, /*waitIterateAll=*/true)`（仓内 `chunk_scaled_dot_kkt` 的写法） | 仍读到 0，且出现 1e10 级脏值 → 已回退 |

**同时修正 §12.8 第 2 条实验的结论**：那次"推迟一个 chunk"用的仍是**单份 `dH`/`T2` 缓冲**，
AIC 在下一个 chunk 会立刻覆盖同一块缓冲，所以那次**不构成有效的时序测试**；有效的推迟测试必须先把
`dH`/`T2` **双缓冲**。

至此已排除 8 种手段（`sync`、`waitIterateAll`、`PIPE_ALL`、writer/reader 两侧 DCCI、UB 预算、
推迟读、flag mode、flag 顺序）。剩余首选项（按把握排序）：

1. **`dH`/`T2` 双缓冲**后做有效版"推迟一个 chunk 再读"；
2. **`GetTensorC(...)`/显式 L1 中转**替代 `IterateAll` 的隐式 C→GM；
3. 每个 matmul 后插一次 **16×16×16 dummy matmul**，把上一次的 fixpipe 写回挤出来（用于确认
   "L0C 双缓冲导致延迟写回"这一假设）；
4. 把状态更新改写成**一次矩阵乘**（`[diag(decay) | kᵀ] @ [h ; bf16(v_new)]`），让状态更新留在
   Cube 侧，彻底去掉"AIV 读 AIC 的 C"这条依赖。

### 12.10 v2 第五轮：双缓冲推迟读也无效 —— 现象收敛为"平台级 AIC→AIV 可见性"（2026-09-28）

本轮把 §12.9 第 1 条落地：**`dH`/`T2` 各双缓冲一份**（本核 workspace 512→768 KiB），
AIC 把 chunk c 的结果写进 `(c&1)` 那一份，AIV 在 chunk c 读取 `(c-1)&1` 那一份
（该份在 AIC 这一侧不会再被覆盖），并按奇偶**直接分支**取 buffer（避免 `GlobalTensor` 拷贝传给
`IterateAll` 时地址不生效的嫌疑）。结果：**仍然读不到** —— h 半边 matched=0.0089，
且 head0 出现 4.4e16 量级的**未初始化脏值**，说明 AIV 读的那份 buffer **从未被写入过**。

与之前一致的证据链（累计 9 组排除实验）：

| 观察 | 含义 |
| --- | --- |
| AIV 写 GM → AIC 的 matmul 能读到（4 个 matmul 数值全对） | **AIV→AIC 可见性正常** |
| AIC 的 matmul C 在运行期被 AIV 读：立刻读=0、推迟一整个 chunk 读=脏值 | **AIC→AIV 在运行期不可见** |
| 同一份 buffer 在**所有 chunk 结束之后**（AIC 空转）由 AIV 读 | **正确**（-0.58828） |

⇒ 现象已收敛为：**本平台上 MIX 内核里，Cube(fixpipe) 写出的 GM 数据在 AIC 仍在连续发 matmul 期间
对配对 AIV 的 MTE2 读不可见**；与 flag 顺序、`sync`/`waitIterateAll`、writer/reader 两侧 DCCI、
UB 预算、双缓冲推迟读都无关。

> 过程教训（已写进本节，避免重复踩）：第一次"双缓冲推迟读"构建其实**编译失败**（AIC 类漏声明
> `dHF1_`/`t2F1_`），而当时装上的还是上一轮 v1 的 wheel，跑出的"matched=1.000000"是 **v1 的结果**。
> 之后每次都必须先 `grep -ci error:` 确认构建 0 错误、并核对 wheel 时间戳，再看数值结论。

**下一步（优先级已调整）**：

1. **先做一个最小 MIX 实验**（AIC 单次 matmul 写 GM → flag → AIV 读并回写），确认这是不是
   950/3510 上 MIX 的普遍约束（而不是我们这条流水特有的问题）；
2. 若确认是普遍约束，则改结构**彻底避开"AIV 读 AIC 的 C"**：把 `v_new` 也放到 Cube 侧
   （`v_new = v - W_c@bf16(h)` 可写成一次带 `-1` 的 matmul 加 `v` 的 elementwise…仍需 AIV 参与），
   或改让 **AIV 自己算 `vTmp`**（h 半边用向量、m 半边用 Cube），把跨核数据流限制在"AIV→AIC"这一向；
3. 或退一步：**以 v1（已验证正确）为基础做向量侧优化**，先把"能跑且正确"版交付，性能另立目标。

### 12.11 找到"非惯用法"根因：950 的 AIC→AIV 交接要用 CATLASS `BlockMmad`（2026-09-28）

对照仓内现成实现（`chunk_scaled_dot_kkt`）后确认了差异：

| 路径 | 950（`__CCE_AICORE__ == 310`）| 非 950 |
| --- | --- | --- |
| AIC 算 score | **CATLASS `BlockMmad`**：`blockMmad.preSetFlags()` → 计算 → **`blockMmad.finalWaitFlags()`** → `CrossCoreSetFlag<0x2, PIPE_FIX>` | `matmul::Matmul`（client）+ **`WaitIterateAll()`** |
| AIV 收 score | `Catlass::Arch::CrossCoreWaitFlag(scoreReadyFlag_[slot])` | 同左 |

而 `chunk_kkt_cube.h` 里用 `MatmulImpl` 的那处（正是我抄的写法）**输出只被核外消费**（写进 op 的
输出张量），核内没有"AIC 写 C → AIV 读"的依赖，所以它不需要任何完成等待。

⇒ 结论：**在 950 上做核内 cube→vector 交接时，`MatmulImpl::IterateAll`（含 `sync=true` / `End()` /
writer|reader 两侧 DCCI）都不构成"C 已写回 GM 且对 AIV 可见"的保证**；仓内的惯用法是 CATLASS
`BlockMmad` + `finalWaitFlags()`（显式等 MMAD/fixpipe 队列）。

**下一步（明确、可执行）**：把 v2 的 AIC 侧从 `MatmulImpl` 换成 **CATLASS `BlockMmad`**（照
`chunk_scaled_dot_kkt` 的 950 路径：`Resource<Ascend910_950>` 取 L1/L0 buffer、`TileCopy`+`TileMmad`、
`preSetFlags/finalWaitFlags`、C 经 fixpipe 落 GM），4 个 matmul 的形状固定为
`[64,128]×[128,128]` 与 `[128,64]×[64,128]`，L1/L0 占用都在限额内。

### 12.12 v2 第六轮：AIC 换成 CATLASS `BlockMmad`，**编译通过、m 半边基本正确**（2026-09-28）

已按 §12.11 的结论把 AIC 侧从 `MatmulImpl` 换成 CATLASS `BlockMmad`（`MmBlockNT`/`MmBlockTA` 两套类型：
`preSetFlags()` → `mm(blockA, blockB, blockC, shape)` → `finalWaitFlags()` → CrossCore flag）。落地要点：

* **`CATLASS_ARCH` 必须在包含任何 catlass 头之前 `#define`**（950 → `3510`），否则
  `catlass/gemm/tile/tile_copy.hpp` 会因 `ScaleGranularity`/`CopyToGM` 未定义而报上百个错；
* 还需要 `catlass/arch/cross_core_sync.hpp` 与仓内 `kernel_utils/block/block_mmad_pingpong_tla_multi.hpp`
  （`MmadPingpongTlaMulti` 定义在这里）。

**首次上板结果（T=256, HK=HV=2，模型同构数据）**：

| 半边 | 本轮（CATLASS） | 上一轮（MatmulImpl） |
| --- | --- | --- |
| m | **matched=0.992188，max_abs≈7e-02**（≈ 只有每个 head 的第 0 行错） | matched=0.9905，max_abs≈2.6e+02（垃圾） |
| h | head0=0、head1≈1e17（仍错） | 同样为 0 / 垃圾 |

⇒ CATLASS 路径确实解决了 §12.11 指出的"C 对 AIV 不可见"这一层（m 半边从垃圾变成基本正确、
误差量级从 1e2 降到 7e-02），方向正确。**剩余两个问题**：

1. **matmul②（`dH = kᵀ@bf16(v_new)`）的输出仍不可用**（h 半边为 0 / 脏值）；
2. **每个 head 的第 0 行错**（与 v1 早期"每轮第一条迭代丢写"是同一类现象）。

下一步：用之前那套探针单独校 `vTmp`/`dH`/`T1`/`T2` 四个 C，定位是哪一个 matmul 的
`GemmCoord`/layout/`GetTile` 参数没对上；再处理"第 0 行"这类首迭代效应。

### 12.13 v2 第七轮：探针定位到"AIV 双子核互相覆盖"并已修（h 仍待解）（2026-09-28）

给 CATLASS 版加了逐级探针（bf16 用 DataCopy 回读 + 向量 Cast），单次调用的实测值：

| 观测量 | 值 | 判定 |
| --- | --- | --- |
| `vTmp`（matmul① C） | [0.01672, -0.00102] | ✅ 量级合理 |
| `t1`（matmul③ C） | [0.00094, -0.00166] | ✅ |
| `dH`（matmul② C） | **[0.57314, 0.40876]** | ✅ 非零且合理 |
| `t2`（matmul④ C） | [0.0054, -0.00133] | ✅ |
| `mBf` / `mF32` | 0.97266 / 0.97776 | ✅ 一致且正确 |
| **`hBf` / `hF32`** | **-0.58203 / 0.00000** | ❌ **同一个更新迭代里，bf16 副本对、fp32 状态是 0** |

`hBf` 与 `hF32` 在同一轮更新里给出的结果不一致 ⇒ **两个 AIV subcore 在做同一份工作、互相覆盖**
（每个 subcore 各自写 hF32 与 hBf，交错后就出现"一半对一个不对"）。已按仓内惯例修掉：
所有按行/按 token 的循环都改成 subcore 切分 —— `for (r = subIdx_; r < N; r += subNum_)`，
覆盖 prologue、staging、`v_new`、状态更新与 epilogue（与 `recompute_w_u_fwd_vector.h` 里
`vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx() → continue` 同一思路）。

修完复测：**m 半边仍基本正确（matched=0.992188、max_abs≈7e-02），h 半边仍为 0/脏值** ✗
⇒ h 数据链上还有第二个问题（与 subcore 覆盖无关）。四个 matmul 的 C 都已验证为合理值，
所以嫌疑集中在"AIV 读 `vTmpF_`（matmul① 的 C）→ 算 `v_new` → 写 `vNewBf_`"这一段，
以及"推迟一个 chunk 的状态更新"里 `decayPrevF_` 的配对。

下一步：把探针限定到**单 chunk（T=64）**，逐个打印 `vTmp → vNewBf → dH → hF32/hBf`，
确认是"读到旧 vTmp"还是"decayPrev 配错"；两者都是局部问题，不需要再改结构。

### 12.14 v2 第八轮：排除"matmul 占用了 workspace 起始处"，问题收敛到 h 的 fp32 状态写入（2026-09-28）

上一轮看到"`hF32_` 恒为 0、而 `hBf_`（bf16 副本）是对的"，且 `hF32_` 原本在**偏移 0**，于是怀疑
CATLASS matmul 在本核 workspace 起始处用了自己的暂存区。本轮把它验证掉：

* 把 `WS_H_F32` 从偏移 0 **挪到所有缓冲的最后**（`WS_GATE + 4096`），其余布局不变；
* 复测结果**完全不变**（h 半边仍为 0）⇒ **"matmul 冲掉偏移 0"的假设排除**。

挪动后再读一次探针（值随缓冲一起搬走，所以与上一轮同值），结论进一步收敛：

| 观测量 | 值 | 判定 |
| --- | --- | --- |
| `vTmp` / `t1` / `dH` / `t2`（四个 matmul 的 C） | [0.01672,-0.00102] / [0.00095,-0.00166] / [0.57314,0.40876] / [0.0054,-0.00133] | ✅ 全部合理 |
| `vNewBf` / `t1Bf` / `mBf`（bf16 中间量） | [-0.40625,-0.53125] / [1.19531,-0.39844] / [0.98438,-0.00044] | ✅ 非零合理 |
| **`hBf`（bf16 状态副本）** | **[-0.58203, -0.97266]** | ✅ 非零 |
| **`hF32`（fp32 状态本体）** | **0.0** | ❌ 写不进去 |
| `m` 半边（1..127 行） | matched=0.992188（仅第 0 行被调试覆盖） | ✅ |

⇒ 现象已收敛成一句话：**h 的 fp32 状态写路径失效，而同一迭代里它的 bf16 副本写入正常**；
四个 matmul 的 C、以及 m 的 fp32/bf16 状态都正常。

下一步（继续缩小）：把探针切到**单 chunk（T=64）**用例 —— 此时"推迟一个 chunk"的更新逻辑退化
（没有上一 chunk），`h` 的期望值就等于 `dH`，一次性看 `vTmp → vNewBf → dH → hF32/hBf` 是否自洽；
若单 chunk 正确、多 chunk 错，则问题在 `decayPrevF_` 与双缓冲的配对；若单 chunk 就错，问题在
h 的写路径本身（同一段代码里 m 能写、h 不能写，说明是缓冲/同步细节）。

### 12.15 ✅ v2（Cube/CATLASS）精度 6/6 全通过 + 性能比 v1 快 17 倍（2026-09-28）

**根因（哨兵实验直接定位）**：在 prologue 往 `hF32_` 写哨兵 0.5 后，单 chunk 用例读回的
`hF32 = 0.5`（哨兵原值）而 `hBf = 0.63672`（= dH，正确）⇒ **h 的 fp32 状态写读到的是 `Add` 之前的值**，
即"**向量结果 → MTE3（写 GM）缺 V→MTE3 同步**"——与 v1 里踩过的同一类坑。修法：把所有
"向量算完紧接 `DataCopy(...GM...)`"前的 `PipeBarrier<PIPE_V>` 换成 `PipeBarrier<PIPE_ALL>`（本文件共 6 处）。

**修完实测**：`npu_smoke_ppfm.py` **6/6 全部 PASS**（模型同构数据，`T=256, K=V=128, BT=64`）：

| 用例 | h 半边 | m 半边 |
| --- | --- | --- |
| gdn-single | matched=1.000000 / max_abs=9.5e-07 | 1.000000 / 2.8e-05 |
| kda-single | 1.000000 / 2.3e-04 | 1.000000 / 3.2e-05 |
| gva-1x2 | 1.000000 / 9.5e-07 | 1.000000 / 2.7e-05 |
| gva-2x4-sub（子区间）| 1.000000 / 7.2e-07 | 1.000000 / 1.8e-05 |
| gdn-multiseg | 1.000000 / 7.2e-07 | 1.000000 / 1.1e-05 |
| kda-multiseg | 1.000000 / 4.8e-07 | 1.000000 / 1.2e-05 |

**性能（`npu_bench_ppfm.py`，同一个脚本同一批 shape，与 v1 直接可比）**：

| 用例 | v1（AIV 向量版） | **v2（Cube/CATLASS）** | 加速比 |
| --- | --- | --- | --- |
| T=1024, HK=HV=4 | 38.25 ms | **2.278 ms** | **16.8×** |
| T=1024, HK=HV=8 | 38.28 ms | 2.299 ms | 16.6× |
| T=4096, HK=HV=8 | 152.76 ms | 8.934 ms | 17.1× |

⇒ **v2 已在仓内可用**（当前在用内核即 v2），精度与 v1 同级（两半都 matched=1.0）、速度快约 17 倍。

**与 1.0× H20 目标的差距与下一步优化**（按收益排序）：

1. **AIC/AIV 串行**：当前每个 chunk 内 AIC 等 AIV、AIV 等 AIC（纯串行），需要**跨 chunk 流水**
   （双缓冲 `vTmp`/`T1`/`dH`/`T2` + 输入 staging），把 cube 与 vector 重叠起来；
2. **AIV 逐行小拷贝**：staging/`v_new`/状态更新目前是"每行一次 DataCopy（512B）+ 多次 PIPE_ALL"，
   改成**整块 [BT,K]/[K,V] 拷贝 + 整块向量运算**，再把不必要的 `PIPE_ALL` 降级；
3. 按 `design.md` 的分核方案把任务从"1 段 × 1 head"细分到能填满 28 个 AIC 的粒度。

### 12.16 v2 性能优化：状态更新批量 + gate 处理，T=1024 从 2.278 → 0.949 ms（2026-09-28）

在 §12.15 精度全过的基础上做了三轮优化，每轮都先跑通 `npu_smoke_ppfm.py` 再量 `npu_bench_ppfm.py`：

| 版本 | 关键改动 | T=1024/HV=4 | T=4096/HV=8 | 精度 |
| --- | --- | --- | --- | --- |
| v1（AIV 向量版） | — | 38.25 ms | 152.76 ms | 6/6 ✅ |
| v2-a（Cube/CATLASS + V→MTE3 修复） | 见 §12.15 | 2.278 ms | 8.934 ms | 6/6 ✅ |
| v2-b（+ 状态更新按 16 行批量） | `h/m` 更新从"每行 8 次 DataCopy/PIPE_ALL"改成"每 16 行一次搬运 + 块内逐行 Muls" | 1.219 ms | 4.721 ms | 6/6 ✅ |
| **v2-c（当前在用）** | v2-b + staging/`v_new` 保持逐行（更稳）；gate 改成按需标量 | **0.949 ms** | **3.488 ms** | **6/6 ✅** |

⇒ **v2 当前比 v1 快约 40×（T=1024）/ 44×（T=4096）**，精度仍为 6/6（两半 matched=1.000000）。

**本轮踩到并记档的坑（重要）**：把 staging 与 `v_new` 也改成"整块 [BT,K] 搬运"后，
速度进一步到 **0.607 ms**，但 **GDN（USE_G）的 h 半边出现 ~0.4% 系统性偏差**（KDA 不受影响，
m 半边也完全正确）——说明问题出在 h 数据链（`kBf_` / `v_new` / `vTmp`）与 dg 的配合上，
与大块 `DataCopy` 的分段/截断、向量化 `Exp` 都无关（两者都单独验过）。
为保证交付物正确性，曾经**退回逐行 staging/`v_new`**；后续二分已把范围收窄（见下方更正）。

补充的排查证据（下一步直接从这里接）：整块 `v_new` 依赖 `vBlkBf_`（UB 里的 v 块），而它只在
**整块 staging** 里被填充 —— 所以"整块 v_new + 逐行 staging"这种组合本身不成立（会读到陈旧 UB）。
真正的二分实验应做在这两种组合上：
① 整块 staging + 逐行 v_new（v_new 从 GM 的 `vBf_` 读）；② 整块 staging + 整块 v_new（从 `vBlkBf_` 读）。
> **2026-09-28 二分结果**：**组合 ① 通过** —— 当前在用版本就是"整块 staging + 逐行 `v_new`"，
> 实测 **6/6 PASS、T=1024 = 0.950 ms、T=4096 = 3.463 ms**。也就是说 **整块 staging 是正确的**，
> 出问题的只有**整块 `v_new`**（组合 ②）。它与逐行版的差别只有：v 从 UB 的 `vBlkBf_` 读（而非 GM 的
> `vBf_`）、`scrF_`/`scrBf_` 按整块 [BT,V] 使用。下一步在"整块 v_new"内部再二分即可定位：
> ⓐ 只把 4 次 `DataCopy` 合成块、算术仍逐行；ⓑ 再合并 `Sub`；ⓒ 再合并 `Cast`。
>
> **ⓐ 已实施并通过**（2026-09-28）：`v_new` 里只把"GM↔UB 的搬运"合成 16 行块（`DataCopy`），
> `Cast/Sub/Muls/Cast` 仍逐行 —— 与已验证正确的版本数学完全一致。实测 **6/6 PASS**
> （h 半边 matched=1.000000、max_abs ≤ 9.5e-07），性能 **T=1024 0.950→0.841 ms、T=4096 3.463→3.049 ms**。
> ⇒ **块级 `DataCopy` 安全、块级算术（块 `Cast`/`Sub`）会引入偏差**；这也排除了"元素数上限/截断"
> 的解释（失败版本每段只有 1024 个元素，已在 4095 以内）。下一步 ⓑ/ⓒ 要针对"块算术"单独定位
> （怀疑与 `scrF_`/`scrBf_` 在两个 subcore 上的交叉写、或块 `Cast` 的 bf16↔fp32 分段语义有关）。

### 12.17 ✅ 找到块算术的真因（双 subcore 竞态）并修掉：T=1024 → 0.501 ms（2026-09-28）

**真因**：块级算术（`Cast`/`Sub`）是由**两个 subcore 共同**执行的，而块内有一部分行属于
另一个 subcore 的逐行 `Muls` —— 那个 subcore 还没算完，块级算子就已经把整块读走 ⇒ **跨 subcore 读半成品**
（竞态，所以偏差小且不稳定，正好是 ~0.4%、只有 USE_G 的 h 命中）。
staging 的 `left` 也有同一隐患（当时侥幸通过）。

**修法**：**按"整段（16 行）"分配 subcore** —— 段内的 `Cast→缩放→Cast→落盘` 全部由同一个 subcore 完成，
不存在跨 subcore 读取。`left` 与 `v_new` 都改成这个模式；状态更新本来就是"整块归单个 subcore"，
符合该模式。

**修完实测**（`npu_smoke_ppfm.py` 6/6 PASS，h/m 两半 matched=1.000000）：

| 版本 | T=1024/HV=4 | T=4096/HV=8 | 相对 v1 |
| --- | --- | --- | --- |
| v1（向量版） | 38.25 ms | 152.76 ms | 1× |
| v2-c（§12.16） | 0.950 ms | 3.463 ms | 40× / 44× |
| v2-d（ⓐ 块搬运） | 0.841 ms | 3.049 ms | 45× / 50× |
| **v2-e（整段分配 subcore + 向量化 dg）** | **0.501 ms** | **1.693 ms** | **76× / 90×** |

**与 1.0× H20 的差距**：每 chunk 成本从优化前的 139 µs 降到 **26.5 µs**；按 model-gk
（176 chunk/链、32 链）外推约 **4.7 ms**，对 H20 的 1.62 ms ⇒ **约 2.9×**（优化前估计 ~6×）。
**剩余优化项（按收益）**：① AIC/AIV 跨 chunk 流水重叠（当前每个 chunk 内两者串行等待，理论 ~2×）；
② staging 的 6 次大块 `DataCopy` 目前两个 subcore 各做一遍，可分工；③ epilogue 逐行拷贝改块级。

> 已顺手做的一项：把 AIC 的 `dH`/`T2` 两次跨核通知**合并成一次**（两者都在下一个 chunk 开头才被 AIV 使用），
> 精度保持 6/6，性能基本持平（T=1024 0.501 ms、T=4096 1.686 ms）—— 说明**瓶颈不在 flag 往返**，
> 而在 AIC 的 matmul 与 AIV 的 elementwise 本身的耗时。

> 又把 **staging 的 6 次大块 `DataCopy` 也改成"按整段分配 subcore"**（每个 subcore 只搬/清零自己那几段，
> 段内完成 `left` 计算），消除了两个 subcore 的重复搬运。精度保持 6/6，性能仍基本持平
> （T=1024 0.501 ms、T=4096 1.717 ms）⇒ 瓶颈同样不在 staging 的搬运量。
> 结论：当前每 chunk ≈ 26 µs，主要由"AIC 4 个 matmul + AIV elementwise 的串行链"构成；
> 下一步优先做**结构性并行**而不是微调：把 `mm1`（`W_c@hBf`）与 `mm3`（`W_c@mBf`）**合并成一次
> `W_c @ [hBf | mBf]`**（同一个 A 操作数，N 由 128 变 256），可省一次 L1 装载与一次 matmul 调用。
另外要留意 `scrF_`/`scrBf_` 在 staging 与 `v_new` 之间是复用的，整块版下两个子核都会写整块，
需要确认没有 subcore 交叉覆盖（KDA 全对、只有 USE_G 的 h 偏 0.4%，指向 dg 与 v_new 的配合）。

**与 1.0× H20 的差距**：H20 在 `model-gk`（T=11264, HK=HV=32, 176 chunk/链）上 `default` 为 1.62 ms；
按当前 v2-c 的每 chunk 成本（~0.949 ms / 16 chunk = 59 µs/链·chunk）估算，模型 case 约
176 × 59 µs ≈ 10 ms（32 条链 / 28 核 ≈ 1.1 链/核）⇒ 仍比 H20 慢约 6×。
后续可做的三件事（按收益）：① 修好整块 staging/`v_new`（~1.6×）；② AIC/AIV 跨 chunk 流水重叠
（当前纯串行，理论 ~2×）；③ 把"1 段 × 1 head"细分填满 28 个 AIC。
