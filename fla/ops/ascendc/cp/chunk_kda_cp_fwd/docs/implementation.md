# A5 KCP v1 落地计划

2026 年 11 月 24 日 · 原生实现与验证记录 · 对标 FLA v0.5.2

[设计详设](../../../../../../../../outputs/KCP_A5_连续CP单大算子详设/design.md) · [接口契约](../../../../../../../../outputs/KCP_A5_连续CP单大算子详设/api.md) · [验证矩阵](../../../../../../../../outputs/KCP_A5_连续CP单大算子详设/validation.md)

## 当前单 kernel 实现

**A5 零数据轮已复用原 chunk32。** `backend="fused"` 的计划在每个 rank 的 `本地段数 × value head 数 ≤ 64` 时走原核 NTD 路径。任务粒度为本地段 × 完整 value head，计算窗口为 128 token。任一 rank 的任务数超出该范围时，全组调用 `ChunkKdaCpFwd` 内部的 chunk32 零轮路径。需要覆盖两种入口的基础包同时选择 `chunk_kda_cp_fwd` 和 `chunk_kda_fwd`。以下表格描述现有 `ChunkKdaCpFwd` 设备路径。

**A5 L2 独立摘要。** [`chunk32_summary.h`](../op_kernel/device/chunk32_summary.h) 复用原 chunk32 K-only 准备、块状 K/K 和三角求解。一个 128-token 窗口由两个 chunk 对组成，AIV 保存下一对的预备量，AIC 依次合成当前对的 E/M。以 `S=[E|M]` 表示待更新状态，先算 `Z=[U|0]-W@S`，再算 `Snext=d[:,None]*S+Kg.T@Z`。滚动状态和通信载荷仍为 FP32，进入 Cube 的操作数为 BF16。输出是每个 K 行依次放 E 和 M 的 FP32 矩阵。独立直调、CPU FP64 对照和设备打点由 [`kcp_summary32`](../../../../../../examples/cp/kcp_stage0/README.md#summary32) 提供。L3 复用同一计算实现生成尾段摘要。

`backend="fused"` 经 Stable ABI 激活 `aclnnChunkKdaCpFwd`。计划生成在调用外完成，每 rank 下发一次 `ChunkKdaCpFwd` 数据 kernel。输入为原始 BF16 Q/K/V、FP32 gate/beta 和本地 packed 偏移。AIC 和两个 AIV 按 chunk32 的 128-token 窗口协作。任务按本地逻辑段 × head 分配，额外保留一组控制资源处理 MC2 生命周期。

| 模式 | 控制 AIV | 计算核组 |
| --- | --- | --- |
| 零轮 | 完成 MC2 初始化和结束协议 | 每个本地序列段从零态跑前向 |
| 整批 | 等首批摘要写完，提交并等待 AG0 | 先处理出站摘要，AG0 期间执行具有本地初态的段。接收可读且独立前向阶段汇合后，按 head 做首段 Merge 和前向 |
| 双批 | 等 AG0 后提交 AG1，用两个互斥的 head 区间接收 | AG0 期间准备第二批摘要，AG1 期间合并第一批并计算第一批来态前向，同时推进独立段，最后消费第二批 |

每个计算核组预留 768 KiB GM。一般出站摘要用 202 KiB，其中两个 64-token 因子槽轮转。两个 AIV 在 AIC 更新当前对第一个 chunk 时准备下一对 K 侧因子。FP32 E/M 在每个 AIV 的 64 KiB UB 中推进，BF16 状态视图供 Cube 消费。状态投影走低地址 UB。状态增量在下一对 K 因子打包结束后，用一次 Fixpipe 分行写给两个 AIV，各写 `[184,248) KiB` 的已释放原始输入区。两个 AIV 消费结束后再继续递推，段末发布 FP32 载荷。出站段若还需要入站初态，则在真实前向阶段重新生成已释放的因子。

出站尾段在本 rank 起始时，直接用零初态跑 chunk32 真实前向，将该段末态同时发布为 E。一个 rank 可以用这条路径处理尾段，同时对属于其他序列的首段执行入站 Merge 与真实前向。

Merge 的 AIV→AIC 和 AIC→AIV 通知分别使用同方向独占的事件 ID。计算核组从真实前向进入下一批 Merge 时，可以独立等待新状态与新矩阵积就绪。

短任务轮转沿用 chunk32 的滚动事件编号。单个 128-token 计算窗同样登记五个被省略的 L1 释放通知，使 AIC 和 AIV 的事件队列按相同顺序推进。Host 为 `SyncAll` 参与核启用 batch mode。A5 CP2 H32 的常规和弱衰减、CP4 H32 的弱衰减均已通过独立 FP64 与三次重复精确比对。CP4 覆盖曾经超时的首尾跨不同序列用例。 [设备计数与精度记录](../../../../../../../../outputs/KCP_A5_连续CP单大算子详设/evidence/20261124-chunk32-cp/summary-delta/README.md)

通信区按 rank-major 的 `[P,H,128,256]` 和发送区 `[H,128,256]` 分配，双批按 head 切成互不重叠的区间。接收区在调用结束前保留。Merge 直接更新核组工作区中的状态。算子自身计算和通信工作区为 `workers * 768 KiB + (P + 1) * H * 128 KiB`，零轮省略第二项。系统通信预留由 SDK 另外计量。packed 偏移和可选分段末态分别属于计划数据和用户输出。

### CP4 真实状态接力

单条连续序列跨 CP4 全组、H32、两批且每卡至少 512 token 时，普通 fused 入口选择真实状态接力。全局 plan 已校验各 rank 的序列边界和选项。其他输入按其现有计划调度。

批次在 rank 间斜向推进。rank 0 从零初态开始，每个后继 rank 直接装载前一波前驱的 FP32 H。每个 rank 对每个 head 只执行一次 chunk32 前向。最后一个有效窗口把实际末态写成连续的 FP32 矩阵，每批 1 MiB。控制组用 CCU AllGather 传输这一个批次，等待返回再放行下一波。接收区按波次奇偶轮转，计算窗口和通信区大小均与 token 数无关。

末 rank 的第一批已经取得上一波的 H，且这批没有后继发送。末 rank 的计算 AIV 先参加当前轮的发送屏障，再推进第一批 chunk32 前向。控制组在屏障后提交最后一轮 AllGather，与第一批前向交叠。第一批读上一轮接收槽，CCU 写另一槽。第二批在前一批算完、CCU Wait 和本卡放行屏障均结束后，才能读取自己的初态。其余 rank 沿用原阶段顺序。

接力路径的自管 GM 为 `(1 + 2P) × 1 MiB`，包括一个批次发送槽和两份 rank-major 接收区。CP4 共 9 MiB。真实 H 直接从接收槽装载，摘要和 Merge 的核组 scratch 由原摘要路径按需分配。SDK 和 HCCL 保留区另计。

Host 按批次宽度启动 16 个计算核组及一个 CCU 控制组，每个计算核组处理各批对应的一个 head。每卡 512 token、H32、弱衰减的最小回归中，四卡输出和末态通过独立 FP64，与原 27 个计算核组的接力版逐字节一致。全局 32K、128K 前向中位数分别为 2.067、7.592 ms，同轮 PR #866 为 3.021、11.203 ms。 [四卡实测及精度](../../../../../../../../outputs/KCP_A5_连续CP单大算子详设/evidence/20261125-cp4-pipeline/independent-ready/README.md#cp4-接力核组数)

### 实机验证

- 备用 A5，CANN 9.2.0-beta.2，物理卡 1/2，`CCU_SCHED`，32 MiB HCCL buffer。CP2 每卡 65 token、3 head 的长序列和 packed 变长，分别跑整批与双批，8 组 rank/形状/配置的逐 token 输出和每段末态通过独立 FP64 标杆。
- 同一环境中，129 token、32 head 的长序列与 packed 变长，整批与双批共 8 组结果均通过，输出和末态最大绝对误差分别为 `4.270e-4`、`4.488e-3`。9 token、3 head 的完全对齐输入验证了零轮路径。
- 原生单卡从原始输入到摘要和真实前向共 8 个窗口、4 条状态链。E、M、末态和输出最大绝对误差分别为 `2.745e-4`、`9.705e-5`、`3.368e-4`、`2.585e-5`。Host 合同 10 项通过。Stable ABI 核对通过。
- 状态增量单次传递和本地尾段复用通过 A5 CP4 H32 的 96 条 rank 精度结果，含首尾跨不同序列的 packed。输出和末态相对 FP64 最大绝对误差 `9.906e-4`、`1.493e-2`，每种配置连续三次逐元素一致。长序列端到端收益及 CP8 仍需继续验收。 [CP4 同卡计时与流水优化](../../../../../../../../outputs/KCP_A5_连续CP单大算子详设/evidence/20261125-cp4-pipeline/README.md)

后续各节保留首轮实现计划和历史试验记录。

## 1. 首版交付

A5 单机 CP2/4/8，每 rank 一次用户设备 kernel。输入为 `B=1` 的原始 KDA 张量及全局 `cu_seqlens`，覆盖单序列和连续 packed 变长序列。输出为本地 KDA 输出与可选的每个本地逻辑段末态。首版固定 K/V=128、chunk64、KFirst、完整 value head 递推和全有界计算/通信 scratch。工作项按本地逻辑序列段及完整 head 分配。低 head 特化、卡内序列再切分、反向与 PTP 将另行设计。

只有跨出本卡的尾段生成出站 E/M。只有从上游延续的首段需要入站 Merge。一个 head 完成 Merge 并获得自己的初态后可推进真实前向。同一段同时入站和出站时，出站摘要独立于真实初态。实现前锁定布局、gate/norm/beta 等数值契约及组内逻辑 rank 到接收槽的映射。

先实现整批和两批两种配置。整批 `G=HV, D=1, J=1` 用于同图基线并作为有独立本地工作的默认。长单序列等形状若组内存在一个只有延续段的 rank，且 `ceil(HV/2)<=C_use<HV`，默认两批 `G=ceil(HV/2), D=2, J=2`。全组无跨 rank 依赖时走零轮。采用组内共同的能力下界决定 `C_use`。入口已有的 staged、pipeline 和 mc2 契约继续各自保持。

## 2. 实现顺序与阶段门槛

| 顺序 | 当阶段交付 | 验收门槛 |
| --- | --- | --- |
| 1. Host 计划、独立标杆 | 推导本地 `cu_seqlens`、首尾角色、前缀起点、全组轮次、owner 和 workspace。把设计中的 CPU 标杆固化成可重复用例 | CP2/4/8 覆盖跨多个 rank、rank 首尾异序列、零轮、奇数 head 尾批。强制整批与两批得到同一输出和末态。元数据错配在数据轮次开始前被发现 |
| 2. A5 packed 分阶段对照 | 复用现有 Prepare、尾段摘要、Merge、FwdH、Finalize。先跑能暴露首尾边界错误的小例，再跑模型形状 | 对同一原始输入逐 token 和逐段对齐 BF16 容差。记录各段初态、输出、末态和必要的中间量。分阶段对照的 launch 次数与容量单列 |
| 3. A5 原生通信探针 | 官方 stock 控制先过资源初始化。再用一个 MIX kernel 验证 FP32 device AllGather 和独立计算组能同时推进。先一轮，后两轮及双槽 | 两 rank 下设备传输与计算真实重叠，逐元素校验载荷。区分本地摘要可见、提交、接收可读、源释放与旧槽释放。给一个 rank 加延迟，确认其他核组的可运行任务继续推进。协议通过后扩到 CP4/8 |
| 4. 原生计算 Stage | chunk64 做 Stage 0 的 gate/norm/beta 和尾部掩码。逐步加因果块准备、摘要、Merge、真实状态与输出，在当前阶段通过数值测试后再进入下一阶段 | 每步留存小用例、实际设备结果、UB/L1/L0/GM 地址和事件记录。先用零轮计划验收本地计算，再用跨卡计划验证状态路径。两个窗口各自具备 ready/free 和复用证据 |
| 5. 单 kernel 集成 | 接入 `backend="fused"`，按计划组织控制角色和计算组。让整批和两批复用计算 Stage，差异限定在通信轮次和放行 | 每 rank 真正一次数据 kernel。单序列和 packed 在 CP2/4/8 通过精度、槽位回绕、组内 rank 重排、超时和清理测试。用设备时间线确认整批和两批各自的覆盖来源 |
| 6. 选型与交付 | 以 32 head、全局 32K/128K token 为主，对比整批、两批和同图串行，并量化 J1/J2、KFirst 重建和首窗预备 | 至少 20 次 warmup、100 次迭代、5 个交错 A/B 窗口。以最慢 rank 完整前向时间为主。用重复基线标定噪声。若自动规则与实测结论不同，更新对应形状的选择规则 |

Host 和计算 Stage 的工作可以独立于通信初始化推进。CCU 探针与本地 Stage 验收各自有结论，两条线都通过后才进入跨卡融合。先保留可独立运行的 A5 分阶段基线，单 kernel 验收只使用原生融合路径的数据 kernel 和性能证据。

### chunk32 · L3 跨卡接线

跨 rank 路径按本 rank 有无入站首段选择计算。有入站首段的出站 rank 用 `Summary32` 计算本地 E/M。仅有出站尾段的 rank 用零态 chunk32 真实前向同时产出该段输出和 E。Merge 在第一个来源直接取 E，这条路径的 M 槽写零。已完成的尾段从后续本地前向任务中跳过。最后一次 GM 写入完成后才进入发送屏障。不发送摘要的 rank 将同一布局的发送槽置零，仍参加相同轮次。

通信控制使用 CCU AllGather、完成等待和全组屏障。Merge 按来源 rank 顺序形成 FP32 初态。AIC 用 64×128 tile 计算矩阵乘，两个 AIV 各自按 16 行读取 E、累加并写回连续的状态行。Merge 临时区位于各 AIV 的前 16 KiB UB。随后两个 AIV 把各自 64 个 K 行交给原 chunk32 的初态装载入口，真实前向复用原 AIC/AIV 窗口流水。无上游依赖的本地段以零态进入同一流水。计算侧复用已有的定长 worker workspace，摘要临时量和 Merge 状态错时使用。

单段入站 rank 在最后一轮接收屏障前，用两个 AIV 生成第一个 64-token 对的独立 K/Q 与 gate 因子。Merge 完成后沿原 chunk32 真实前向消费首对，初态从 Merge 输出装载。其余任务按原窗口流水供数。L3 用现有整批和两批配置核对跨 rank 结果与屏障语义。

## 3. 实现时重点检查的三个风险

- **通信进度。** 现有 CCU 原型的全 AIV 等待与屏障需要在探针阶段核对。若后端只能让所有计算 AIV 进入阻塞屏障，它不能放行本设计的重叠模式。先核查 worker 参与范围和设备侧完成语义。
- **分核和等待。** 优先产出最早可发批次，能调度后续摘要时保持下一轮进度。接收可读后，单列 `recv_ready → merge_start` 的等待，避免长独立前向占满核组而把已就绪 Merge 推迟。等待初态的任务留在待派状态，控制角色负责确认通信完成。
- **存储与开销。** 独立核算发送、接收、初态和核内窗口的最后读者。固定资源配置下，算子 scratch 不随总 token 量增长。packed 元数据、可选末态和后端预留分别计量。检查两批是否真正减少了最慢 rank 的等待，避免把通信区间与大 kernel 的外层交叠当成有效覆盖。

## 4. 当前证据与开始条件

- 设计评审阶段已有 192 组 CPU FP64 边界模型验证，覆盖整批和两批。尚需将其固化成仓内可重复测试，并核对新代码的实际执行结果。
- 2026 年 11 月 24 日，两台 A5 都以卡 1、2 复测了官方 stock CCU 控制。主机在 `hcclCommInitRootInfoConfig` 返回 4，日志为 `~/kcp_v1_ccu_20261124.qoaifU/stock.log`。备用机在 `HcclAllocComResourceByTiling` 返回 4，日志为 `~/kcp_v1_ccu_20261124.x0QhJt/stock.log`。这是当前通信环境的失败证据，原生 FP32 probe 尚无通过证据。
- Host 计划和 packed 分阶段对照已在内部开发入口运行。单 kernel 的公开 `fused` 入口等待跨卡集成验收。

## 5. 2026 年 11 月 24 日实现记录

环境为 A5 备用机、Ascend950PR、CANN 9.2.0-beta.2、torch 2.11.0、torch_npu 2.11.0.post3。开发记录位于本机远端目录 `~/kcp_v1_packed_20261124.wrkOtJ/`。

| 检查 | 实际证据 | 判定 |
| --- | --- | --- |
| Host plan 合同 | `python -m unittest discover -s tests -p test_kcp_contract.py -v`，10 项通过。192 组边界覆盖 CP2/4/8 的前缀来源、尾段归属、零轮等条件。整批、两批的容量按 27 个核组核对 | 通过 Host 算法及容量模型 |
| A5 packed staged CP2 | 卡 1、2，`CCU_SCHED`，HCCL buffer 32 MiB。两组 `(T,HV)=(65,3),(129,32)`。长序列、边界对齐及 packed，分别用整批和两批，共 24 个 rank/形状/组宽结果。3 head 最大绝对误差为输出 `2.808e-4`、末态 `2.377e-3`。32 head 分别为 `4.101e-4`、`4.488e-3`。`cp2.json`、`cp2_h32.json` | 通过该组 BF16 对照 |
| A5 packed staged CP4 | 卡 1、2、3、4。`CCU_SCHED`、`AI_CPU` 均在首次 `HcclAllGather` 失败。`CCU_SCHED` 的首个 HCCL 错误为 `CalcAlgHierarchyInfoV2` 返回 5。拓扑显示卡 4 与前三张卡不在同一 UB 互连域。`cp4.log`、`cp4_aicpu.log` | 当前组未通过，换健康同域卡组复测 |
| 官方 BF16 CCU 对照 | 备用机在同一 CP2 卡组、`CCU_SCHED` 和 32 MiB buffer 下，`aclnnAllGatherMatmulV2` 的 gather 和输出逐元素匹配。`stock_ccu_sched.log` | 通过该环境的官方对照 |
| 原生 FP32 传输 | 修复前，现有算子在 `AclrtSynchronizeDeviceWithTimeout` 返回 507018，设备日志显示 `RunAicpuKfcSrvLaunch` 参数错误。对照当前 SDK 的官方 Host 入口，修正 Host executor 的 CCU 服务编号。修复后，同一 CP2 卡组 3 head × 3 次和 16 head × 8 次，gather 逐元素相等且有序前缀相符。最新默认包重建、Stable-ABI 检查及 16 head × 3 次复跑通过。`native_cp2_service/native_gate.json`、`native_cp2_h16/native_gate.json`、`native_v1_default.log` | CP2 原生 FP32 单轮传输通过 |
| 原生一轮进度探针 | 16 head，两个 MIX 核组分别执行 CCU 及独立 AIV 计算，三个样本等待区间交叠 `63243/64981/64693` cycles。msprof 导出了两张卡的融合任务及 CCU 记录。`native_progress/native_gate.json`、`progress_profile_retry/` | 单 kernel 内观察到独立计算与提交至接收完成区间交叠 |
| 原生两轮进度探针 | 15 head，按 8+7 顺序调用，同一次 kernel 内两轮写不同 head 区域。两个 rank 各 3 次逐元素通过。独立计算和第一轮等待交叠，第二轮交叠为零。`native_progress2/native_gate.json` | CP2 两个通信 handle 和奇数头分片通过。此探针在两轮后才 Merge |
| 原生 Stage 0 | 单卡 5，长度 1/64/65/9，3 head、5 组。Q/K 的 BF16 最大绝对误差均为 0，log2 gate 累计 `7.891e-5`，beta `4.959e-8`。msprof 记录一个 `kcp_stage0_check` MIX_AIC launch。`kcp_stage0/run.log`、`kcp_stage0/stage0_profile/` | 通过数值及 launch 检查。完整 kernel 的跨核、跨窗口流水另验 |
| Stage 1 AIV 与 Cube | 单卡 5，长度 1/33/64，KK、QK 和 padding。AIV 生成的 BF16 因子与主机舍入参考逐元素相同。Cube score 相对独立乘积最大绝对误差 `4.470e-8`，相对未量化公式 `1.742e-4`。`kcp_stage0/run1_factors.log` | 两个原生计算环节分别通过，当前用两次 launch 交接 |

### 原生进度探针复现

在隔离 wheel 环境构建单算子包。`FLA_KCP_A5_PROGRESS_PROBE=1` 为一轮，改成 2 为两轮。诊断 build 将 rank 0 初态的首 64 字节用于两个对齐计时记录，runner 解析后清零这些测试字节，再按原前缀检查数值。普通包使用原始初态布局。

```sh
export FLA_KCP_A5_PROGRESS_PROBE=1
export CXXFLAGS="${CXXFLAGS:-} -DFLA_KCP_A5_PROGRESS_PROBE=$FLA_KCP_A5_PROGRESS_PROBE"
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=kcp_all_gather_merge -j8
python -m fla_npu_opp_manage --overlay --source "$(find build_out -maxdepth 1 -name '*.run' -print -quit)"
HCCL_OP_EXPANSION_MODE=CCU_SCHED HCCL_BUFFSIZE=32 \
  torchrun --standalone --nproc_per_node=2 examples/cp/check_kcp_native.py \
  --heads 16 --repeats 3 --progress-probe --probe-rounds 1 --output outputs/kcp_wait_probe
```

两轮测试使用 15 head 和 `--probe-rounds 2`。其两个区间位于同一个 rank-major 接收张量内。探针 3 使用 `FLA_KCP_A5_PROGRESS_PROBE=3` 和 `--early-merge --probe-rounds 2`。CP2 rank 1 在第一轮可读后、第二轮等待返回前合并第一批前缀，三个样本均校验载荷和可检查的前缀字。计时占用首个前缀的 16 个 FP32 字，fixture 将相应原始值设为零。计时证明第二轮提交后的放行次序，实际传输与该计算的交叠需结合 CCU 任务时间线分析。
