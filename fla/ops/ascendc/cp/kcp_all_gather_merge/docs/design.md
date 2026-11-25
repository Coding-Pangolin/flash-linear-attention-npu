# KcpAllGatherMerge 详设

方案设计规则版本：V2。接口边界见 [api.md](api.md)。

## 目标与计算顺序

以单个 bounded MIX work 替换框架 AllGather 和独立 Merge launch。
`X[rank,h,k,:]=concat(E,M)`，输出按 `H=0; H=M_i@H+E_i, i=0..r-1` 有序递推。
不改成 SUM AllReduce，不压低摘要 dtype，不改变本仓 Merge 的累加和舍入点。
本 kernel 完成通信后才消费 gathered；跨组重叠由父层整前向编排实现，须由时间线证明。

## Stage 与工作映射

| Stage | 执行单元 | 行为 | 完成/依赖 |
|---|---|---|---|
| G0 | 两个 AIV | InitV2、SetCcTilingV2、AllGather<true> | 生产者已完整写出 summary；true 表示提交，不表示完成 |
| G1 | 两个 AIV | Wait、SyncAll<true>、Finalize | 所有通信客户端完成后才能复用本地资源 |
| B | AIV → AIC | MTE3 发布 flag 8；AIC 的 MTE2 等待 | A2 mode 2 成对；A5 mode 4，AIV1 使用 24 |
| M | 一 AIC + 两 AIV | 调用现有 Merge 的 Init/Process | rank 0 跳过；r=1 只复制 E0；r>=2 执行 GEMM/加法 |

blockDim=1、usedAic=1。每个 AIV 按 head 奇偶分工；Cube 按原 Merge 的 head/rank
次序处理。该上限用于初始正确性和避免跨 block 屏障依赖尚未调度的 work，不代表
性能最优并行度。r<=1 的验证无法覆盖真正 GEMM，必须补 CP4 的 r=2/3。

## 内存与所有权

设 `B=H*128*128*4`，库 workspace 为 W。

| 区域 | 字节区间/大小 | 所有者与生命周期 |
|---|---|---|
| summary | `2B` | 调用方生产；至通信完成只读 |
| gathered | `P*2B` | HCCL 写；Wait 后 Merge 读；独立于 h |
| library workspace | `[0,W)` | CANN/通信服务；不能假设为零 |
| H scratch | user workspace `[0,B)` | Vector 写、Cube 读/写、Vector 加 E |
| M ping/pong | user workspace `[B,3B)` | Vector pack 与 Cube MTE2，沿用 Ready/Free 保护 |
| h | `B` | Vector 最终写；rank 0 由调用方置零 |

总请求 `W+3B`。A2 使用 GetUserWorkspace，A5 沿用现有 Merge 的 raw user-pointer
约定；不能再次盲目加 W。A5 H=3 的实际诊断请求为 17,367,040 bytes。
HCCL 新增持久通信资源不包含在这个数里，须额外测量，不以 torch allocator peak 代替。

Merge 的 Vector 使用 UB `[0,64KiB)` FP32、`[64,128KiB)` FP32、
`[128,160KiB)` BF16。A2 Cube L1 H 从 0 起，M 两半位于 128/192KiB；
L0C 两个 tile 位于 0/64KiB。A5 Cube 的双 L1 stage 各 256KiB，L0C tile 位于
0/128KiB，并直接 Fixpipe 到对应 AIV UB 0。所有地址和 pipe event 由引用的 Merge
头统一定义；本算子只新增 G/B 阶段，不改变其内部生命周期。

通信对象 Finalize 完成后才建立 Merge 的 TPipe；AIC 在 flag 8/24 前不初始化 Merge
资源。Merge 的 flags 2..5 与新增 flags 不冲突。Ready、CUbFree、tile events 均由
原 Merge 对称 drain。不得在保留 HCCL 活跃状态时直接拼接 Prepare 的 248KiB UB。

## 平台与模板

Host 用 `GetCurNpuArch()==NpuArch::DAV_3510`；kernel 用 `__NPU_ARCH__==3510`。
仅一 FP32 D_T=30 实例，TilingKey=30。A2 默认 AICPU；A5 候选使用 CCU。
MC2 tiling engine CCU_SCHED=6 与 device HcclServerType::CCU=5 是不同枚举。
前者依据 ops-transformer 155e4400 的 `Mc2Comm::ENGINE_CCU_SCHED`；后者依据
目标 SDK 的 `hccl_common.h`。Python 环境的 CCU_SCHED 字符串另属通信域配置。

内部生成的 aclnnInner 符号通过局部 linker map 隐藏，公开 ABI 只有手工声明的一对
aclnnKcpAllGatherMerge 入口。CMake 在构建目录展开 Merge 头依赖，保证隔离打包可构建。

## 验证矩阵

依次验证：参数负向 → FP32 rank/head/坐标/调用编号模式 → exclusive prefix → CP4
非平凡 GEMM → 连续调用 → 非 WORLD 子域 → 父层整前向/保存缓存/显式反向。
H=1/2/3/8/16、P=2/4/8 和两个平台的最终覆盖仍按父层验证记录逐项验收。
同步/越界工具与性能采集独立运行，插桩结果不用于加速结论。
