# ChunkKdaFwdPrepare 设计

方案设计规则版本：V2。公开合同见 [api.md](api.md)。

### A2 R26 key 与不可变数学策略

A2 R26 使用共享 CANN 声明生成六个 Generic runtime key，key 只编码
`normMode × gateMode`：0/1 为 Identity/L2 + Precomputed，2/3 为
Identity/L2 + Softplus，4/5 为 Identity/L2 + SafeSigmoid。

`normMode/betaMode/gateMode/outputMode/useExp2` 由 host 每次调用写入不可变 tiling，
AIC/AIV 读取同一份合同。AIC/AIV 共用同一 pipeline、relay dtype、资源选择和 workspace
上界；host 序列化、device wire、slot 大小由同一字段声明和静态断言约束。
R26 A2 的 dtype 合同是 q/k/v=BF16、g=FP32、beta=BF16、chunk=64、K=V=128，
并要求 `use_exp2=true`；不满足时 host 明确返回参数错误。A5 保留原入口、key 编码和 arch35 实现。

Generic 运行时字段覆盖 six-key 对应的 norm/gate 组合；beta/output 等数学模式仍在一次调用内
不可变，不重新展开旧数学开关的笛卡尔积。CommonMath 和指数域常量折叠仅在其已验收合同下启用，
其余允许组合保留完整泛化路径或按 host 合同拒绝。

### A2 Generic 的布局、求逆与中间量

完整 FP32 右向消元求解两个32阶叶子；每叶保持递增列的 Mul/Sub 顺序，
不采用 Target 的 L³ 截断。两个独立消元链交错发射；每行系数 Gather 为8 lane，
Mul 的 block stride0复用系数，减少系数搬运而不改变逐元素浮点顺序。
beta band、norm 和 V6 同样使用紧凑行标量广播。叶子最后以批量 CAST_RINT 和
二维 UB copy 恢复既有 Akk payload；中间 scratch 均在末次消费后才复用。

FP32 gate 的 V0 norm 按32行计算平方、归约、epsilon、Sqrt、Div、Mul和BF16舍入，
Q/K顺序复用私有区。相对 private base，values占 `[0x8000,0xC000)`、
squares占 `[0xC000,0x10000)`。相对共享scratch起点 `0x1A000`，repeat占
`[0,0x400)`、Brcb占 `[0x400,0x800)`、ones占 `[0x800,0x880)`，
reduceTmp提供对齐地址 `0x1000`；均低于beta偏移表 `0x1F00`。
当前验证的CANN9.2 ReduceSum AR/isReuseSource使用squares原位归约，
不访问external temporary。该SDK行为是资源规划的依赖，不能外推为其他版本通过。
norm期间共享scratch已归当前pair所有，FP32 gate位于独立共享G区，beta保持原地址。
BF16 gate 保留既有逐行 norm。

head-major V1及FP32 relay V6使用16行tile；sequence-major使用7行，保留beta Gather偏移区。
这些路径的factor使用共享UB `[0x1A000,0x1C000)`，stage使用尾部
`[0x2E000,0x30000)`。BF16 relay的head-major V6使用32行tile，借用另一pair
已退休的Q/K区：factor为other private `[0,0x4000)`，stage为 `[0x4000,0x8000)`。
借用前消费other ioFree的MTE3_MTE2许可，再以MTE2_V传递给Vector；随后归还
同一ioFree许可。下一代输入还需等待当前VF末次使用后发布的sharedFree，故不能提前覆盖。
两份private `[0,0x12000)`、`[0x1C000,0x2E000)`
由各自ioFree保护；共享G/scratch由sharedFree保护，V1和V6不同时使用临时区。
V3 beta band临时区为共享 `[0x1A000,0x1B300)`，与betaEff、偏移表不重叠。

BF16 relay限定为 FP32 gate、BF16 beta、L2、Sigmoid beta、SafeSigmoid、exp2、
Save、head-major、合法整数GVA映射（HK>0、HV>=HK、HV%HK=0）、
epsilon=1e-6和lower_bound=-5。scale沿用host检查后的动态有限值，
不再以精确等于1/√128作为能力条件。scale仍在原位置缩放Aqk/qg_scaled，
不改变求逆或FP32 T relay。
能力判断不绑定模型编号、精确T/H或cu_seqlens分布；其余配置走FP32 fallback。
此范围使用两个32行锚点构造四个细粒度score band，减少Kminus重复处理；
改变内部因子舍入的范围须由原生双参考独立验收。

GVA的value head映射为hk=floor(hv/(HV/HK))。各hv仍独立保存gate、beta、
score、求逆、Akk和RHS；仅IsQkOutputOwner写入共享HK的公开Q/K检查点。
此扩展不新增跨value-head的Q/K复用，不改变slot、事件或workspace。

C4/C5 的 B、X0、T和跨叶子乘法保留FP32。C5完成求逆后，BF16 relay将完整Akk
转BF16 NZ直写L1，由FIX_MTE1保护C7读取；fallback保留FP32 GM中转。
V6在FP32中形成K/V RHS，按原舍入点转BF16，再整head写出两个独立GM lane。
MTE3_V完成后才允许覆盖打包区。C7为BF16操作数、FP32累加；fallback保持原精度。
关闭的公开输出不充当内部scratch。

V6只把独立factor计算放在已提交的kBeta写回之后，覆盖stage前仍等待MTE3_V。
既有DCCI/DDR、ioFree/sharedFree和跨核RHS ready/free保持独立职责。
C7 pair1的最后L1读取由MTE1_MTE2许可交给下一轮C2 pair1，初始化、空任务、
回卷和退出均闭合一次许可；workspace free不能替代L1末次读取完成。

规范varlen索引且未拆head时，两核类型按绝对token位置的单调lower_bound划分
连续chunk区间。端点为0/total，区间不重不漏；非规范索引、定长和head-split
保留原分核。原32模型仍是BNSD，BSND性能不从BNSD结果推断。

### A2 R26 pipeline variant

R26 A2 统一使用 `PreparePipeline::Generic` 及 `arch22_rewrite` 的完整流水，
host 当前不选择旧 `PreparePipeline::Target`。旧 H96/T8192 Target 说明只作为历史资料，
不属于本 PR 的 A2 正向支持范围，也不能用旧 Target 的 key 或性能结果替代 R26 验收。
A5 仍从原 arch35 入口独立选择其资源和 key。13 个输出槽位、公开接口、事件账本和
workspace 归属保持不变；关闭的公开输出不得用作 scratch。

## 1. 边界

`ChunkKdaFwdPrepare` 是独立的 Ascend C 算子，完成：

```text
Q/K L2 norm + gate cumsum + prepare + post-WU
```

算子返回固定的 13 槽位合同。`gk/Aqk/w/u/kg/qg_scaled` 是 Prepare 与 FwdH/Finalize
之间的正向必需输出；其余七项是可按反向策略裁剪的检查点。chunk 间状态递推由 FwdH
完成，最终 attention 输出由 Finalize 完成，二者不属于本算子。

A5 对每个 chunk/value head 仅读取一次的原始 V 和 gate 禁用 L2 cache；
`HK=HV` 时 Q/K 也禁用 L2，GVA 场景保留 Q/K 的默认缓存策略供多个 value head 复用。
beta、A_log、dt_bias、workspace 和后续阶段会重读的 gk 保持原缓存策略。
纯推理 `OutputMode::None` 的 q/k 归一化仍计算 rstd 并在寄存器内使用，
但不将仅供反向保存的 q_rstd/k_rstd 写入 UB；其他输出模式不变。

Shape 符号沿用 [KDA 模型符号表](../../README.md#核心符号)，公开输入输出与已知限制统一见
[算子 README](../README.md#输入输出)。

## 2. 数学语义

对长度 `M<=64`、K/V 维为 128 的每个 chunk，先生成：

```text
q_rstd[i] = rsqrt(sum_d(q[i,d]^2) + epsilon)
k_rstd[i] = rsqrt(sum_d(k[i,d]^2) + epsilon)
q_hat = cast_bf16(q * q_rstd)
k_hat = cast_bf16(k * k_rstd)
```

关闭 L2 norm 时，`q_hat/k_hat` 等于输入，两个 rstd 输出为 FP32 1。

gate 输入先变成自然对数域 step：

```text
x[i,d] = g_raw[i,d] + optional_dt_bias[d]
delta_g[i,d] = g_raw[i,d]                              # 预计算 step
               -exp(a_log[h]) * softplus(x[i,d])       # Softplus
               lower_bound/(1+exp(-exp(a_log[h])*x))   # SafeSigmoid
```

`use_exp2=true` 时在 cumsum 前乘 `1/ln(2)`，累计量保存于 log2 域；否则保存于自然对数域。
后续统一以 `E(x)` 表示对应域的指数函数。

每个 16 行子块选择中点参考行，S 固定为 4：

```text
Gref[s] = G[floor((16*s + min(M,16*(s+1)))/2)]
Qplus[i] = q_hat[i] * E(G[i] - Gref[s(i)])
Kplus[i] = k_hat[i] * E(G[i] - Gref[s(i)])
Kminus[s,j] = k_hat[j] * E(Gref[s] - G[j])
```

每个 `Kminus[s]` 只保存该 query band 可访问的 key 前缀，四段分别为 16、32、48、64 行。
`Gref` 始终只是 `[4,128]` FP32，不物化 broadcast 矩阵。

C2 得到两个因果 score 矩阵后，V3 施加 mask 和 beta，完成两个 32x32 叶子逆并构造
`Aqk/Akk`。C4/C5 计算跨叶子修正，V6 生成：

```text
qg = cast_bf16(q_hat * E(G))
qg_scaled = cast_bf16(fp32(qg) * scale)
kg = cast_bf16(k_hat * E(Glast-G))
K_positive = cast_bf16(k_hat * E(G))
K_beta_g = cast_bf16(fp32(K_positive) * beta_eff)
V_beta = cast_bf16(fp32(v) * beta_eff)
```

C7 最后计算 `W=Akk@K_beta_g` 和 `U=Akk@V_beta`。BF16 操作数均使用 FP32 累加。

### 2.1 输出生命周期

三种公开输出策略如下：

```text
none:
    gk, Aqk, w, u, kg, qg_scaled

recompute:
    none 的六项
    + Akk, q_hat, k_hat, q_rstd, k_rstd, beta_eff

save:
    recompute 的十二项 + qg
```

策略只影响最终的公开 GM store。V0 仍生成 norm、gate、beta 状态；V3 仍把 `Akk` 写入
workspace 并供 C4/C5/C7 使用；V6 仍生成 UB 中的 `qg`，再基于其 BF16 舍入值计算
`qg_scaled`。因此三档的公式、UB/L1/workspace 布局、VF 调用数和同步协议一致。

算子 IR 和 kernel ABI 固定保留 13 个 `REQUIRED` 输出槽位。L2 未请求的槽位传
`nullptr`，L0 用不会被当前编译实例写入的合法 descriptor 占位，防止 launcher 压缩参数。

L2 的输出档位用整档匹配，实际接受四种 mask：`none`（六个必选槽）、`forward`
（`none` + `Akk`）、`recompute`（`forward` + `q_hat/k_hat/q_rstd/k_rstd/beta_eff`）、
`save`（全部 13 槽）。Python 入口 `fla_npu.ops.ascendc.chunk_kda_fwd_prepare` 的
`backward_mode` 与这四档一一对应，默认 `save`；未选中的槽在 L2 层传 `nullptr`。
反向只需要 L2 norm 回代时用 `recompute` 即可，无需把 `qg` 也拉出来。

格式判据：L2 只拒绝私有（分形）格式，`ND`/`NCHW`/`NCL`/`NHWC` 等非私有拼写均接受，
张量需连续；此前强制 `FORMAT_ND` 会把 NCHW 拼写的调用方误判为非法参数。

`output_final_state` 与 `return_intermediate_states` 属于完整 forward，分别控制用户可见的
`final_state/h`，和本节的反向策略正交；Prepare 不接收这两个属性。

## 3. 八阶段流水

| Stage | 核 | 内容 |
| --- | --- | --- |
| V0 | Vector | norm、beta、gate、cumsum，生成并写出 Q/K 保存量与 `gk/beta_eff` |
| V1 | Vector | 一次 VF 生成 S=4 的 `Qplus/Kplus/Kminus` |
| C2 | Cube | 四个 stacked band MMAD，同时生成 raw Aqk/raw Akk |
| V3 | Vector | causal mask、beta、叶子逆，生成 Aqk/B/X0/X1/negX1/Akk |
| C4 | Cube | `M>32` 时计算 `T=B@X0` |
| C5 | Cube | `M>32` 时计算 `q10=negX1@T` |
| V6 | Vector | 生成 qg/qg_scaled/kg/K_beta_g/V_beta |
| C7 | Cube | 计算 W/U |

每个 Vector Stage 统一执行一次阶段计算；A2泛化路径可读取调用内不变的数学模式。每个 Stage
只属于 Vector 或 Cube 一类，Cube 不读取同一 Stage 新生成的数据。

## 4. C2 分块

每个 band 把 Qplus 和 Kplus 沿行堆叠为 `[32,128]`：

```text
[Qplus_s] @ Kminus_s^T = [rawAqk_s]
[Kplus_s]                [rawAkk_s]
```

四次 MMAD API 的 N 依次为 16、32、48、64，每次同时覆盖两个数学乘积。因此 full chunk
是 4 次 MMAD 提交、8 个数学矩阵乘积，不是 16 次 `16x16` 独立提交。

## 5. 分核与 GVA

Prepare 不存在 chunk 间依赖，Host 首先仅按 chunk 分核。只有 chunk work item 少于可用 AIC
时才增加 head partition。head partition 的边界按 `group_size=HV/HK` 对齐，禁止拆开共享同一
Q/K head 的完整 GVA value-head 组。

A5 dense 在全局只有一个 head partition、但 chunk 数不能整除已用核时，继续保持完整轮次只按
chunk 分核。例如 128 个 chunk、28 个核时，每核先处理 4 个完整 chunk，共 112 个；剩余 16 个
chunk 已不足以再覆盖全部核，因此只把这部分按 4 个 value head 的硬件处理组展开并均摊。仅当
`group_size=HV/HK` 能整除 4 时启用该路径，保证尾部边界仍不拆开 GVA 组。AIC 和两个 AIV
使用完全相同的主体、尾部映射顺序，尾部不同核写入互不重叠的 `(chunk, head)` 区域。变长、
A2/A3、已经启用全局 head partition 或 GVA 边界不满足条件的场景仍使用原调度。

每个 AIC workgroup 一轮最多处理 4 个 value head。A5 的 AIV0 处理 local head 0/1，AIV1
处理 2/3；A2/A3 的两个 AIV 按 pair wave 处理 0/2 和 1/3。不同 value head 即使映射到同一
Q/K head，也只允许 GVA 组首 owner 写 `q_hat/k_hat/q_rstd/k_rstd`，避免 GM 重叠写。

当前实现仍按 value-head slot 搬运和归一化 Q/K。多个 value head 映射到同一 Q/K head 时，
会从 GM 重读对应 Q/K；这样无需引入跨 AIV 的 UB 共享、额外 VF 调用或 UB 搬位，同时保持
静态双缓冲和既定 Stage 合同。后续若消除该重读，必须同时证明跨 AIV 共享与成对同步不会
破坏当前生命周期。

## 6. 静态内存

内核不使用动态 UB 队列，所有 UB/L1/workspace 地址在编译期固定。地址可在不同 Stage 换义，
但活跃生命周期不重叠前不会复用，也不在 UB/L1 内移动数据。

- A5 UB 为两个 112 KiB 计算槽和两个 12 KiB 状态槽，总计 248 KiB。
- A2 Generic sequence-major使用184KiB UB，head-major使用192KiB；尾部8KiB
  由V1/V6分时复用，不视作可无条件借用的额外双缓冲。
- L1 为 512 KiB，四个 head lane 保存 C2 payload，其余区域保存 X0、negX1、T、Akk，
  以及 C7 的 RHS（K_beta_g/V_beta）专用 lane。RHS 不再复用 head lane：C2 的
  score 与 C7 的 RHS 同址时，下一个 chunk 的 C2 会在 C7 读完 RHS 之前把该 lane
  覆盖成 Qplus/Kplus。
- Arch35 每个 workspace slot 为 `0x1A400` Byte，保存 Qhat、Khat、betaEff
  和 72 KiB payload。
- Arch22 Generic每slot追加20KiB AIC relay、64KiB RHS独立区及16KiB FP32 Akk区，
  总计 `0x33400` Byte。BF16分支保留相同slot大小和间隔，每个RHS只用所保留32KiB
  的前16KiB，其余空间不借给活跃数据。V6→C7不与V1 payload换义地址。
  C2
  写四段 raw score，V3 消费后 C4 在同址前 4 KiB 写 `T`。host 根据目标
  平台选择 slot 大小，A5 不预留 Arch22 的追加区。

V1 payload 由 16 KiB Qplus、16 KiB Kplus 和 40 KiB Kminus prefix 组成，共 72 KiB。

## 7. 同步

A5 使用 `AscendC::Mutex` 约束 MTE2、V、MTE3、MTE1、M 和 Fixpipe 对静态本地地址的
生命周期；A2/A3 使用配对的 HardEvent。核间数据通过 ready/free 双向握手传递：

A5 的 V1/V3 使用公开 `asc_vf_call` 启动 VF。目标 CANN 9.1 规定该接口返回时 VF 内部
Reg 矢量指令已经完成，因此编译器会在调用边界生成 V 到 Scalar 的 join；旧
`AscendC::VF_CALL` 包装生成同一条 join，当前版本没有可替换的 SIMD 异步入口。Mutex
仍负责 MTE2、V、MTE3 对静态 UB 槽位的生命周期交接，不能用 VF 返回语义替代。

```text
AIV: wait free -> 写 workspace/UB -> set ready
AIC: wait ready -> 消费 -> set free
```

A5 在 AIC 视角使用 ready `0/1/16/17` 和 free `4/5/20/21`；A2/A3 mode2 pair wave使用
ready `0/1` 和 free `2/3`。同一 flag ID 在上一代 set/wait 闭环后才复用，不依赖核启动顺序。

C4 对全部 active head 先提交一轮，再提交 C5，使后续 head 的独立 C4 能进入流水。C4 把
B/X0/negX1/Akk 搬入每 head 独立 L1 后即可归还 workspace payload，V6 可与 C4/C5 排空重叠。

## 8. 模板轴

A2 R26 key 仅选择 `normMode × gateMode` 六个 Generic 实例；g/beta dtype、pipeline、relay
dtype 和资源合同由 host/device 共同固定为 R26 支持范围。数学和四档输出模式由不可变 tiling
传递，不能通过 key-filter 隐藏未编译组合。A5 保留原编译期数学与输出模板。
`q/k/v` 固定为 BF16，R26 A2 只接受 g=FP32、beta=BF16、chunk=64、K=V=128、exp2=true。

A2 的 outputMode 只控制公开写回，不改变内部 scratch 和事件计数；关闭的公开输出不能复用为
scratch。架构通过 `__CCE_AICORE__` 编译宏选择，kernel 内没有运行时 A2/A5 混合分支。
