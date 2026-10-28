# ChunkKdaFwdPrepare A2 设计（R37）

> 本文按仓库 CANNBot/CATLASS 五阶段工作流（PR833）记录 A2 的实际实现。设计、直调验证、适配后 ATK 验收和上库门禁分开记录；本文只描述已经进入候选源码的实现，不把实验性方案当成支持合同。

- 关联实现：`fla/ops/ascendc/kda/chunk_kda_fwd_prepare`
- PR：[#994](https://github.com/flashserve/flash-linear-attention-npu/pull/994)
- 验证源码：`2f58ee1011a616f0220072d6e88838e4700f87b3`（`r37_pr_alog`）
- 960 合入基线：`5c176bcdcb762e2cfb689cd99e2a9dfe06d56521`；当前 main 基点：`a7c608f2e3f828779ceeec1fcf155a6e37ede83c`
- ATK 交付：[`work_tool !32`](https://gitcode.com/chen-linxin4/work_tool/pull/32)

## CANNBot/PR833 阶段映射

| 阶段 | 本算子交付物 | 本候选状态 |
| --- | --- | --- |
| 01 接口确认 | `README.md`、`docs/api.md`、host 参数检查和 13 槽 ABI | 已绑定 |
| 02 数学标杆 | FP64 truth、FP32-BF16 control、布局/变长/输出掩码合同 | 已冻结并由 ATK executor 复用 |
| 03 直调实现 | A2 `arch22` kernel、六个 runtime key、资源和事件账本 | 已 clean 构建 |
| 04 直调验证 | 小 shape 生命周期、32 模型、泛化合同、dtype 和 DC50 | 已完成对应范围；见 ATK 绑定清单 |
| 05 适配与上库 | Stable ABI、Python wrapper、ATK 独立交付和 PR 门禁 | PR994/!32；MSS 与全量性能仍按报告逐项复核 |

## 1. 目标

融合 Q/K norm、gate prefix、因果 score、完整块三角逆及 W/U，chunk 间不传播状态。
FwdH 和 Finalize 的递推与 attention 输出不属于本算子。A2 只有统一 BF16 resident relay；
必要的归一化、gate、prefix、score 后处理和 Cube 累加保持 FP32。
BF16 输入/relay 不代表全程 FP32 求逆，精度以冻结的 FP64 truth / FP32-BF16 control 比较为准。

使用用户确认的 value-head pair：一个 AIC 的两个 AIV 各负责一个完整 value head，
不改成横切同一 head。此映射沿用已验证的资源、协议和性能方案，是 R01/R20 的本算子既有映射。
开发参考为 CHUNK_INDEPENDENT V1，来源 a78efa9512dfa64c20b2a58a75e84e88a054d694；
采用 grid-stride、resident 生命周期及 ready/free 方法，具体 lane 映射和地址由本算子定义。

性能固定使用实际 32 模型输入，seed3、safe gate、gate-in-kernel、Save，
对照 PR 当前 R26 与相同输入的 A5，目标逐例 A5/A2 ≥ 0.5。
warmup5、repeat50、3 groups，A2 ABBA；Task Duration 决定性能，Event 独立列出。
已验收版本、逐模型结果和实验取舍收录在独立测试交付件，不能由窄合同 fast 的结果替代。
此处不把阶段候选或尚未完成的最终测试标为通过。

## 2. Stage 完整详设

### 2.1 符号、工作域和选择

M=64，D=128，r∈[1,64] 是有效行数，G=HV/HK，pairs=ceil(HV/2)。
Nbase=fixed 时 B×ceil(T/64)，varlen 时为规范 chunk_indices 的记录数。
Nwork=Nbase×pairs；blockDim=min(AIC_count,Nwork)，job=core+n×blockDim。
每个 job 描述一个 chunk 和一对 value heads，slot=n&1。
lane l 的 hv=2×pair+l，active=(hv<HV)，hk=floor(hv/G)。
Q/K 检查点 owner 仅为 hv%G==0；不同 hk 的两个 AIV 各自准备自己的 Q/K。
当前实现不跨 lane 共享 Q/K prepare；同一 hk 时仍独立计算。
例如模型31 B=1、T=131072、HK=HV=4：Nbase=2048、pairs=2、Nwork=4096。
实际 blockDim 取平台 AIC 数，尾 wave 使用同一 grid-stride 和完整协议。

六个 tiling key 为：

|key|norm|gate|
|---|---|---|
|0|Identity|Precomputed|
|1|L2|Precomputed|
|2|Identity|Softplus|
|3|L2|Softplus|
|4|Identity|SafeSigmoid|
|5|L2|SafeSigmoid|

Norm/Gate 为静态策略。Raw/Sigmoid/TwoSigmoid、输出掩码及实际标量是 immutable tiling。
现有 CANN dtype 注册分别生成 g/beta 四种输入对象，每对象仍只有六个数学 key。
不按模型编号、精确 shape、epsilon、scale 或 lower_bound 开优化资格。
A2 host 拦截非 Exp2，内核中的指数域辅助函数不构成未验证合同的公开支持声明。
A5 保留原模板声明、arch35 实现和字段；下面的资源和协议专指 A2。

### 2.2 唯一资源账本

地址常量由 `arch22/chunk_kda_fwd_prepare_resource.h` 同时供 host/device 消费。
每 AIV 一个 0x2f800=190KiB UB，硬件上限192KiB，未增加整套 UB 双缓冲：

|半开字节区间|物理用途|活跃与复用条件|
|---|---|---|
|[0,0x8000)|q_：低16KiB Q，另16KiB临时|Q checkpoint、score payload、RHS 完成消费后用于 Lower|
|[0x8000,0x10000)|k_：低16KiB K，另16KiB临时|同上；不能在 MTE3 未排空前改写|
|[0x10000,0x18000)|g_ FP32 64×128|输入/转换→gate→prefix→score/RHS，下一 work 前退休|
|[0x18000,0x28000)|work_ FP32 64KiB|BF16 g 暂存前16KiB先 Cast 到独立 g_，然后才用于 norm、factor、raw|
|[0x28000,0x2c000)|v_ BF16 16KiB|Kminus→qg_scaled/kg→Aqk，按末次 DMA 消费复用|
|[0x2c000,0x2f000)|small_ 12KiB|rstd、beta、bias、广播量和 scalar DMA 分区|
|[0x2f000,0x2f800)|mask 2KiB|640B strict/inclusive packed triangle，初始化一次，只读|

small_ 内 rstd 为 [0x2c000,0x2c200)，FP32 beta 为 [0x2c200,0x2c300)，
beta DMA [0x2e000,0x2e800)，BF16 pack [0x2e800,0x2e880)。
sequence-major beta 每标量行补到32B，BF16/FP32 pitch分别16/8元素。
FP32 beta 直接保留低位，不经 BF16 中转；FP32 beta_eff 按真实值写回。

每 AIC 的 L1 为192KiB，两 lane 各96KiB。每 lane 相对地址：

|区间|用途|生命周期|
|---|---|---|
|[0,0x12000)|score Q+/K+/分带 K- payload|score MTE1 最后读取完成后释放|
|[0,0x2000)|P，下三角 BF16|score 释放后开始，到最后 merge 读完|
|[0x2000,0x4000)|X，BF16完整逆|eye 初始化→叶子逆→merge→W/U 最后 LoadA|
|[0x4000,0x8000)|Tmp，merge临时及空隙|按连续 GEMM 的 FIX_MTE1 闭环复用|
|[0x8000,0xa000)|Power，叶子幂|叶子迭代期间活跃|
|[0xc000,0x10000)|RHS K 16KiB|scoreReleased 后提前载入，到W/U LoadB结束|
|[0x10000,0x14000)|RHS V 16KiB|同上，不能被下一 work score 覆盖|
|[0x14000,0x16000)|I，BF16 identity|每 lane 首次工作初始化，跨 work 常驻|

score 阶段与逆/RHS阶段的地址重叠有 scoreReleased 事件，表中同时活跃的区域不重叠。
L0A/L0B/L0C 各64KiB；A/B 两 lane 各有32KiB独立 bank，C通过 FIX_M 保护。
W/U 共用一次逆的 LoadA，同时准备两份 RHS 的 LoadB；下一 head bank 装载与当前 MAC 交叠。

GM workspace 每 lane152KiB，record字段如下；每 slot 两lane=304KiB，每core两slot=608KiB：

|相对地址|大小|生产→消费|
|---|---|---|
|0x00000 Payload|72KiB|AIV score payload→AIC score|
|0x12000 Raw|32KiB|AIC FP32 score→AIV Lower/Aqk|
|0x1a000 Lower|8KiB|AIV BF16 strict lower→AIC inverse|
|0x1c000 Eye|8KiB|AIV identity→AIC X/I|
|0x1e000 RhsK|16KiB|AIV当前 RHS→AIC W|
|0x22000 RhsV|16KiB|AIV当前 RHS→AIC U|

lane base=core×0x98000+slot×0x4c000+lane×0x26000。
workspace总量=平台libApiWorkspace+usedCoreNum×0x98000，由host checked multiply/add产生。
关闭的公开输出不充当 scratch。Norm/Gate 六个key使用相同资源上界和事件集合。

### Stage 0：Vector，输入、norm、gate prefix

仅 DMA 有效 r 行，内部矩阵先补零。Identity 保留 Q/K，rstd=1；L2 在 FP32 执行
sum(q²)+epsilon、sqrt、div、mul，再 CAST_RINT 到 BF16 q_hat/k_hat。
仅 owner 写 HK checkpoint；每 HV 独立写 beta_eff。Raw beta 跳过非线性；Sigmoid 为
1/(1+exp(-beta))，TwoSigmoid 在结果上乘2，不修改用户 allow_neg。
g BF16 先扩展到 FP32；FP32 g 直接载入同一 FP32 区。
Precomputed 直接使用 step；Softplus 用 max(x,0)+log1p(exp(-abs(x))) 的补偿实现；
SafeSigmoid 使用 lower_bound/(1+exp(-exp(a_log)×(g+bias)))。
norm和 gate 一直保持各自明确的 FP32 中间量。prefix 按8行段扫描并加carry，仅当前chunk，
不跨序列；Exp2 的域变换沿用当前实现的舍入顺序，不能交换成另一种变换后宣称等价。
gk 在仍活跃时写回，checkpoint DMA 的源在 score payload 消费期间保持不变。

### Stage 1：Vector，四个16行 band 的 score payload

每band选有效区域中点作为 FP32 Gref。Qplus=q_hat×E(G-Gref)，
Kplus=k_hat×E(G-Gref)，Kminus=k_hat×E(Gref-G)，操作数按现有位置转BF16。
第s band的Kminus只保留16(s+1)行，完整payload72KiB。
指数输入按既有base2界[-126,120]限制；padding使用完整64行内部格式，公开输出仅有效行。
本候选保留四band，不混入单锚点或32行band数值实验。
MTE3排空后 q/k/v 的高半区才能复用给当前 RHS，随后发布 score-ready。

### Stage 2：Cube，score

对每个有效lane、每个band分别计算 Qplus @ transpose(Kminus) 和
Kplus @ transpose(Kminus)，形状为[16,128]@[128,16(s+1)]，
BF16 ND/NZ操作数、FP32 MMAD累加，Fixpipe以FP32写Raw。
两lane由同一 AIC 依次消费。最后一次 L1 MTE1读取结束后发布scoreReleased，
Raw实际完成后发布raw-ready，二者职责不同。

### Stage 3：Vector，当前 RHS、公开输出

此Stage在score发布后立即开始，与Stage2交叠，不等待inverse。
FP32中按既有[-80,80]界计算E(G)；kgRHS先将k_hat×E(G)转BF16，再扩展FP32乘beta，
最终BF16舍入到RhsK。RhsV由V在FP32乘beta后转BF16。
qg先BF16舍入，再FP32乘scale并BF16输出qg_scaled；Save在qg活跃时写回。
kg使用最后有效行G[r-1]，不取补零行。两个RHS的内部64行均零填充，写入独立GM区。
最后DMA发布rhs-ready，AIC在scoreReleased后即可提前装入L1。

### Stage 4：Vector，strict Lower 与 Aqk

等待raw-ready及前序work_ Vector消费结束。FP32 score乘scale或-beta，
分别施加inclusive/strict三角mask，再在原舍入点转BF16，生成Aqk及内部P=-beta×score。
640B packed mask一次初始化替代逐work127条按行Duplicate；Select产生相同正零位模式。
不改变乘法、舍入顺序或score精度。Akk在需导出时先有效区域清零，Cube随后填入完整对角和下块。
Lower全64×64写入workspace，padding P=0、inverse diagonal=1。
所有源DMA排空后发布lower-ready，才允许下一work改写Q/K和checkpoint来源。

### Stage 5：Cube，完整叶逆和分层合并

BF16 P、Power、X和I常驻L1。四个16阶叶子通过4轮幂倍增形成完整16阶nilpotent逆，
每次 GEMM 为BF16操作数、FP32累加、Fixpipe BF16舍入回L1；绝非全程FP32的消元。
合并先为16+16，再32+32：Tmp=X_lower @ P_cross，X_cross=Tmp @ X_upper。
每次读取等待FIX_MTE1，复用等待末次MTE1；完整阶数不能替代对BF16每轮舍入的数值验收。
需要公开Akk时，将叶对角与完整跨叶下块在真实活跃期写回；tail仅写有效行。

### Stage 6：Cube，复用 inverse 计算 W/U

从同一 resident X 一次LoadA；W=X @ RhsK，U=X @ RhsV，
[m,m]@[m,128]，m=ceil(r/16)×16，BF16操作数、FP32累加、BF16 Fixpipe输出有效r行。
不同lane的A/B bank独立，当前lane MMAD时预载下一lane。
L1 lane free只在最后LoadA/LoadB完成后发布，不能用GM slot free代替。

### 2.3 事件、复用与退出

跨核flag分别为每slot：score-ready=slot，raw-ready=2+slot，
lower-ready=4+slot，work-free=6+slot，rhs-ready=8+slot。
AIC启动时为两个slot各发布work-free credit；两个AIV统一消费同一事件序列：

```text
AIV: wait work-free → Prepare/ScorePayload → publish score-ready
     → current RHS/output → publish rhs-ready
     → wait raw-ready → Lower/Aqk → publish lower-ready → next work
AIC: wait score-ready → score → scoreReleased → publish raw-ready
     → wait rhs-ready → preload RHS → wait lower-ready
     → inverse/merge → W/U → publish work-free → next work
exit AIV: consume both work-free credits
exit AIC: drain C-free, A/B-free and L1-lane-free, then release local events
```

inactive奇数lane不访存、不写公开输出，但完整参与每个跨核等待和通知。
scoreReleased=MTE1_MTE2，L1 laneFree=MTE1_MTE2，A/B free=M_MTE1，
C free=FIX_M，copyReady=MTE2_MTE1，loadReady=MTE1_M，matrixReady=M_FIX，
resident写回=FIX_MTE1。TPipe按各HardEvent类别分配/释放，ID上限由实际编译及simulator检查。
Vector DMA发布使用MTE3，UB覆盖前显式V_MTE2、MTE2_V、MTE3_V等本地依赖。
work-free保护GM两代record；独立的RHS-ready不会推迟当前RHS到inverse之后。
每core连续多work、slot回卷、最后active/inactive lane和最终credit排空由小shape实际binary捕获验证。
正常性能测量关闭执行记录，不以131k大模型仿真替代小shape生命周期证据。

### 2.4 搬运与数值边界

每HV工作首次读取Q/K/g/beta各一次、V在RHS阶段一次；GVA当前重复读取Q/K，
仅owner写checkpoint。payload、Raw、Lower、RHS各按表中生产/消费一次；I的L1载入每lane首次一次。
RHS与resident inverse不经公开输出中转，完整Save的所有写出成本属于算子性能。
layout只改变输入地址/strided DMA，内部矩阵统一；varlen由序列起止和规范索引限定。
host拒绝非法head比、chunk/K/V、dtype、标量、metadata及DMA stride溢出；
关闭输出仅返回None，不改变内部必须的矩阵生产。
极端gate/beta、norm epsilon、tail padding及高阶逆的数值路径遵守冻结比较器，
失败合同明确标失败，不新增FP32 relay回退族，也不更改truth/control或容差。
