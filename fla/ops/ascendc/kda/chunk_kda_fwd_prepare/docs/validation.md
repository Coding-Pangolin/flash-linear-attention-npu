# Token-major g/beta 迁移验证记录

2026-10-29，开发期记录。A5、MSS 警告和完整 forward 正式验收尚未完成，不能作为最终归档。

## 实际源码与构建

设备验证源码为 `81fc9dac9b3624a3102ed39032803a62ab9668ed`，性能对照为旧合同最优
`1ff4dc0296c21e0054dfc8a264a76f380647e39e`。后续文档、Example 和诊断脚本修改不改变
本次构建的 host/kernel/wrapper，不能把其他未构建的 device 修改绑定到这些结果。

| clean 包 | SHA256 |
| --- | --- |
| A2 Prepare release | `47897c197b16393deb507c0124d6e3f3eba38f035a9fe405e099fcb4105c5047` |
| A2 sanitizer,ccec_g | `103d346284831311e6d7110b8fbafa1826977b72ab7217e2e026edaad9b1c0d1` |
| A2 完整 forward | `be894bbec178bde959bccce5d47b427afc54b0ab049c3c5d2dc832d2b04a452c` |
| A5 Prepare | `400b729983767c91a910666b30303b36866f210d1d7d5fa50ea74aab651f1c37` |

release 命令：`bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu --ops=chunk_kda_fwd_prepare -j24`。
插桩增加 `--op_debug_config sanitizer,ccec_g`；完整 forward 使用 `--ops=chunk_kda_fwd`；
A5 使用 `--soc=ascend950`。四种 g/beta dtype 对象各含六 key，无 ZIP device 对象复用。
Stable C++/schema 90文件相同后复用匹配 ABI 库，其 hash、真实安装和卡号由构建清单单独绑定。

## 合同与原生双标杆

qkv 四 layout；g/beta 固定 token-major，与 qkv layout 无关，分别支持 BF16/FP32。
六个 Norm×Gate key、chunk64、K=V128、13槽及 BF16 relay 不变；没有全量 L2 transpose。
FP32 Precomputed 增加 TwoSum residual，key0 在 UB 内完成 FP32 完整64阶逆后一次舍入至 BF16 relay。
修正前源码63734eb的同输入六key为5/6、76/78，失败key0 Akk/U RMS比2.7638384101477302、
1.520506862403927；修正后13/13及六key6/78通过，没有增加FP32 relay回退族。

ATK26.9.24 原生 `cv_fused_double_benchmark`：FP64 TRUTH / FP32-BF16 CONTROL，
max/avg/RMS比阈值5/1.5/1.5，原生小值域计数默认2不变，不是 `mixed_tolerance_bm`。
独立交付 executor 对冻结版本 AST 核对只有两个 g/beta 输入解析表达式变化，数学、舍入和角色不变。

| 范围 | 用例通过 | 输出通过 |
| --- | --- | --- |
| 原200例通用精度，最大T2048 | 200/200 | 2096/2096 |
| seed3全32模型，FP32 g/BF16 beta、safe gate、Save | 32/32 | 416/416 |
| 六key | 6/6 | 78/78 |
| 数值边界 | 9/9 | 86/86 |
| 布局/dtype小边界 | 16/16 | 208/208 |
| dtype专项 | 24/24 | 220/220 |
| head专项 | 16/16 | 208/208 |

200例原生报告SHA256：`7b939b58f9b1df1752b277ce93ba0b2b725f4667993893576ae72f7f76addded`。
执行配置与公开200例 JSON 的规范化 hash 相同。模型缓存只读复用，g/beta在RAM重排并逆排列断言；
当前NPU DUT重新执行，每例三角色实际输入hash一致，不继承旧合同通过。公开交付件现场生成输入与
双参考，不依赖私人缓存。head交换、尾行篡改、缺失None槽错误注入均检出；同一g/beta四layout输出
逐位相同。可区分旧形状、错误HV、FP16 g/beta在host拒绝，T=HV排列歧义不能靠shape识别。

## DC50 与 MSS

按用户确认，每实际key一例、各六例；不扩成200例。执行与公开六例配置规范化hash一致。

- DC50：六例原生通过，每例50轮。
- initcheck/racecheck：实际key0–5，零ERROR、零WARNING。
- memcheck：零ERROR、56,460条WARNING；独立输出地址观察后逐条审计，全部区间在实际Akk逻辑
  tensor内，零未审计区间。包含关系本身不证明工具误报，保留原日志及ownership审查。
- synccheck：零ERROR、6,900条冗余set/wait WARNING（5,940 set、960 wait），不称零警告通过。

Akk审查：AIV初始化/叶逆store退休后发LowerReady，AIC等待后写叶块或合并；每chunk/value-head
区域独立，GVA QK checkpoint不与Akk别名。未为消除工具警告修改同步或输出指针。

## 配对性能与回退

A2 ABBA=A1/B1/B2/A2，warmup5/repeat50/3groups，每窗口150次，Task/Event分列。
同逻辑输入在API外重排g/beta，实际两端物理hash、逆排列及输出重复性核对，独占监测无外来进程。
32模型首窗口、四模型第二窗口及公开20组合的224个窗口、33,600个Task样本，与原始msprof
`op_summary` 的 `Task Duration(us)` 逐项相同。Event来自同一批profiler调用，另行统计。

| 模型 | 旧Task p50(us) | 候选Task p50(us) |
| --- | --- | --- |
| 1 | 3439.1485 | 3432.8385 |
| 2 | 6744.4750 | 6751.9550 |
| 31 | 3128.2825 | 3140.1430 |
| 32 | 6220.5340 | 6238.4150 |

全32逐例变化−0.1835%～+0.4562%，不宣称亚百分比优化收益，四模型独立第二窗口另列。
公开20组合使用冻结PR32 shape/参数/值域，实际输入hash绑定A/B；不声称相同seed跨ATK/Torch
生成器就有相同tensor。全部T131046，逐例Task/Event及身份收录在独立交付绑定与CSV。

有两项明确回退：key1 L2+Precomputed 3274.74→4145.61us（+26.59%）；key0 Identity+Precomputed
3458.89→5261.93us（+52.13%）。包含必要数值修正成本，不用其他组合平均抵消，也不能回退到未通过
双标杆的路径。第二独立窗口分别为+26.30%、+52.68%，已复现。其余18组合约−0.15%～+0.61%。BSND/TND H4约3379/3380us，与同组BNSD3422us接近。

最早私有组合19使用signed模型g并关闭norm，旧/新DUT均非有限；冻结FP64 truth部分值超过BF16
上界，CONTROL也溢出。保留该失败诊断，不作为通过性能证据；公开20原配方、200精度及阈值未改动。

## Simulator 和调用链

当前release实际metadata/binary/key绑定的小shape simulator：key0、BSND、T1665、HK=HV3。
core0捕获三代work及真实Cube/Vector、Vector/MTE交叠；core13 jobs13/33/53，slot0/1/0，最后一行
有效，lane1 inactive仍完成跨核协议。两次capture退出0、DUT返回、binary未变；模拟微秒不作NPU
性能，也不冒充六key全部核心证明。

完整forward旧/新源码各clean构建，四layout、K64/K128 legacy/强制V2共12例、132可见输出逐位相同。
更新公开Example在BNSD/BSND/NTD三例前向NPU冒烟通过，包含GVA/尾块。反向从保留canonical输入
建立原Head First张量，BWD算子合同未改；本轮未声称反向设备全量验收。

另一个完整forward原生双标杆诊断仍仅2/12例、122/132输出通过，十例失败为W小值域计数。
参考解析不变、旧/新逐位一致只证明迁移不新增差异，不替代端到端正式验收。

## 剩余门槛

A5当前源码clean包已生成，但空闲物理卡1/2/5/6的最小Torch分配/矩阵乘探针均报
`507033 / TsdOpen failed`，发生在调用本算子前；没有重启或停止共享作业/服务。
新合同A5精度和hash匹配性能尚未完成，逐例A5/A2≥0.5无新证据，不继承旧窄合同79.37%。
Precomputed性能回退、MSS警告及完整forward失败均保留；当前整体不能标为最终验收通过。
