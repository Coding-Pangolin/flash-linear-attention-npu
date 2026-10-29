# Token-major g/beta 迁移开发记录

状态：开发中，不能作为最终设备验收。

用户确认 g/beta 固定 token-major，qkv layout 独立。方案变更为 Stage0 的输入地址、DMA stride、beta UB 解包和索引初始化；Stage1–6 的数学、BF16 relay、资源及跨核协议保持既有实现。

CPU 标杆数学/比较器不变，AST 检查确认只有两个输入解析表达式变化。纯 CPU 96 组合覆盖四 layout、六 key、四 dtype，两套参考共比较1824输出；仅6个 control beta_eff 输出因 CPU sigmoid 的连续/strided vector 路径出现最多1.1920928955078125e-7差值，其余输出逐位相同。这是标杆解析核对，不代替原生 NPU 双标杆，也不改变正式容差。

首个独立 Prepare 地址迁移候选 `564a86cd42794be9df926202eb434347ca9f9bee` 已 clean 源编译，
A2 包 SHA256 为 `a8824b9a8afc34bc791ac0e8c8f1974f5a76aff9299ece1f9a88ee8219482ecb`。
原生小例16/208、模型1/2/31/32共4/52通过；head交换、尾行篡改和缺失None槽的错误注入均检出。
四layout复用相同g/beta对象，13槽输出逐位相同；旧形状、错误HV和FP16 g/beta在host拒绝。
首个四模型ABBA窗口的Task变化依次为case1 -0.242%、case31 +0.303%、case2 +0.104%、
case32 +0.494%；仅表明此窗口未见明显回退，不将亚百分比差异宣称为优化收益。

完整 forward/V2 的公共校验、raw g地址及 fused beta adapter 已同步源码，CPU输入解析
等价性32组合/768输出逐位通过；其设备闭环待新构建。独立 Prepare 的上述通过记录不覆盖
本段新增的完整forward修改或后续TwoSum合并，不得宣称端到端可交付。

已知数值问题：基线1ff的部分 FP32 Precomputed gk 前缀未通过冻结标准。TwoSum 独立实验
`bf09ac393911e2a21e693d4a6403658f1efc4cac` 已通过旧合同六key、补充数值及泛化集，但有独立性能成本。
地址迁移的四模型证据建立后，候选源码纳入这项修正；新合同组合后的原生数值与性能仍须重新验证。

最终验证范围：新合同全32/416、紧凑泛化与同规模组合、六 key MSS/DC、实际对象/key 绑定、全成本 ABBA5/50/3及A5匹配证据。资源仍为 AIV UB190KiB、两代 GM slot 与原十个跨核 flag；运行验证待补充。

## 2026-10-29 key0局部64阶求逆实验

组合基线63734ebaf639ca3fe9a4521a3df0ab9589019216已clean构建，新合同六key为5/6、76/78。
失败仅key0的Akk、U，原生RMS比为2.7638384101477302、1.520506862403927，阈值仍1.5。
旧TwoSum源码bf09ac393911e2a21e693d4a6403658f1efc4cac在相同q/k/v/g/beta实际hash上复现相同失败。
CPU逐Stage诊断中，BF16叶逆/合并复现误差；局部FP32完整64阶求逆后一次BF16舍入贴近CONTROL。
CPU诊断只用于定位，不是设备验收，不调整TRUTH/CONTROL/容差。

本实验只改已有key0的局部求逆及其Cube消费，不改变固定token-major合同、六key、BF16 relay或其他数学key。
先用同一key0输入原生比较13槽，再六key及数值9例；通过才晋级dtype/heads/小边界/通用集。
DUT采集按ATK --save_data output保留13槽，直接观察gk、Aqk、Akk、W/U和beta_eff；CPU诊断保持原顺序分离
叶逆/合并与一次BF16舍入。shape/dtype/有效区按spec限定，仅比较有效token，不将padding纳入精度统计。
目标是消除由已定位合并舍入导致的额外误差，原冻结原生双标杆为唯一PASS/FAIL。
资源为既有190KiB UB、192KiB L1及两代workspace；地址/活跃期与PIPE_V、MTE3退休见design_a2 Stage4/5。
clean源码、安装包hash、实际binary/key和NPU精度/性能待本次实验补充；当前不能视作已通过候选。
